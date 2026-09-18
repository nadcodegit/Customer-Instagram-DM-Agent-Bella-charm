"""The single deployable app: the incoming-DM webhook and the owner's
review dashboard, served together so the whole thing runs as one process.

Run with:
    uv run uvicorn bella_charm_agent.web:app --reload

--- Webhook (incoming DMs) ---

`GET /webhook` handles Meta's verification handshake (done once, when
the Callback URL is saved in the Meta dashboard); `POST /webhook` is
what Meta calls on every incoming event afterwards.
`_extract_incoming_message` is the *only* place that understands the
incoming payload's shape -- it accepts both Meta's real Instagram
messaging format and the simulated {"customer_id", "text"} payload
still used for local testing (runner.py's CLI, POST /webhook directly,
the test suite). Swapping/extending the real format later means
changing that one function, not the route, not runner.py, not
graph.py.

--- Review dashboard ---

Lists every conversation currently paused for review, each with
Approve / edit-then-send / Reject forms -- plain HTML, no JavaScript.
This is meant to be exactly what the business owner eventually uses, so
it only shows what's relevant to her. Simulating an incoming customer
message stays a developer-only tool (see runner.py's CLI, or POST
/webhook directly), not something she should see a button for.

--- Delivering the final reply ---

`_deliver_to_customer` calls Instagram's Send API for real when
META_ACCESS_TOKEN and META_IG_USER_ID are set; falls back to logging
only (the old stub behavior) otherwise, so local dev/tests still work
without real Meta credentials. Everything upstream already treats
"sent"/"approved" as final, so nothing else needed to change here.

--- Auth ---

The dashboard carries bank details and customer messages, so it's
gated behind HTTP Basic Auth (DASHBOARD_USERNAME / DASHBOARD_PASSWORD,
required -- this module refuses to import without them set, so it's
never possible to accidentally deploy it unprotected). `/webhook` is
deliberately left unauthenticated: Meta doesn't authenticate a webhook
with HTTP Basic Auth -- that's the verify token at subscription time
(GET /webhook) plus, going forward, a per-request signature header this
module doesn't check yet (see README's known limitations).
"""

import html
import os
import secrets

import httpx
import sentry_sdk
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

from . import observability
from .runner import list_pending_reviews, resolve_pending_review, submit_customer_message

observability.configure()
logger = observability.logger

app = FastAPI(title="Bella Charm London")

_DASHBOARD_USERNAME = os.environ.get("DASHBOARD_USERNAME")
_DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")
if not _DASHBOARD_USERNAME or not _DASHBOARD_PASSWORD:
    raise RuntimeError(
        "DASHBOARD_USERNAME and DASHBOARD_PASSWORD must both be set before "
        "this app can start -- the review dashboard carries bank details "
        "and customer messages, and must never be reachable without a "
        "password. Set them in .env for local dev, or your hosting "
        "platform's secrets panel for a real deployment."
    )

# All three optional -- unset locally, where real Instagram delivery/
# verification isn't needed (or possible). A real deployment sets all
# three once the business's Meta app has a token (see OWNER_GUIDE.md /
# README for where each of these comes from).
_META_ACCESS_TOKEN = os.environ.get("META_ACCESS_TOKEN")
_META_IG_USER_ID = os.environ.get("META_IG_USER_ID")
_META_VERIFY_TOKEN = os.environ.get("META_VERIFY_TOKEN")

_security = HTTPBasic()


def _require_owner(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    # constant-time comparisons -- a naive `==` leaks how many leading
    # characters matched via how long the comparison took.
    correct_username = secrets.compare_digest(credentials.username, _DASHBOARD_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, _DASHBOARD_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


def _deliver_to_customer(customer_id: str, text: str) -> None:
    """Real Instagram Send API call when credentials are configured;
    otherwise the old log-only stub, so local dev/tests never need
    real Meta credentials just to exercise the rest of the flow."""
    if not (_META_ACCESS_TOKEN and _META_IG_USER_ID):
        logger.info("Would send to %s: %s", customer_id, text)
        return

    try:
        response = httpx.post(
            f"https://graph.instagram.com/v25.0/{_META_IG_USER_ID}/messages",
            headers={"Authorization": f"Bearer {_META_ACCESS_TOKEN}"},
            json={"recipient": {"id": customer_id}, "message": {"text": text}},
            timeout=10,
        )
        response.raise_for_status()
    except Exception:
        # The graph/state already treat this reply as sent by this
        # point -- delivery failing here means the customer never
        # actually got it, so this must never pass silently even
        # though (see below) it also must not fail the webhook request.
        logger.exception("Failed to deliver Instagram message to %s", customer_id)
        sentry_sdk.capture_exception()


# ---------------------------------------------------------------------------
# Webhook (incoming DMs)
# ---------------------------------------------------------------------------


class SimulatedDMPayload(BaseModel):
    """Stand-in for Meta's real webhook payload -- see the module
    docstring. Deliberately just the two things submit_customer_message
    already needs."""

    customer_id: str
    text: str


def _extract_incoming_message(payload: dict) -> tuple[str, str] | None:
    """(customer_id, message_text) from the incoming payload, or None if
    there's nothing worth processing.

    Understands two shapes:
    - Meta's real Instagram messaging webhook (`{"object": "instagram",
      "entry": [{"messaging": [...]}]}`). The same webhook also delivers
      events this function must ignore rather than mistake for a
      customer message: echoes of our *own* sent replies (`is_echo`,
      since Meta reflects those back for multi-surface consistency --
      treating one as incoming would feed the bot's own bank-details
      reply back into itself), and non-message events (reactions, read
      receipts, postbacks) that have no `message.text` at all.
    - The simulated {"customer_id", "text"} payload used for local
      testing (runner.py's CLI, POST /webhook directly, the test
      suite).
    """
    if payload.get("object") == "instagram":
        try:
            messaging = payload["entry"][0]["messaging"][0]
        except (KeyError, IndexError):
            return None
        message = messaging.get("message", {})
        if message.get("is_echo") or "text" not in message:
            return None
        return messaging["sender"]["id"], message["text"]

    simulated = SimulatedDMPayload.model_validate(payload)
    return simulated.customer_id, simulated.text


@app.get("/webhook", response_class=PlainTextResponse)
def verify_webhook(request: Request) -> str:
    """Meta's one-time verification handshake, sent when the Callback
    URL + Verify token are saved in the app's webhook settings. Must
    echo back hub.challenge verbatim; anything else and Meta refuses
    to save the subscription."""
    params = request.query_params
    token_ok = _META_VERIFY_TOKEN and secrets.compare_digest(
        params.get("hub.verify_token", ""), _META_VERIFY_TOKEN
    )
    if params.get("hub.mode") == "subscribe" and token_ok:
        return params.get("hub.challenge", "")
    raise HTTPException(status_code=403, detail="Webhook verification failed")


@app.post("/webhook")
def webhook(payload: dict) -> dict:
    try:
        extracted = _extract_incoming_message(payload)
    except Exception:
        logger.exception("Failed to parse incoming webhook payload: %r", payload)
        sentry_sdk.capture_exception()
        raise HTTPException(status_code=400, detail="Unrecognized webhook payload.")

    if extracted is None:
        return {"status": "ignored"}
    customer_id, text = extracted

    try:
        result = submit_customer_message(customer_id, text)
    except Exception:
        # A customer's message failed to process (e.g. Groq is down or
        # times out) -- log the full traceback and report it to Sentry
        # so this doesn't just sit unnoticed in Railway's log stream,
        # then fail the request without leaking internals to the caller.
        logger.exception("Failed to process message from %s", customer_id)
        sentry_sdk.capture_exception()
        raise HTTPException(
            status_code=500,
            detail="Something went wrong processing this message.",
        )
    if result["status"] == "sent":
        _deliver_to_customer(customer_id, result["draft_reply"])
    return result


# ---------------------------------------------------------------------------
# Review dashboard
# ---------------------------------------------------------------------------

_PAGE = """<!doctype html>
<html>
<head>
  <title>Bella Charm London -- Pending Reviews</title>
  <style>
    body {{ font-family: sans-serif; max-width: 700px; margin: 2rem auto; padding: 0 1rem; }}
    .card {{ border: 1px solid #ccc; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
    .draft {{ white-space: pre-wrap; background: #f7f7f7; padding: 0.5rem; border-radius: 4px; }}
    textarea {{ width: 100%; min-height: 5rem; box-sizing: border-box; font-family: inherit; }}
    button {{ margin: 0.75rem 0.5rem 0 0; padding: 0.5rem 1rem; }}
  </style>
</head>
<body>
  <h1>Pending Reviews</h1>
  {cards}
</body>
</html>
"""

_CARD = """<div class="card">
  <p><strong>Customer said:</strong> {customer_message}</p>
  <p><strong>Draft reply:</strong></p>
  <div class="draft">{draft_reply}</div>
  <form method="post" action="/resolve/{customer_id}">
    <textarea name="edited_text">{draft_reply}</textarea>
    <button type="submit" name="action" value="approve">Approve &amp; send</button>
    <button type="submit" name="action" value="edit">Send edited text</button>
    <button type="submit" name="action" value="reject">Reject</button>
  </form>
</div>
"""


def _render_dashboard() -> str:
    pending = list_pending_reviews()
    cards = "\n".join(
        _CARD.format(
            customer_id=html.escape(review["customer_id"]),
            customer_message=html.escape(review["customer_message"]),
            draft_reply=html.escape(review["draft_reply"]),
        )
        for review in pending
    )
    return _PAGE.format(cards=cards or "<p>No pending reviews right now.</p>")


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(_require_owner)])
def dashboard() -> str:
    return _render_dashboard()


@app.post("/resolve/{customer_id}", dependencies=[Depends(_require_owner)])
def resolve(customer_id: str, action: str = Form(...), edited_text: str = Form("")) -> RedirectResponse:
    if action == "edit":
        decision = {"action": "edit", "text": edited_text}
    elif action == "reject":
        decision = {"action": "reject"}
    else:
        decision = {"action": "approve"}
    result = resolve_pending_review(customer_id, decision)
    if result.get("approved"):
        _deliver_to_customer(customer_id, result["draft_reply"])
    return RedirectResponse(url="/", status_code=303)
