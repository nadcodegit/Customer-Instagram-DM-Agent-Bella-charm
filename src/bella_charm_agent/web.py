"""The single deployable app: the incoming-DM webhook and the owner's
review dashboard, served together so the whole thing runs as one process.

Run with:
    uv run uvicorn bella_charm_agent.web:app --reload

--- Webhook (incoming DMs) ---

`/webhook` accepts a simulated payload -- Meta's real Instagram webhook
JSON isn't wired up yet (blocked on the business's Meta access token, not
on anything here). `_extract_incoming_message` is the *only* place that
understands the incoming payload's shape; swapping in the real Meta
format later means changing that one function (and the Pydantic model
above it), not the route, not runner.py, not graph.py.

--- Review dashboard ---

Lists every conversation currently paused for review, each with
Approve / edit-then-send / Reject forms -- plain HTML, no JavaScript.
This is meant to be exactly what the business owner eventually uses, so
it only shows what's relevant to her. Simulating an incoming customer
message stays a developer-only tool (see runner.py's CLI, or POST
/webhook directly), not something she should see a button for.

--- Delivering the final reply ---

Also blocked on the same Meta access token: `_deliver_to_customer` is
the one place a real Instagram Send API call will go once it's
available. Everything upstream already treats "sent"/"approved" as
final, so nothing else will need to change when that arrives.

--- Auth ---

The dashboard carries bank details and customer messages, so it's
gated behind HTTP Basic Auth (DASHBOARD_USERNAME / DASHBOARD_PASSWORD,
required -- this module refuses to import without them set, so it's
never possible to accidentally deploy it unprotected). `/webhook` is
deliberately left unauthenticated: it's meant to be called by Meta, not
a browser, and HTTP Basic Auth isn't how Meta authenticates a webhook
anyway (that's a verify token at subscription time plus a signature
header on each request -- both arrive with the real webhook format).
"""

import html
import os
import secrets

import sentry_sdk
from fastapi import Depends, FastAPI, Form, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from fastapi.responses import HTMLResponse, RedirectResponse

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
    """Stub: logs instead of actually sending, until a real Instagram
    Send API call can replace this body."""
    logger.info("Would send to %s: %s", customer_id, text)


# ---------------------------------------------------------------------------
# Webhook (incoming DMs)
# ---------------------------------------------------------------------------


class SimulatedDMPayload(BaseModel):
    """Stand-in for Meta's real webhook payload -- see the module
    docstring. Deliberately just the two things submit_customer_message
    already needs."""

    customer_id: str
    text: str


def _extract_incoming_message(payload: SimulatedDMPayload) -> tuple[str, str]:
    """(customer_id, message_text) from the incoming payload. The one
    function that needs to change when the real Meta webhook format
    replaces this simulated one."""
    return payload.customer_id, payload.text


@app.post("/webhook")
def webhook(payload: SimulatedDMPayload) -> dict:
    customer_id, text = _extract_incoming_message(payload)
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
