"""A minimal owner-facing review dashboard: list pending reviews, and
approve/edit/reject each one. No JavaScript -- plain HTML forms, since
there's nothing here that needs it.

This is meant to be exactly what the business owner eventually uses, so
it only shows what's relevant to her. Simulating an incoming customer
message stays in runner.py's CLI on purpose, not here -- that's a
developer testing tool, not something she should see a button for.

Run with:
    uv run uvicorn bella_charm_agent.web:app --reload
"""

import html

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from .runner import list_pending_reviews, resolve_pending_review

app = FastAPI(title="Bella Charm London -- Pending Reviews")

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


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _render_dashboard()


@app.post("/resolve/{customer_id}")
def resolve(customer_id: str, action: str = Form(...), edited_text: str = Form("")) -> RedirectResponse:
    if action == "edit":
        decision = {"action": "edit", "text": edited_text}
    elif action == "reject":
        decision = {"action": "reject"}
    else:
        decision = {"action": "approve"}
    resolve_pending_review(customer_id, decision)
    return RedirectResponse(url="/", status_code=303)
