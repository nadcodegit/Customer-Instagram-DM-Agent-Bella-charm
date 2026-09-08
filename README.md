# Bella Charm London — Instagram DM Agent

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent that reads
incoming Instagram DMs for **Bella Charm London**
([@bellacharmlondon](https://www.instagram.com/bellacharmlondon/)), a
handmade Italian charm bracelet stall in Camden, London. Most replies —
price/browse/cart/checkout, store hours, the proof-of-payment answer, and
even a genuinely out-of-scope message — send automatically, since the
owner doesn't have time to review every DM and gating everything on her
just recreates the delay this agent exists to remove. The one exception
is the final purchase confirmation (the message carrying bank/PayPal
details and the order total): that pauses via LangGraph's `interrupt()`
for the owner to approve, edit, or reject before anything goes out,
since it's the one reply where a wrong output has real financial
consequences. Built as a hands-on project to learn LangGraph (state,
nodes, conditional edges, checkpointer/memory, structured tool-calling,
human-in-the-loop interrupts) against a real small business, not a toy
example.

## What it handles

Full scope is defined in
[customer-conversation-scenarios.md](customer-conversation-scenarios.md).
In short:

- **Price / browse / cart / checkout** — price questions for
  bracelets, charms, watches, or accessories; browsing categories and
  variants; adding items (with quantity) to a cart; delivery
  destination and address; final confirmation with payment details
  (bank transfer or PayPal).
- **Store hours / location** — fixed answer.
- **"Can I send proof of payment here?"** — fixed answer.
- **Everything else** (complaints, wholesale, custom orders, specific
  charm designs we don't catalog, unrelated questions) — a fixed holding
  reply ("the owner will get back to you personally") sends
  automatically, and the raw message is logged to `owner_followups` for
  her to follow up on in her own time. Not improvised, and not held up
  waiting for her either.

A customer can jump ahead in the flow (name a category *and* variant in
one message, decline an item but name a different one instead, answer a
yes/no question indirectly) and the agent fast-forwards as far as it
validly can, rather than forcing a rigid one-question-at-a-time script.

## Architecture

```
                (pure Python -- current_step decides this)
                        entry_router
                    /        |        \
   handle_new_request  handle_step_answer  handle_post_order_start
   (fresh message,      (mid-flow: match/     (order placed, don't know
    current_step        extract against the    if paid yet -- ask before
    == "start")          pending question)      anything else)
                    \        |        /
                  await_owner_approval
              (auto-approves unless needs_human is
               True -- only the final purchase
               confirmation sets that; see below)
                             |
                            END
```

- **State** (`state.py`) — a `TypedDict` carrying the conversation
  history, `current_step`, the item being built up (`pending_selection`),
  the `cart`, delivery info, and flags for the owner. `messages` uses the
  `add_messages` reducer (append, not overwrite); every other field is
  replaced outright by whatever a node returns for it.
- **Nodes** (`graph.py`) — each LLM call is a Pydantic
  `with_structured_output` schema, not free text: a classifier for fresh
  messages, a per-step answer matcher/extractor for most steps, and a
  few steps (`awaiting_category`, `awaiting_more_items`,
  `awaiting_add_to_cart_confirm`, `awaiting_browse_offer`,
  `awaiting_delivery_address`) get their own handler because they need to
  extract more than a plain yes/no.
- **Conditional edges** — `entry_router` picks the first node with no
  LLM call at all, just `current_step`.
- **Checkpointer** — `build_graph()` uses a `MemorySaver` keyed by
  `customer_id` (the LangGraph `thread_id`), so the flow picks up where
  it left off across separate incoming DMs.
- **Off-topic handling** — if a message doesn't answer the pending
  question, it's re-classified (same classifier as a fresh message)
  before assuming it's out of scope: an answerable digression (a
  different product, store hours, a greeting) gets answered inline and
  the original question is re-asked; a genuinely out-of-scope message
  gets its fixed holding reply and is logged to `owner_followups`
  instead, same as a fresh "other" message.

### Human review (`await_owner_approval`)

Every reply passes through one more node before the run ends. For almost
everything, `needs_human` is `False` and it sends immediately, no one in
the loop. The one exception is `_t_final_confirm`'s "Yes" branch — the
message carrying the bank/PayPal details and the order total — which sets
`needs_human = True`. There, `await_owner_approval` calls LangGraph's
`interrupt()`, which pauses the whole graph run and hands the draft (plus
the customer's last message and context) to whoever is reviewing it.
Execution — and the checkpointed state — sits frozen there until it's
resumed with one of three decisions:

- `{"action": "approve"}` — sends the draft as-is
- `{"action": "edit", "text": "..."}` — sends the edited text instead
- `{"action": "reject"}` — sends nothing

`runner.py` splits this into two functions to match the two separate
triggers now possible: `submit_customer_message()` runs the graph up to
the pause (or straight through, if nothing needs review), and
`resolve_pending_review()` resumes it once a decision comes in. This is
also where real Instagram sending will eventually plug in: once
`resolve_pending_review` returns `approved: True`, that's the point where
a future webhook adapter would call the actual Instagram Send API instead
of just returning the text.

If another message arrives from the same customer while a review is
still pending, `submit_customer_message` does **not** run an independent
turn on it — that used to silently orphan the pending review, since
there was nothing left to resume once a second, unrelated turn had
already completed. Instead it appends the new message to the paused
conversation via LangGraph's `update_state()` (no node executes), so the
owner sees the *full* accumulated context — not just the first message —
whenever she gets to it.

## Project structure

```
src/bella_charm_agent/
  state.py         ConversationState, CartItem, cart helpers
  constants.py     prices, categories/variants, store hours/address
  step_config.py   per-step question text + valid-answer options
  payment.py       reads bank/PayPal details from a local secrets file
  graph.py         nodes, LLM schemas, conditional edges, build_graph()
  runner.py        submit_customer_message() / resolve_pending_review()
                   / list_pending_reviews() + an interactive terminal
                   chat demo
  web.py           the owner-facing review dashboard (FastAPI)
tests/
  test_transitions.py       pure FSM transition functions (no LLM)
  test_cart_helpers.py      cart_line / cart_total
  test_llm_nodes.py         LLM-calling nodes, with the LLM mocked
  test_interrupt_flow.py    the review pause/resume, at the graph level
  test_runner_queuing.py    a message arriving while a review is pending
  test_sqlite_persistence.py  state survives a simulated process restart
  test_web.py               the review dashboard's routes (FastAPI TestClient)
  test_llm_quality_manual.py  opt-in: same tricky cases against the *real* model
  test_step_config.py       structural check (every step has a config entry)
  test_payment.py           env-var vs. local-file precedence for payment details
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
cp .env.example .env              # fill in GROQ_API_KEY
cp config/payment_secrets.example.json config/payment_secrets.json  # fill in real details
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).
Groq's model lineup changes over time — if `BELLA_AGENT_MODEL` 404s, check
[console.groq.com/docs/models](https://console.groq.com/docs/models) for a
current alternative.

Conversation state (including any pending review) is persisted with a
`SqliteSaver`, not an in-memory checkpointer — it survives a restart.
The database file defaults to `data/conversations.sqlite` (gitignored,
created automatically); override with `BELLA_AGENT_DB_PATH` to point it
somewhere else (e.g. a mounted volume in a real deployment).

Both secrets work the same way: a real deployment sets them as actual
environment variables (in the hosting platform's own secrets panel,
never in a file that ships with the code); locally, `GROQ_API_KEY` comes
from `.env` and the bank/PayPal details from
`config/payment_secrets.json`, same as before. For payment details
specifically, six `PAYMENT_*` environment variables
(`PAYMENT_BANK_NAME`, `PAYMENT_ACCOUNT_NAME`, `PAYMENT_SORT_CODE`,
`PAYMENT_ACCOUNT_NUMBER`, `PAYMENT_IBAN`, `PAYMENT_PAYPAL`) take
priority over the file whenever any of them are set -- see
`payment.py`.

## Run

```bash
uv run python -m bella_charm_agent.runner
```

Chats with one fixed `customer_id` in your terminal, playing both the
customer and the owner. Most replies print immediately as `[AUTO-SENT]`;
the final purchase confirmation instead prompts you, as the owner, to
approve/edit/reject before it's marked sent. Nothing is ever actually
sent anywhere — it's all local. Because state is persisted (see Setup),
you can quit (Ctrl+C) mid-conversation and pick it back up next run —
including a still-pending review.

### Web app (webhook + review dashboard)

```bash
uv run uvicorn bella_charm_agent.web:app --reload
```

This is the single process meant to actually be deployed -- everything
lives in one FastAPI app (`web.py`):

- `POST /webhook` -- where an incoming DM arrives. For now it accepts a
  **simulated** payload (`{"customer_id": "...", "text": "..."}`), not
  Meta's real Instagram webhook format -- that's blocked on the
  business's Meta access token, not on anything here.
  `_extract_incoming_message` is the one function that knows the
  payload's shape, so swapping in the real format later means changing
  that function (and the Pydantic model above it), not the route, not
  `runner.py`, not `graph.py`.
- `GET /` -- the review dashboard. Lists every conversation currently
  paused for review (reads the same `data/conversations.sqlite` the CLI
  writes to, so a pending review created via `/webhook` or the CLI shows
  up here either way), each with an Approve / edit-then-send / Reject
  form. This is meant to be exactly what the business owner uses day to
  day -- there's deliberately no way to simulate an incoming customer
  message from this page; that stays a developer-only tool (the CLI
  above, or `POST /webhook` directly), to keep "what she needs" and
  "what I need to test with" separate.

Try it locally without a real webhook:

```bash
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "test_customer", "text": "how much is a charm?"}'
```

Delivering the final reply (auto-sent or approved) is also a stub for
now -- `_deliver_to_customer` just prints, until a real Instagram Send
API call can replace it once the access token arrives. Neither route has
authentication yet; fine for localhost, needed before this is deployed
anywhere reachable.

## Test

```bash
uv run pytest              # fast, deterministic, mocked LLM -- no API calls
uv run pytest -m live_llm  # slower, hits the real Groq API, needs GROQ_API_KEY
```

The `live_llm` suite isn't run by default — it's a set of regression
checks for specific real-world phrasings that tripped up the model during
manual testing (e.g. an indirect variant description, a customer naming a
category and variant in one message). Re-run it after changing a prompt
in `graph.py` or after Groq changes the configured model.

## Known limitations (see "Not in v1" in the scope doc)

- No per-design catalog: individual charm designs (a specific dog breed,
  a specific zodiac sign, "union jack") aren't tracked, only the
  subcategory. Asking about one gets a generic answer, not a real yes/no.
- No image understanding: customers who send photos (common in practice)
  get no special handling.
- No live Instagram integration yet — `/webhook` accepts a simulated
  payload, not Meta's real format, and replies are never actually
  delivered (`_deliver_to_customer` just prints). Both are isolated,
  small changes once the business's Meta access token arrives — see
  `web.py`.
- Neither `/webhook` nor the review dashboard has authentication yet —
  fine while this only runs on localhost; needs at least a shared
  password before it's deployed anywhere reachable.
- Not deployed anywhere yet — runs locally only, as a single `uvicorn`
  process (`web.py`). Railway (or similar) is the next step.
