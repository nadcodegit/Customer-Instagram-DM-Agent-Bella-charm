# Bella Charm London — Instagram DM Agent

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent that reads
incoming Instagram DMs for **Bella Charm London**
([@bellacharmlondon](https://www.instagram.com/bellacharmlondon/)), a
handmade Italian charm bracelet stall in Camden, London, and drafts
replies for the owner to review and send — it never sends anything
itself. Built as a hands-on project to learn LangGraph (state, nodes,
conditional edges, checkpointer/memory, structured tool-calling) against
a real small business, not a toy example.

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
  charm designs we don't catalog, unrelated questions) — flagged for
  the owner, not improvised.

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
  gets flagged for the owner instead.

## Project structure

```
src/bella_charm_agent/
  state.py         ConversationState, CartItem, cart helpers
  constants.py     prices, categories/variants, store hours/address
  step_config.py   per-step question text + valid-answer options
  payment.py       reads bank/PayPal details from a local secrets file
  graph.py         nodes, LLM schemas, conditional edges, build_graph()
  runner.py        send_message() + an interactive terminal chat demo
tests/
  test_transitions.py       pure FSM transition functions (no LLM)
  test_cart_helpers.py      cart_line / cart_total
  test_llm_nodes.py         LLM-calling nodes, with the LLM mocked
  test_llm_quality_manual.py  opt-in: same tricky cases against the *real* model
  test_step_config.py       structural check (every step has a config entry)
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

## Run

```bash
uv run python -m bella_charm_agent.runner
```

Chats with one fixed `customer_id` in your terminal, printing each draft
reply (nothing is ever sent anywhere).

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
- No live Instagram integration yet — this runs against a simulated DM
  input (`runner.py`), not the real Meta/Instagram Messaging API.
