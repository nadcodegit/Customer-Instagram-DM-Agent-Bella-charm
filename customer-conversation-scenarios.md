# Bella Charm London — Customer Conversation Scenarios (Complete Scope)

## Important

This document defines the **complete, exhaustive scope** of what the agent
should handle autonomously — there is nothing beyond what's written here.
Any question or request outside these scenarios must NOT be answered by the
agent. Instead it should tell the customer the business owner will
personally follow up, and the conversation should be flagged for human
attention. Do not try to generalize or improvise beyond this scope.

## Scenario 1: Price inquiry → Browse → Cart → Checkout

### Step 1 — Price inquiry

Customer asks the price of a bracelet, charm, watch, or accessory (in any
phrasing).

| Product     | Price |
|-------------|-------|
| Bracelet    | £15   |
| Charm       | £5    |
| Watch       | £25   |
| Accessories | £5    |

### Step 2 — Offer to browse

After answering the price, the agent asks: "Want me to show you the
options?"
- Yes → Step 3
- No → end turn, wait for next message

### Step 3 — Category selection

Show the top-level categories: **Charm / Bracelet / Watch / Accessories**

### Step 4 — Variant selection (depends on chosen category)

**Watch** — 3 options: Gold, Silver, Apple Watch

**Bracelet** — color options: Gold, Silver, Copper, Rose Gold, Pink, Black,
Purple, Brown, Red

**Charm** — subcategories: Flag, Zodiac, Heart, Animal, Birth Month,
Letters. Picking the subcategory is the final selectable item (flat £5
regardless of subcategory) — there is no further design-level choice,
since individual charm designs within a subcategory can number 1000+ and
don't affect price. (Real customer designs go deeper than this -- e.g. a
specific dog breed, or "union jack" -- but matching an exact design is out
of scope for v1; see "Not in v1" below.)

**Accessories** — 1 option: Key (the connector for adding charms onto a
bracelet). Flat £5. Just one item for now, but modeled as a proper
category (not a special case) so more accessories can be added later.

### Step 5 — Add to cart

For whichever specific item the customer lands on, ask: "Want me to add
this to your cart?"
- Yes → add {category, variant, price} to the cart
- No → discard the pending selection

Either way, continue to Step 6.

### Step 6 — Loop: more items?

After a cart decision (added or not), ask: "Want me to see more options to
pick something else?"
- Yes → back to Step 3 (category selection)
- No → show cart summary (items + total price), then Step 7

This is a single repeating question — keep asking it after every item
decision until the customer says they're done, then move on.

### Step 7 — Delivery

Ask the customer directly whether delivery is **UK** or **International**
(not inferred from the address alone), then ask for the delivery address
and cross-check it against the stated destination for consistency.

| Destination   | Delivery fee |
|---------------|--------------|
| UK            | £3           |
| International | £10          |

If the address looks inconsistent with the stated destination, don't guess
— ask the customer to clarify before continuing.

### Step 8 — Final confirmation & payment

Show the full order summary (items + delivery fee + total) and ask for
final confirmation.
- Yes → tell the customer to transfer payment to the business's bank
  account (see "Payment details" below) and that the order ships in 3–5
  business days. Order is now "placed" (payment not yet confirmed).
- No → go back to Step 6 so the customer can adjust their cart.
  *(Assumption — the original scope didn't cover a decline at this step;
  looping back to the cart-adjustment loop seemed like the least-surprising
  behavior. Flag if this should instead go to a human.)*

### After an order is placed

The agent has no visibility into whether the bank transfer actually
happened. So if the same customer (matched by their Instagram
`customer_id`) sends another message after Step 8:
- Ask whether they've already paid for the previous order, or want to add
  more items to it before paying.
- **Not paid yet, adding more** → keep the existing cart, resume browsing
  (back to category selection).
- **Already paid** → clear the cart and start a fresh order from scratch.

## Scenario 2: Store hours / location

Fixed-format answer whenever asked if the shop is open / what hours / where:

- Open daily, **9:00 – 18:00**
- Address: **Inverness St, London NW1 7HJ, United Kingdom**

## Scenario 3: "Can I send proof of payment here?"

Fixed answer whenever a customer asks (in any phrasing) whether they can
send a receipt, payment screenshot, or other proof of payment through the
same DM conversation:

> Yes, just send it here and the owner will see it -- your order will
> then be shipped to you.

This applies both to a fresh message and to a digression mid-flow (e.g.
right after an order is confirmed, or mid-checkout) -- same as Scenario 2.

## Everything else → hand off to the human

Any question or request that isn't Scenario 1 or Scenario 2 — complaints,
wholesale inquiries, custom orders, anything not covered above — the agent
replies that the business owner will personally get back to them, and does
**not** attempt to answer or improvise. This sets `needs_human = true` so
the business owner can see it needs a reply.

### Off-topic messages *in the middle* of Scenario 1

If the customer is mid-flow (e.g. the agent is waiting for a yes/no or a
color choice) and sends something unrelated or unparseable (an unrelated
question, a complaint, etc.), the agent should:
1. Acknowledge it'll pass that along to the business owner for a reply
   (logged for follow-up — this does **not** set the full `needs_human`
   handoff, since the purchase flow keeps going).
2. Re-ask the same pending question so the flow isn't derailed.

## Payment details

Two payment options are offered at final confirmation: bank transfer and
PayPal. Neither is hardcoded into any prompt or source file -- both live
in a local, gitignored secrets file (same pattern as this repo's `.env`),
and only the final-confirmation step reads from it.

## Not in v1 — future idea (parked, not built yet)

Once the website and product photos exist, there's an idea to tag each
photo with a unique code (e.g. `charmAnimal5004`, `watchGolden2001`) and
eventually use a vector database to recommend items based on customer
preference — a much richer version of Step 4's category/variant picking.
This is explicitly **out of scope for v1**: there's no site, no photos, and
no preference data to build it against yet. It's a natural candidate for a
v2 phase (and a good next LangGraph learning exercise — retrieval/tool
nodes) once the underlying data exists. The `cart` item shape in v1 leaves
room for this later (see architecture notes) without being built out now.
