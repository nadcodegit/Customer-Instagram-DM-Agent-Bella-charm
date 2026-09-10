# Bella Charm London — Review Dashboard: Owner's Guide

This is a plain-language guide for using the review dashboard day to
day. If you want the technical details instead, see
[README.md](README.md).

## What this is

An assistant that reads incoming Instagram DMs and handles the common,
repetitive questions for you automatically — prices, opening hours,
browsing the catalog, adding items to a cart, checkout. For anything
involving your bank/PayPal details, or anything it isn't sure how to
answer, it stops and waits for you.

## Where to find it

Dashboard: https://customer-instagram-dm-agent-bella-charm-production.up.railway.app

It's password-protected — ask Nadereh for the username and password if
you don't have them. Bookmark the link once you've logged in once.

## What you'll see

Every time you open the dashboard, it shows a card for each
conversation currently waiting on you. Each card has:

- **Customer said:** the customer's message
- **Draft reply:** what the assistant is proposing to send
- A text box (pre-filled with the draft) and three buttons

## What to do with each one

- **Approve & send** — the draft looks right as-is. Sends it exactly
  as written.
- **Send edited text** — change anything in the text box first (fix a
  typo, adjust the wording, correct a detail), then click this. It
  sends whatever is currently in the box, not the original draft.
- **Reject** — don't send anything for this one. Use this if the
  situation needs a reply you'll write yourself outside the assistant
  entirely.

There's no "leave it for later" button — every card needs one of the
three actions before it disappears from the list. If you're not ready
to decide, just close the tab and come back later; the card will
still be there.

## When a card shows up

Only when a customer has said "yes" to a final order — the message
would include your bank transfer and PayPal details, so that's the
one moment a human always checks before anything goes out. Simple
things (prices, hours, "can I pay this way", browsing, adding to
cart) are answered automatically and won't show up here at all — you
won't see those, they're already sent.

## What isn't live yet

This is not yet connected to your real Instagram DMs — that's waiting
on an access token from Meta, which is a separate step on your end.
Until that's connected, nothing here is reaching real customers
automatically; the whole flow has only been tested with simulated
messages. Once it's connected, this same dashboard and these same
three buttons are what you'll use for real.

## If something looks wrong

If a draft reply looks confusing, wrong, or you're just not sure —
don't approve it. Reject it (or just leave the tab open and message
Nadereh) rather than guessing. It's much easier to fix a reply that
never got sent than one that already went to a customer.
