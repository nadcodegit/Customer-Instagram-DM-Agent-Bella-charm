"""LangGraph graph for the Bella Charm London DM agent.

Every run processes exactly one incoming customer message and ends with a
single `draft_reply` -- there's no looping *inside* one run. The Step
3/6 "browse more" loop, and everything else that spans multiple DMs,
happens *across* separate runs via the persisted `current_step` (see
state.py) and the checkpointer (see build_graph below).
"""

import os
from typing import Literal

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .constants import (
    CATEGORY_PLURALS,
    CATEGORY_PRICES,
    CATEGORY_VARIANTS,
    STORE_ADDRESS,
    STORE_HOURS,
)
from .payment import load_payment_details
from .state import ConversationState, Step, cart_line, cart_total
from .step_config import STEP_CONFIG

MODEL_NAME = os.environ.get("BELLA_AGENT_MODEL", "openai/gpt-oss-120b")

OWNER_HANDOFF_REPLY = (
    "Thanks so much for reaching out! That's something Bella Charm "
    "London's owner will need to get back to you on personally -- I've "
    "flagged your message for her."
)


def _llm() -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=0)


def _last_message_text(state: ConversationState) -> str:
    return state["messages"][-1].content


def _finalize_reply(update: dict) -> dict:
    """Every draft-producing return passes through here (see the end of
    handle_new_request / handle_step_answer / handle_post_order_start):
    defaults needs_human to False so a stale True from an earlier turn
    never silently carries forward onto an unrelated, well-handled reply
    -- LangGraph only replaces keys a node actually returns, so a node
    that forgets to mention needs_human would otherwise leave whatever
    was already in state untouched. Only one place sets it True
    explicitly (_t_final_confirm's "Yes" branch -- the purchase
    confirmation carrying bank/PayPal details, the one reply where a
    wrong LLM output has real financial consequences), and that explicit
    value always wins here since setdefault only fills in a *missing*
    key. An out-of-scope ("other") message is logged to owner_followups
    for her to follow up on personally, but -- like everything else --
    sends its fixed, pre-approved holding reply automatically; it no
    longer blocks on review.
    """
    update.setdefault("needs_human", False)
    return update


# ---------------------------------------------------------------------------
# Entry routing (pure Python, no LLM -- current_step alone decides this)
# ---------------------------------------------------------------------------


def entry_router(
    state: ConversationState,
) -> Literal["handle_new_request", "handle_step_answer", "handle_post_order_start"]:
    if state["current_step"] == "start":
        return "handle_new_request"
    if state["current_step"] == "order_placed":
        return "handle_post_order_start"
    return "handle_step_answer"


# ---------------------------------------------------------------------------
# current_step == "start": classify a fresh message into one of the two
# scenarios, or hand off to the owner.
# ---------------------------------------------------------------------------


class NewRequestClassification(BaseModel):
    intent: Literal["price_inquiry", "store_hours", "greeting", "payment_proof", "other"]
    category: Literal["Bracelet", "Charm", "Watch", "Accessories"] | None = Field(
        default=None,
        description=(
            "Only for price_inquiry, when the customer named a specific "
            "category. Null if they asked about prices generally."
        ),
    )


_CLASSIFY_INSTRUCTIONS = (
    "Classify this Instagram DM to a handmade jewelry business (Bella "
    "Charm London). intent is one of: 'price_inquiry' (asking about, or "
    "showing interest in, the price or availability of a bracelet, charm, "
    "watch, or accessory (e.g. the 'key' connector for adding charms to a "
    "bracelet) -- any phrasing counts, e.g. 'do you have a monkey charm?', "
    "'do you sell watches?', or a plain typo-ridden question, not just "
    "literal 'how much' questions), 'store_hours' (asking if the shop is "
    "open, hours, or location), 'greeting' (a plain greeting or small talk "
    "with no other content, e.g. 'hi', 'hello', 'hey there'), "
    "'payment_proof' (asking whether they can send a receipt, payment "
    "screenshot, or other proof of payment via this same DM), or 'other' "
    "(anything else -- complaints, wholesale, custom orders, unrelated "
    "questions).\n\n"
)

PAYMENT_PROOF_REPLY = (
    "Yes, just send it here and the owner will see it -- your order will "
    "then be shipped to you."
)

GREETING_REPLY = (
    "Hi there! Thanks for messaging Bella Charm London. Feel free to ask "
    "about our bracelets, charms, or watches, or our opening hours."
)


def _classify_new_request(state: ConversationState) -> NewRequestClassification:
    classifier = _llm().with_structured_output(NewRequestClassification)
    return classifier.invoke(_CLASSIFY_INSTRUCTIONS + f"Message: {_last_message_text(state)}")


def _price_answer(category: str | None) -> str:
    if category:
        return f"Our {CATEGORY_PLURALS[category]} are £{CATEGORY_PRICES[category]}."
    lines = "\n".join(f"- {cat}: £{price}" for cat, price in CATEGORY_PRICES.items())
    return f"Here are our prices:\n{lines}"


def handle_new_request(state: ConversationState) -> dict:
    result = _classify_new_request(state)

    if result.intent == "store_hours":
        return _finalize_reply({"draft_reply": f"{STORE_HOURS}\nAddress: {STORE_ADDRESS}"})

    if result.intent == "greeting":
        return _finalize_reply({"draft_reply": GREETING_REPLY})

    if result.intent == "payment_proof":
        return _finalize_reply({"draft_reply": PAYMENT_PROOF_REPLY})

    if result.intent == "other":
        return _finalize_reply(
            {
                "draft_reply": OWNER_HANDOFF_REPLY,
                "owner_followups": state["owner_followups"] + [_last_message_text(state)],
            }
        )

    return _finalize_reply(
        {
            "draft_reply": f"{_price_answer(result.category)} Want me to show you the options?",
            "current_step": "awaiting_browse_offer",
            # Remember the category they already named (if any) so saying
            # "yes" a moment later can skip straight past re-asking it.
            "pending_selection": {"category": result.category} if result.category else {},
        }
    )


# ---------------------------------------------------------------------------
# Mid-flow: match the reply against the pending step's options (via
# step_config.py) and apply the resulting transition.
# ---------------------------------------------------------------------------


class StepAnswerMatch(BaseModel):
    matched_option: str | None = Field(
        description=(
            "The exact string, verbatim from the provided valid options, "
            "that the customer's message answers -- or null if the message "
            "doesn't answer the pending question at all."
        )
    )


def _match_step_answer(
    state: ConversationState, options: list[str], question: str
) -> StepAnswerMatch:
    matcher = _llm().with_structured_output(StepAnswerMatch)
    result: StepAnswerMatch = matcher.invoke(
        "You're mid-conversation with a customer of Bella Charm London, a "
        "handmade jewelry business, on Instagram DM. They were just asked:\n"
        f'"{question}"\n'
        f"Valid answers: {options}\n\n"
        f'Their reply: "{_last_message_text(state)}"\n\n'
        "Match their reply to one of the valid answers if it's a reasonable "
        "interpretation of their underlying intent -- not just an exact or "
        "literal match. Customers often answer indirectly (e.g. naming a "
        "product instead of literally saying 'yes', or only addressing "
        "part of a multi-part question). Use judgment about what they "
        "most likely mean in context, and return the exact matching option "
        "string verbatim. But don't force it: if they're describing "
        "something genuinely different from every listed option (e.g. "
        "saying they'll pick the order up in person when the choices are "
        "'UK'/'International' delivery -- that's neither), leave "
        "matched_option null rather than guessing the closest-sounding one."
    )
    if result.matched_option not in options:
        # Guards against the model returning something outside the allowed
        # set despite instructions -- treat that as no match.
        return StepAnswerMatch(matched_option=None)
    return result


def _handle_off_topic_message(state: ConversationState, resume_question: str) -> dict:
    """Called whenever a message doesn't answer the pending question.

    Before assuming it's out of scope, check whether it's actually a valid
    Scenario 1/2 request in disguise -- customers routinely bring up a
    different product, or ask about hours, in the middle of another flow
    (e.g. "want to see more options?" / "how much are bracelets?"), and
    that shouldn't just get punted to the owner. Only escalate when the
    message is genuinely outside scope. Either way, nothing about the
    in-progress flow (cart, current_step, ...) changes -- we answer (or
    escalate) inline, then re-ask the original pending question so it
    picks back up exactly where it left off.
    """
    result = _classify_new_request(state)

    if result.intent == "store_hours":
        answer = f"{STORE_HOURS}\nAddress: {STORE_ADDRESS}"
    elif result.intent == "price_inquiry":
        answer = _price_answer(result.category)
    elif result.intent == "greeting":
        answer = GREETING_REPLY
    elif result.intent == "payment_proof":
        answer = PAYMENT_PROOF_REPLY
    else:
        return {
            "draft_reply": f"I'll pass that along to the owner so she can get back to you on it. {resume_question}",
            "owner_followups": state["owner_followups"] + [_last_message_text(state)],
        }

    return {"draft_reply": f"{answer}\n\n{resume_question}"}


def _t_browse_offer(state: ConversationState, answer: str) -> dict:
    if answer == "Yes":
        return {
            "current_step": "awaiting_category",
            "draft_reply": STEP_CONFIG["awaiting_category"].question(state),
        }
    return {
        "current_step": "start",
        "draft_reply": "No problem, just let me know if you'd like anything else!",
    }


class BrowseOfferAnswer(BaseModel):
    matched: bool = Field(
        description="Whether the reply addresses wanting to see the options at all (yes or no), even indirectly."
    )
    wants_to_browse: bool = Field(
        default=False,
        description="True if they want to see the options. Only meaningful when matched is true.",
    )
    category: Literal["Bracelet", "Charm", "Watch", "Accessories"] | None = Field(
        default=None,
        description=(
            "If they named a specific category while answering (e.g. "
            "'yes, gold one'), capture it here even if a category was "
            "already known from earlier in the conversation."
        ),
    )
    variant: str | None = Field(
        default=None,
        description=(
            "If they also named one of that category's specific variants "
            "in the same message, capture it here too."
        ),
    )


def _handle_browse_offer(state: ConversationState) -> dict:
    """Special-cased like the other multi-slot steps: a category may
    already be known from the original message that started this flow
    (see handle_new_request, which stashes it in pending_selection) -- a
    plain "yes" shouldn't force the customer to repeat a category they
    already named. This also extracts a *newly* named category/variant
    from the "yes" message itself, so both sources feed the same
    fast-forward logic.
    """
    question = STEP_CONFIG["awaiting_browse_offer"].question(state)
    extractor = _llm().with_structured_output(BrowseOfferAnswer)
    result: BrowseOfferAnswer = extractor.invoke(
        f'A customer was just asked: "{question}"\n'
        f'Their reply: "{_last_message_text(state)}"\n\n'
        "Determine whether they want to see the options (yes/no). If they "
        "also named a specific category (bracelet, charm, or watch), "
        "and/or one of that category's specific variants, in the same "
        "message, capture those too -- exactly as spelled here:\n"
        f"{_variants_hint()}\n\n"
        "If their reply doesn't address this at all, set matched to false."
    )

    if not result.matched:
        return _handle_off_topic_message(state, question)
    if not result.wants_to_browse:
        return _t_browse_offer(state, "No")

    category = result.category or state["pending_selection"].get("category")
    if category:
        return _advance_with_category_and_variant(state, category, result.variant)
    return _t_browse_offer(state, "Yes")


def _t_category(state: ConversationState, answer: str) -> dict:
    pending = {"category": answer}
    peek_state = {**state, "pending_selection": pending}
    return {
        "pending_selection": pending,
        "current_step": "awaiting_variant",
        "draft_reply": STEP_CONFIG["awaiting_variant"].question(peek_state),
    }


def _t_variant(state: ConversationState, answer: str) -> dict:
    pending = {**state["pending_selection"], "variant": answer}
    peek_state = {**state, "pending_selection": pending}
    return {
        "pending_selection": pending,
        "current_step": "awaiting_add_to_cart_confirm",
        "draft_reply": STEP_CONFIG["awaiting_add_to_cart_confirm"].question(peek_state),
    }


def _advance_with_category_and_variant(
    state: ConversationState, category: str, variant: str | None
) -> dict:
    """Jump as far forward as the extracted info supports: straight to
    add-to-cart confirmation if a *valid* variant for that category is also
    known, otherwise just to variant selection with the category pre-filled.
    Shared by every step that might pick up a category+variant together
    (awaiting_category itself, and the more-items digression) so a customer
    who names both in one message ("heart charm please") doesn't have to
    repeat themselves when asked "which one?" a moment later.
    """
    if variant and variant in CATEGORY_VARIANTS.get(category, []):
        return _t_variant({**state, "pending_selection": {"category": category}}, variant)
    return _t_category(state, category)


def _variants_hint() -> str:
    """Rendered so an extraction prompt can name concrete variant strings
    to look for -- without this, a model tends to leave a freeform variant
    field empty even when the customer did name one, since it has nothing
    to match against."""
    return "\n".join(f"- {cat}: {', '.join(opts)}" for cat, opts in CATEGORY_VARIANTS.items())


class CategoryAnswer(BaseModel):
    matched: bool = Field(
        description="Whether the reply names one of the three categories (Bracelet, Charm, Watch) at all, even indirectly."
    )
    category: Literal["Bracelet", "Charm", "Watch", "Accessories"] | None = Field(
        default=None, description="Only meaningful when matched is true."
    )
    variant: str | None = Field(
        default=None,
        description=(
            "If they also named a specific variant/subcategory within that "
            "category in the same message (e.g. a color for Bracelet or "
            "Watch, or 'heart'/'animal'/etc for Charm), capture it here."
        ),
    )


def _handle_category(state: ConversationState) -> dict:
    question = STEP_CONFIG["awaiting_category"].question(state)
    extractor = _llm().with_structured_output(CategoryAnswer)
    result: CategoryAnswer = extractor.invoke(
        f'A customer was just asked: "{question}"\n'
        f'Their reply: "{_last_message_text(state)}"\n\n'
        "Determine which category (Bracelet, Charm, or Watch) they mean. "
        "If they also named one of that category's specific variants in "
        "the same message, capture it too -- exactly as spelled here:\n"
        f"{_variants_hint()}\n\n"
        "If their reply doesn't name a category at all, set matched to false."
    )
    if not result.matched or not result.category:
        return _handle_off_topic_message(state, question)
    return _advance_with_category_and_variant(state, result.category, result.variant)


def _t_more_items(state: ConversationState, answer: str) -> dict:
    if answer == "Yes":
        return {
            "current_step": "awaiting_category",
            "draft_reply": STEP_CONFIG["awaiting_category"].question(state),
        }
    if not state["cart"]:
        # Nothing to check out -- send them back to pick something instead
        # of proceeding to a delivery/payment flow for an empty order.
        return {
            "current_step": "awaiting_category",
            "draft_reply": (
                "Looks like your cart's empty so far! "
                + STEP_CONFIG["awaiting_category"].question(state)
            ),
        }
    lines = "\n".join(cart_line(i) for i in state["cart"])
    summary = f"Here's your cart:\n{lines}\nTotal so far: £{cart_total(state['cart'])}\n\n"
    return {
        "current_step": "awaiting_delivery_destination",
        "draft_reply": summary + STEP_CONFIG["awaiting_delivery_destination"].question(state),
    }


class MoreItemsAnswer(BaseModel):
    matched: bool = Field(
        description="Whether the reply addresses wanting to keep browsing at all (yes or no), even indirectly."
    )
    wants_more: bool = Field(
        default=False,
        description="True if they want to keep browsing. Only meaningful when matched is true.",
    )
    category: Literal["Bracelet", "Charm", "Watch", "Accessories"] | None = Field(
        default=None,
        description=(
            "If they named a specific category while answering (e.g. "
            "'yes, do you have heart charms?'), capture it here so the "
            "flow can jump straight to it instead of asking again."
        ),
    )
    variant: str | None = Field(
        default=None,
        description=(
            "If they also named a specific variant/subcategory within that "
            "category in the same message (e.g. 'yes, heart charm please'), "
            "capture it here too."
        ),
    )


def _handle_more_items(state: ConversationState) -> dict:
    question = STEP_CONFIG["awaiting_more_items"].question(state)
    extractor = _llm().with_structured_output(MoreItemsAnswer)
    result: MoreItemsAnswer = extractor.invoke(
        f'A customer was just asked: "{question}"\n'
        f'Their reply: "{_last_message_text(state)}"\n\n'
        "Determine whether they want to keep browsing (yes/no). If they "
        "also named a specific category (bracelet, charm, or watch) while "
        "answering, capture it. If they also named one of that category's "
        "specific variants in the same message, capture that too -- "
        "exactly as spelled here:\n"
        f"{_variants_hint()}\n\n"
        "If their reply doesn't address this at all, set matched to false."
    )

    if not result.matched:
        return _handle_off_topic_message(state, question)
    if not result.wants_more:
        return _t_more_items(state, "No")
    if result.category:
        return _advance_with_category_and_variant(state, result.category, result.variant)
    return _t_more_items(state, "Yes")


def _t_delivery_destination(state: ConversationState, answer: str) -> dict:
    peek_state = {**state, "delivery_destination": answer}
    return {
        "delivery_destination": answer,
        "current_step": "awaiting_delivery_address",
        "draft_reply": STEP_CONFIG["awaiting_delivery_address"].question(peek_state),
    }


def _t_final_confirm(state: ConversationState, answer: str) -> dict:
    if answer == "Yes":
        reply = (
            "Great, your order is confirmed! You can pay by bank transfer "
            "or PayPal:\n"
            f"{load_payment_details()}\n\n"
            "Your order will ship in 3-5 business days."
        )
        # The one reply with real financial consequences if the LLM got
        # something wrong upstream (cart total, bank details) -- the only
        # step that still waits for a human to look before it goes out.
        return {"current_step": "order_placed", "draft_reply": reply, "needs_human": True}
    return {
        "current_step": "awaiting_more_items",
        "draft_reply": "No problem! " + STEP_CONFIG["awaiting_more_items"].question(state),
    }


def _t_payment_reconciliation(state: ConversationState, answer: str) -> dict:
    if answer == "Already paid":
        return {
            "cart": [],
            "delivery_destination": None,
            "delivery_address": None,
            "current_step": "start",
            "draft_reply": "Great, thank you! Let me know what you'd like to look at next.",
        }
    return {
        "current_step": "awaiting_category",
        "draft_reply": "No problem! " + STEP_CONFIG["awaiting_category"].question(state),
    }


_TRANSITIONS = {
    "awaiting_variant": _t_variant,
    "awaiting_delivery_destination": _t_delivery_destination,
    "awaiting_final_confirm": _t_final_confirm,
    "awaiting_payment_reconciliation": _t_payment_reconciliation,
}


class AddToCartAnswer(BaseModel):
    matched: bool = Field(
        description="Whether the reply addresses adding this item to the cart at all (yes or no), even indirectly."
    )
    add_to_cart: bool = Field(
        default=False,
        description="True if they want it added. Only meaningful when matched is true.",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        description="How many of this item they want, if a number was mentioned. Default 1.",
    )
    alternative_category: Literal["Bracelet", "Charm", "Watch", "Accessories"] | None = Field(
        default=None,
        description=(
            "Only when declining (add_to_cart is false): if they named a "
            "different category they want instead (e.g. 'no, I want rose "
            "charm' -> Charm), capture it here. Null if they didn't ask "
            "for anything else, or if it's the same category as the item "
            "just offered."
        ),
    )
    alternative_variant: str | None = Field(
        default=None,
        description=(
            "Only when declining: if they also named a specific variant "
            "within that category, capture it here too -- exactly as "
            "spelled here (same list as the category prompt uses):\n"
        ),
    )


def _handle_add_to_cart(state: ConversationState) -> dict:
    question = STEP_CONFIG["awaiting_add_to_cart_confirm"].question(state)
    extractor = _llm().with_structured_output(AddToCartAnswer)
    result: AddToCartAnswer = extractor.invoke(
        f'A customer was just asked: "{question}"\n'
        f'Their reply: "{_last_message_text(state)}"\n\n'
        "Determine whether they want this item added to their cart, and "
        "how many (default 1 if they didn't mention a number). If they "
        "decline but name something else they'd rather have instead, "
        "capture that too -- valid categories and variants:\n"
        f"{_variants_hint()}\n\n"
        "If their reply doesn't address adding-or-not at all, set matched "
        "to false."
    )

    if not result.matched:
        return _handle_off_topic_message(state, question)

    if not result.add_to_cart and (result.alternative_category or result.alternative_variant):
        category = result.alternative_category or state["pending_selection"]["category"]
        return _advance_with_category_and_variant(state, category, result.alternative_variant)

    cart = list(state["cart"])
    reply_prefix = ""
    if result.add_to_cart:
        category = state["pending_selection"]["category"]
        variant = state["pending_selection"]["variant"]
        cart.append(
            {
                "category": category,
                "variant": variant,
                "price": CATEGORY_PRICES[category],
                "quantity": result.quantity,
                "tag": None,
            }
        )
        qty_note = f" x{result.quantity}" if result.quantity > 1 else ""
        reply_prefix = f"Added to your cart{qty_note}! "

    peek_state = {**state, "cart": cart}
    return {
        "cart": cart,
        "pending_selection": {},
        "current_step": "awaiting_more_items",
        "draft_reply": reply_prefix + STEP_CONFIG["awaiting_more_items"].question(peek_state),
    }


class AddressExtraction(BaseModel):
    contains_address: bool = Field(
        description="Whether the message contains a delivery address at all."
    )
    address: str | None = Field(default=None)
    consistent_with_destination: bool = Field(
        default=True,
        description=(
            "Only meaningful when contains_address is true: whether the "
            "address plausibly matches the stated delivery destination."
        ),
    )


def _handle_delivery_address(state: ConversationState) -> dict:
    destination = state["delivery_destination"]
    question = STEP_CONFIG["awaiting_delivery_address"].question(state)
    extractor = _llm().with_structured_output(AddressExtraction)
    result: AddressExtraction = extractor.invoke(
        f"The customer said their delivery destination is: {destination}.\n"
        f'Their message: "{_last_message_text(state)}"\n\n'
        "Extract the delivery address they gave (if any), and judge whether "
        "it's plausibly consistent with the stated destination."
    )

    if not result.contains_address:
        return _handle_off_topic_message(state, question)

    if not result.consistent_with_destination:
        return {
            "draft_reply": (
                f"Just double-checking -- you said delivery is {destination}, "
                f'but the address you gave ("{result.address}") looks '
                "different. Could you confirm the address or destination?"
            )
        }

    peek_state = {**state, "delivery_address": result.address}
    return {
        "delivery_address": result.address,
        "current_step": "awaiting_final_confirm",
        "draft_reply": STEP_CONFIG["awaiting_final_confirm"].question(peek_state),
    }


def handle_step_answer(state: ConversationState) -> dict:
    step: Step = state["current_step"]

    if step == "awaiting_delivery_address":
        result = _handle_delivery_address(state)
    elif step == "awaiting_add_to_cart_confirm":
        result = _handle_add_to_cart(state)
    elif step == "awaiting_more_items":
        result = _handle_more_items(state)
    elif step == "awaiting_category":
        result = _handle_category(state)
    elif step == "awaiting_browse_offer":
        result = _handle_browse_offer(state)
    else:
        config = STEP_CONFIG[step]
        options = config.options(state)
        question = config.question(state)
        match = _match_step_answer(state, options, question)
        if match.matched_option is None:
            result = _handle_off_topic_message(state, question)
        else:
            result = _TRANSITIONS[step](state, match.matched_option)

    return _finalize_reply(result)


# ---------------------------------------------------------------------------
# current_step == "order_placed": don't know if the customer has paid yet,
# so ask before doing anything else.
# ---------------------------------------------------------------------------


def handle_post_order_start(state: ConversationState) -> dict:
    """A message came in after an order was placed -- unlike every other
    step, there's no pending question this is answering, so it can't be
    "off-topic" in the usual sense. Still reuse the same
    answer-if-answerable, escalate-otherwise logic (via
    _handle_off_topic_message) rather than blindly overwriting whatever
    the customer just said with the reconciliation question -- otherwise a
    real question (e.g. "want me to send the receipt here?") gets
    silently ignored.
    """
    reconciliation_question = STEP_CONFIG["awaiting_payment_reconciliation"].question(state)
    result = _handle_off_topic_message(state, reconciliation_question)
    return _finalize_reply({**result, "current_step": "awaiting_payment_reconciliation"})


# ---------------------------------------------------------------------------
# Human review: every draft_reply passes through here before the run ends.
# ---------------------------------------------------------------------------


def await_owner_approval(state: ConversationState) -> dict:
    """Almost everything (needs_human False -- price/cart/checkout, store
    hours, payment-proof, and now the "other" holding reply too) sends
    automatically, with nobody in the loop. Only _t_final_confirm's "Yes"
    branch sets needs_human True -- the purchase confirmation carrying
    bank/PayPal details -- and pauses the graph via `interrupt()`, handing
    the draft to whoever is reviewing it (see runner.py's
    resolve_pending_review). The checkpointed state sits frozen here
    until she resumes it with a decision. Nothing is ever sent to the
    customer before this returns `approved: True`.
    """
    if not state["needs_human"]:
        return {"approved": True}

    decision = interrupt(
        {
            "customer_id": state["customer_id"],
            "customer_message": _last_message_text(state),
            "draft_reply": state["draft_reply"],
            "needs_human": state["needs_human"],
        }
    )
    if decision["action"] == "edit":
        return {"draft_reply": decision["text"], "approved": True}
    if decision["action"] == "reject":
        return {"approved": False}
    return {"approved": True}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """Compile the graph. `checkpointer` persists state per customer_id
    (used as the LangGraph thread_id) across separate .invoke() calls --
    without it, every incoming DM would be treated as a brand-new
    conversation with no memory of where the flow left off, and
    `interrupt()`/`Command(resume=...)` (see await_owner_approval) would
    have nothing to pause and resume. Defaults to an in-memory
    checkpointer, fine for local/mock runs.
    """
    graph = StateGraph(ConversationState)
    graph.add_node("handle_new_request", handle_new_request)
    graph.add_node("handle_step_answer", handle_step_answer)
    graph.add_node("handle_post_order_start", handle_post_order_start)
    graph.add_node("await_owner_approval", await_owner_approval)

    graph.add_conditional_edges(
        START,
        entry_router,
        {
            "handle_new_request": "handle_new_request",
            "handle_step_answer": "handle_step_answer",
            "handle_post_order_start": "handle_post_order_start",
        },
    )
    graph.add_edge("handle_new_request", "await_owner_approval")
    graph.add_edge("handle_step_answer", "await_owner_approval")
    graph.add_edge("handle_post_order_start", "await_owner_approval")
    graph.add_edge("await_owner_approval", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
