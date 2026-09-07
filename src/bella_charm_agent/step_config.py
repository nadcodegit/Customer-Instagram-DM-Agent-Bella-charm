"""Per-step question text and valid-answer options for the Scenario 1 flow.

`handle_step_answer` (graph.py, not built yet) will use this to know what
question to (re-)ask for the current step, and what set of answers an LLM
call should try to match the customer's reply against. A step with
`options=None` is free text -- only `awaiting_delivery_address` right now --
and gets matched with custom logic instead of a fixed-choice comparison.

"start" and "order_placed" aren't here: they're each handled by their own
graph node and don't have a fixed question to re-ask on an off-topic reply.
"""

from dataclasses import dataclass
from typing import Callable

from .constants import CATEGORY_PRICES, CATEGORY_SINGULAR, CATEGORY_VARIANTS, DELIVERY_FEES
from .state import ConversationState, Step, cart_line, cart_total

YES_NO = ["Yes", "No"]


@dataclass(frozen=True)
class StepConfig:
    # None means free text -- no fixed set of valid answers to match against.
    options: Callable[[ConversationState], list[str]] | None
    question: Callable[[ConversationState], str]


def _static(options: list[str]) -> Callable[[ConversationState], list[str]]:
    return lambda state: options


def _fixed(question: str) -> Callable[[ConversationState], str]:
    return lambda state: question


def _variant_options(state: ConversationState) -> list[str]:
    category = state["pending_selection"]["category"]
    return CATEGORY_VARIANTS[category]


def _variant_question(state: ConversationState) -> str:
    category = state["pending_selection"]["category"]
    options = ", ".join(CATEGORY_VARIANTS[category])
    return f"Which {CATEGORY_SINGULAR[category]} would you like: {options}?"


def _add_to_cart_question(state: ConversationState) -> str:
    category = state["pending_selection"]["category"]
    variant = state["pending_selection"]["variant"]
    price = CATEGORY_PRICES[category]
    return f"Want me to add the {variant} {CATEGORY_SINGULAR[category]} (£{price}) to your cart?"


def _final_confirm_question(state: ConversationState) -> str:
    lines = [cart_line(item) for item in state["cart"]]
    delivery_fee = DELIVERY_FEES[state["delivery_destination"]]
    total = cart_total(state["cart"]) + delivery_fee
    return (
        "Here's your order:\n"
        + "\n".join(lines)
        + f"\nDelivery ({state['delivery_destination']}): £{delivery_fee}\n"
        f"Total: £{total}\n\n"
        "Shall I confirm this order?"
    )


STEP_CONFIG: dict[Step, StepConfig] = {
    "awaiting_browse_offer": StepConfig(
        # Handled specially in graph.py (also extracts an optional named
        # category/variant, not just yes/no) -- options=None marks it as
        # not using the generic matcher, same convention as the address step.
        options=None,
        question=_fixed("Want me to show you the options?"),
    ),
    "awaiting_category": StepConfig(
        # Handled specially in graph.py (also extracts an optional named
        # variant, not just the category) -- options=None marks it as not
        # using the generic matcher, same convention as the address step.
        options=None,
        # Built from CATEGORY_VARIANTS' keys, not hardcoded -- a hardcoded
        # list here already went stale once (a new "Accessories" category
        # was missing from it after this doc was written).
        question=_fixed(
            "Which would you like to look at: "
            f"{', '.join(CATEGORY_VARIANTS.keys())}?"
        ),
    ),
    "awaiting_variant": StepConfig(
        options=_variant_options,
        question=_variant_question,
    ),
    "awaiting_add_to_cart_confirm": StepConfig(
        # Handled specially in graph.py (also needs to extract an optional
        # quantity, not just yes/no) -- options=None just marks it as not
        # using the generic matcher, same convention as the address step.
        options=None,
        question=_add_to_cart_question,
    ),
    "awaiting_more_items": StepConfig(
        # Handled specially in graph.py (also extracts an optional named
        # category, not just yes/no) -- options=None marks it as not using
        # the generic matcher, same convention as the address step.
        options=None,
        question=_fixed("Want to see more options to pick something else?"),
    ),
    "awaiting_delivery_destination": StepConfig(
        options=_static(["UK", "International"]),
        question=_fixed("Is this delivery within the UK, or international?"),
    ),
    "awaiting_delivery_address": StepConfig(
        options=None,
        question=_fixed("What's the delivery address?"),
    ),
    "awaiting_final_confirm": StepConfig(
        options=_static(YES_NO),
        question=_final_confirm_question,
    ),
    "awaiting_payment_reconciliation": StepConfig(
        options=_static(["Not paid yet - add more items", "Already paid"]),
        question=_fixed(
            "Before we continue -- have you already sent payment for your "
            "previous order, or would you like to add more items to it "
            "first?"
        ),
    ),
}
