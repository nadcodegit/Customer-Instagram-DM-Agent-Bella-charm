"""Conversation state for the Bella Charm London LangGraph agent.

See customer-conversation-scenarios.md for the business logic this state
is shaped around.
"""

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

from .constants import CATEGORY_SINGULAR

# Every point the FSM can be waiting at. "start" means no order in
# progress -- the next message gets classified fresh (price / hours / other).
Step = Literal[
    "start",
    "awaiting_browse_offer",
    "awaiting_category",
    "awaiting_variant",
    "awaiting_add_to_cart_confirm",
    "awaiting_more_items",
    "awaiting_delivery_destination",
    "awaiting_delivery_address",
    "awaiting_final_confirm",
    "order_placed",
    "awaiting_payment_reconciliation",
]


class CartItem(TypedDict):
    category: str
    variant: str
    price: int  # per-unit price
    quantity: int
    # A customer-specified design/color/pattern detail, freeform text --
    # only ever set for Charm (see AddToCartAnswer.detail in graph.py),
    # since a Charm subcategory like "Animal" or "Claddagh" is a broad
    # bucket, not a specific item. None for every other category, and for
    # Charm items where the customer had no particular preference. Also
    # the natural home for the future photo-tag / vector-search idea (see
    # "Not in v1" in customer-conversation-scenarios.md) if that's ever
    # built -- this freeform text is a stopgap for the same need.
    tag: str | None


def cart_line(item: CartItem) -> str:
    suffix = f" x{item['quantity']}" if item["quantity"] > 1 else ""
    singular = CATEGORY_SINGULAR[item["category"]]
    # tag carries a customer-requested design/color/pattern detail (Charm
    # only, see graph.py's AddToCartAnswer) -- it must show up here, since
    # this is what the owner actually reads to fulfill the order, not just
    # a field that's captured and never surfaced anywhere.
    detail = f" ({item['tag']})" if item.get("tag") else ""
    return f"- {item['variant']} {singular}{detail}{suffix} (£{item['price'] * item['quantity']})"


def cart_total(cart: list[CartItem]) -> int:
    return sum(item["price"] * item["quantity"] for item in cart)


class ConversationState(TypedDict):
    # `add_messages` is a reducer: node returns get *appended* to this list
    # instead of replacing it, which is how LangGraph keeps conversation
    # history across turns. Every other field below has no reducer, so a
    # node's return value for that key replaces it outright.
    messages: Annotated[list, add_messages]
    customer_id: str
    current_step: Step
    pending_selection: dict
    cart: list[CartItem]
    delivery_destination: Literal["UK", "International"] | None
    delivery_address: str | None
    draft_reply: str
    needs_human: bool
    owner_followups: list[str]
    # Set by await_owner_approval once the owner has reviewed draft_reply
    # for this turn: True to actually send it, False if rejected. None
    # means no decision has been made yet for the current draft.
    approved: bool | None


def new_conversation_state(customer_id: str) -> ConversationState:
    """The starting state for a customer_id with no prior history."""
    return ConversationState(
        messages=[],
        customer_id=customer_id,
        current_step="start",
        pending_selection={},
        cart=[],
        delivery_destination=None,
        delivery_address=None,
        draft_reply="",
        needs_human=False,
        owner_followups=[],
        approved=None,
    )
