"""Tests for the pure FSM transition functions in graph.py.

None of these touch the LLM -- each `_t_*` function just takes an
already-matched answer string and returns the resulting state update, so
they're plain deterministic functions.
"""

from bella_charm_agent import graph


def test_entry_router_start_goes_to_new_request(make_state):
    state = make_state(current_step="start")
    assert graph.entry_router(state) == "handle_new_request"


def test_entry_router_order_placed_goes_to_post_order_start(make_state):
    state = make_state(current_step="order_placed")
    assert graph.entry_router(state) == "handle_post_order_start"


def test_entry_router_mid_flow_goes_to_step_answer(make_state):
    state = make_state(current_step="awaiting_category")
    assert graph.entry_router(state) == "handle_step_answer"


def test_browse_offer_yes_moves_to_category(make_state):
    result = graph._t_browse_offer(make_state(), "Yes")
    assert result["current_step"] == "awaiting_category"


def test_browse_offer_no_returns_to_start(make_state):
    result = graph._t_browse_offer(make_state(), "No")
    assert result["current_step"] == "start"


def test_category_records_pending_selection_and_asks_variant(make_state):
    result = graph._t_category(make_state(), "Bracelet")
    assert result["pending_selection"] == {"category": "Bracelet"}
    assert result["current_step"] == "awaiting_variant"
    assert "Gold" in result["draft_reply"]


def test_variant_adds_to_pending_selection(make_state):
    state = make_state(pending_selection={"category": "Bracelet"})
    result = graph._t_variant(state, "Rose Gold")
    assert result["pending_selection"] == {"category": "Bracelet", "variant": "Rose Gold"}
    assert result["current_step"] == "awaiting_add_to_cart_confirm"


def test_more_items_yes_returns_to_category(make_state):
    result = graph._t_more_items(make_state(), "Yes")
    assert result["current_step"] == "awaiting_category"


def test_more_items_no_summarizes_cart_with_quantity_math(make_state):
    state = make_state(
        cart=[{"category": "Charm", "variant": "Heart", "price": 5, "quantity": 2, "tag": None}]
    )
    result = graph._t_more_items(state, "No")
    assert result["current_step"] == "awaiting_delivery_destination"
    assert "x2" in result["draft_reply"]
    assert "£10" in result["draft_reply"]  # 5 * 2, not just 5


def test_more_items_no_with_empty_cart_asks_for_a_selection_instead_of_checking_out(make_state):
    # Guards against a customer being sent to delivery/payment for an
    # order with nothing in it (e.g. after declining every item offered).
    result = graph._t_more_items(make_state(cart=[]), "No")
    assert result["current_step"] == "awaiting_category"


def test_delivery_destination_records_choice(make_state):
    result = graph._t_delivery_destination(make_state(), "UK")
    assert result["delivery_destination"] == "UK"
    assert result["current_step"] == "awaiting_delivery_address"


def test_final_confirm_yes_places_order_and_includes_payment_details(make_state, monkeypatch):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    result = graph._t_final_confirm(make_state(), "Yes")
    assert result["current_step"] == "order_placed"
    assert "Sort Code: 000000" in result["draft_reply"]


def test_final_confirm_no_returns_to_more_items(make_state):
    result = graph._t_final_confirm(make_state(), "No")
    assert result["current_step"] == "awaiting_more_items"


def test_payment_reconciliation_already_paid_clears_cart_and_resets(make_state):
    state = make_state(
        cart=[{"category": "Charm", "variant": "Heart", "price": 5, "quantity": 1, "tag": None}]
    )
    result = graph._t_payment_reconciliation(state, "Already paid")
    assert result["cart"] == []
    assert result["current_step"] == "start"


def test_payment_reconciliation_not_paid_leaves_cart_untouched_and_resumes_browsing(make_state):
    state = make_state(
        cart=[{"category": "Charm", "variant": "Heart", "price": 5, "quantity": 1, "tag": None}]
    )
    result = graph._t_payment_reconciliation(state, "Not paid yet - add more items")
    # No "cart" key in the update at all -- the existing cart is meant to
    # survive untouched, not be explicitly copied back.
    assert "cart" not in result
    assert result["current_step"] == "awaiting_category"


# handle_post_order_start now classifies the incoming message (to answer or
# escalate before asking reconciliation) -- see test_llm_nodes.py, since
# that needs the LLM mocked rather than being a pure function.
