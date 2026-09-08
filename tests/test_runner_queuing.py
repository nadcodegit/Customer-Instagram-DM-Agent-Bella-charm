"""Tests for the queuing behavior in runner.py's submit_customer_message:
a second message arriving while a review is already pending on the same
thread must not run independently -- that's exactly how a pending review
got silently lost during manual testing (a "girlfriend's birthday"
message escalated and paused, a second message ran the graph fresh, and
the first message could never be resumed afterwards). Instead, the new
message should be appended to the paused conversation, and the owner
should see the full context once she resolves it.

Only the final purchase confirmation (see graph.py's _t_final_confirm)
still pauses for review, so that's what's used here to get a thread into
a pending state.
"""

import pytest
from langchain_core.messages import HumanMessage

from bella_charm_agent import graph, runner
from bella_charm_agent.state import new_conversation_state

_CART = [{"category": "Charm", "variant": "Heart", "price": 5, "quantity": 1, "tag": None}]


@pytest.fixture(autouse=True)
def _isolated_graph(monkeypatch):
    """runner._graph is backed by a real, persistent SQLite file (so
    conversations survive a restart) -- tests must not write fake
    customer_ids into it. Swap in a fresh, in-memory-backed graph for the
    duration of each test instead."""
    monkeypatch.setattr(runner, "_graph", graph.build_graph())


def _seed_pending_final_confirm(customer_id: str, first_message: str) -> dict:
    """Puts a thread into a pending-review state (as if the customer had
    just confirmed their order) without going through the whole
    category/variant/cart/delivery funnel -- that's covered elsewhere;
    this test is specifically about what happens *after* a pause."""
    config = {"configurable": {"thread_id": customer_id}}
    seed = {
        **new_conversation_state(customer_id),
        "current_step": "awaiting_final_confirm",
        "cart": _CART,
        "delivery_destination": "UK",
        "delivery_address": "1 High St",
        "messages": [HumanMessage(content=first_message)],
    }
    return runner._graph.invoke(seed, config)


def test_second_message_while_pending_gets_queued_not_processed_independently(monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))

    customer_id = "queue_test_1"
    first = _seed_pending_final_confirm(customer_id, "yes")
    assert "__interrupt__" in first  # sanity: it did pause

    result = runner.submit_customer_message(customer_id, "also, can you add a bracelet?")
    assert result["status"] == "pending_review"
    assert result["queued_messages"] == ["yes", "also, can you add a bracelet?"]
    # The original pending draft (bank details) is unchanged -- a second
    # message doesn't alter what's actually being reviewed.
    assert "Sort Code: 000000" in result["draft_reply"]


def test_a_third_message_keeps_queuing_onto_the_same_pending_review(monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))

    customer_id = "queue_test_3"
    _seed_pending_final_confirm(customer_id, "yes")
    runner.submit_customer_message(customer_id, "also a bracelet")
    result = runner.submit_customer_message(customer_id, "actually make that two bracelets")

    assert result["status"] == "pending_review"
    assert result["queued_messages"] == ["yes", "also a bracelet", "actually make that two bracelets"]


def test_resolving_after_queued_messages_still_works_and_keeps_full_context(monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 111111")
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))

    customer_id = "queue_test_2"
    _seed_pending_final_confirm(customer_id, "yes")
    runner.submit_customer_message(customer_id, "one more thing...")

    result = runner.resolve_pending_review(customer_id, {"action": "approve"})
    assert result["approved"] is True
    assert "Sort Code: 111111" in result["draft_reply"]
    assert [m.content for m in result["messages"]] == ["yes", "one more thing..."]


def test_no_pending_review_processes_normally(monkeypatch, fake_llm):
    # Sanity check that the queuing branch doesn't accidentally trigger
    # for a fresh/completed thread with nothing pending.
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="store_hours")))
    result = runner.submit_customer_message("queue_test_4", "are you open?")
    assert result["status"] == "sent"
    assert "9:00" in result["draft_reply"]
