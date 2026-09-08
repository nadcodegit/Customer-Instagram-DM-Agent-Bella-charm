"""Tests for runner.list_pending_reviews() and the web.py dashboard --
both reuse runner.submit_customer_message/resolve_pending_review, so
these mostly check that pending reviews are correctly discovered and
listed, and that resolving one through the HTTP route has the same
effect as resolving it directly.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from bella_charm_agent import graph, runner, web

_CART = [{"category": "Charm", "variant": "Heart", "price": 5, "quantity": 1, "tag": None}]


@pytest.fixture(autouse=True)
def _isolated_graph(monkeypatch):
    """Same reasoning as test_runner_queuing.py: runner._graph is backed
    by the real, persistent SQLite file -- tests must use a fresh,
    in-memory-backed graph instead."""
    monkeypatch.setattr(runner, "_graph", graph.build_graph())


def _seed_pending_final_confirm(customer_id: str) -> None:
    config = {"configurable": {"thread_id": customer_id}}
    state = {
        **runner.new_conversation_state(customer_id),
        "current_step": "awaiting_final_confirm",
        "cart": _CART,
        "delivery_destination": "UK",
        "delivery_address": "1 High St",
        "messages": [HumanMessage(content="yes")],
    }
    runner._graph.invoke(state, config)


@pytest.fixture
def client():
    return TestClient(web.app)


def test_list_pending_reviews_is_empty_with_nothing_pending():
    assert runner.list_pending_reviews() == []


def test_list_pending_reviews_finds_a_pending_thread(monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))
    _seed_pending_final_confirm("dashboard_test_1")

    pending = runner.list_pending_reviews()
    assert len(pending) == 1
    assert pending[0]["customer_id"] == "dashboard_test_1"
    assert "Sort Code: 000000" in pending[0]["draft_reply"]


def test_list_pending_reviews_excludes_completed_threads(monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="store_hours")))
    runner.submit_customer_message("dashboard_test_2", "are you open?")  # auto-sends, reaches END

    assert runner.list_pending_reviews() == []


def test_dashboard_shows_no_pending_reviews_message(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "No pending reviews" in response.text


def test_dashboard_lists_a_pending_review(client, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))
    _seed_pending_final_confirm("dashboard_test_3")

    response = client.get("/")
    assert "dashboard_test_3" in response.text or "Sort Code: 000000" in response.text
    assert "Sort Code: 000000" in response.text


def test_resolve_approve_via_http_matches_direct_call(client, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))
    _seed_pending_final_confirm("dashboard_test_4")

    response = client.post(
        "/resolve/dashboard_test_4", data={"action": "approve"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert runner.list_pending_reviews() == []  # resolved, no longer pending

    final = runner._graph.get_state({"configurable": {"thread_id": "dashboard_test_4"}})
    assert final.values["approved"] is True


def test_resolve_edit_via_http_sends_the_edited_text(client, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))
    _seed_pending_final_confirm("dashboard_test_5")

    client.post(
        "/resolve/dashboard_test_5",
        data={"action": "edit", "edited_text": "Please double check the total with me first!"},
        follow_redirects=False,
    )
    final = runner._graph.get_state({"configurable": {"thread_id": "dashboard_test_5"}})
    assert final.values["draft_reply"] == "Please double check the total with me first!"


def test_resolve_reject_via_http(client, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))
    _seed_pending_final_confirm("dashboard_test_6")

    client.post("/resolve/dashboard_test_6", data={"action": "reject"}, follow_redirects=False)
    final = runner._graph.get_state({"configurable": {"thread_id": "dashboard_test_6"}})
    assert final.values["approved"] is False


# ---------------------------------------------------------------------------
# /webhook
# ---------------------------------------------------------------------------


def test_webhook_auto_sendable_message_delivers_immediately(client, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="store_hours")))
    delivered = []
    monkeypatch.setattr(web, "_deliver_to_customer", lambda customer_id, text: delivered.append((customer_id, text)))

    response = client.post("/webhook", json={"customer_id": "webhook_test_1", "text": "are you open?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "sent"
    assert "9:00" in body["draft_reply"]
    assert delivered == [("webhook_test_1", body["draft_reply"])]


def test_webhook_message_needing_review_does_not_deliver_yet(client, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="price_inquiry", category="Charm")))
    delivered = []
    monkeypatch.setattr(web, "_deliver_to_customer", lambda customer_id, text: delivered.append((customer_id, text)))

    response = client.post("/webhook", json={"customer_id": "webhook_test_2", "text": "how much is a charm?"})
    assert response.status_code == 200
    assert response.json()["status"] == "sent"  # price inquiry auto-sends too
    assert delivered  # sanity: this path does deliver


def test_webhook_can_originate_a_fresh_pending_review_and_it_reaches_the_dashboard(
    client, monkeypatch, fake_llm
):
    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    customer_id = "webhook_test_3"
    config = {"configurable": {"thread_id": customer_id}}

    # Drive to a *resting* awaiting_final_confirm state via a real invoke
    # (not update_state, which would leave .next non-empty and make the
    # webhook message below look like it's queuing onto an
    # already-pending review instead of genuinely triggering one for the
    # first time).
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(graph.AddressExtraction(contains_address=True, address="1 High St", consistent_with_destination=True)),
    )
    seed_state = {
        **runner.new_conversation_state(customer_id),
        "current_step": "awaiting_delivery_address",
        "cart": _CART,
        "delivery_destination": "UK",
        "messages": [HumanMessage(content="1 High St")],
    }
    runner._graph.invoke(seed_state, config)
    assert runner._graph.get_state(config).next == ()  # sanity: genuinely at rest, nothing pending

    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))
    delivered = []
    monkeypatch.setattr(web, "_deliver_to_customer", lambda cid, text: delivered.append((cid, text)))

    response = client.post("/webhook", json={"customer_id": customer_id, "text": "yes"})
    assert response.status_code == 200
    assert response.json()["status"] == "pending_review"
    assert delivered == []  # not delivered -- still waiting on the owner

    dashboard_html = client.get("/").text
    assert "Sort Code: 000000" in dashboard_html
