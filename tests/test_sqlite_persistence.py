"""Proves the actual point of switching from MemorySaver to SqliteSaver:
conversation state survives a process restart, as long as the same
database file is reused. Builds two *separate* graph instances (fresh
sqlite3 connections) pointed at the same on-disk file, simulating a real
server restart between them -- MemorySaver could never pass this, since
its state lives only in that one process's RAM.
"""

import sqlite3

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from bella_charm_agent import graph
from bella_charm_agent.state import new_conversation_state


def _graph_on(db_path) -> "graph.StateGraph":
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return graph.build_graph(checkpointer=SqliteSaver(conn))


def test_state_survives_a_simulated_restart(tmp_path, monkeypatch, fake_llm):
    db_path = tmp_path / "test_conversations.sqlite"
    thread_id = "restart_test"
    config = {"configurable": {"thread_id": thread_id}}

    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.NewRequestClassification(intent="price_inquiry", category="Charm"))
    )
    graph_before_restart = _graph_on(db_path)
    state = {**new_conversation_state(thread_id), "messages": [HumanMessage(content="how much is a charm?")]}
    result1 = graph_before_restart.invoke(state, config)
    assert result1["current_step"] == "awaiting_browse_offer"

    # Simulate a restart: a brand new graph instance, brand new sqlite3
    # connection -- nothing shared with the one above except the file.
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.BrowseOfferAnswer(matched=True, wants_to_browse=True)))
    graph_after_restart = _graph_on(db_path)
    result2 = graph_after_restart.invoke({"messages": [HumanMessage(content="yes")]}, config)

    # If state hadn't survived, current_step would still be "start" and
    # this would misfire as a fresh classify instead of the browse-offer
    # answer -- it correctly picked up right where the first instance
    # left off, category and all.
    assert result2["current_step"] == "awaiting_variant"
    assert result2["pending_selection"] == {"category": "Charm"}


def test_a_pending_review_survives_a_simulated_restart(tmp_path, monkeypatch, fake_llm):
    db_path = tmp_path / "test_conversations_pending.sqlite"
    thread_id = "restart_pending_test"
    config = {"configurable": {"thread_id": thread_id}}
    cart = [{"category": "Charm", "variant": "Heart", "price": 5, "quantity": 1, "tag": None}]

    monkeypatch.setattr(graph, "load_payment_details", lambda: "Sort Code: 000000")
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.StepAnswerMatch(matched_option="Yes")))
    graph_before_restart = _graph_on(db_path)
    state = {
        **new_conversation_state(thread_id),
        "current_step": "awaiting_final_confirm",
        "cart": cart,
        "delivery_destination": "UK",
        "delivery_address": "1 High St",
        "messages": [HumanMessage(content="yes")],
    }
    result1 = graph_before_restart.invoke(state, config)
    assert "__interrupt__" in result1  # sanity: it's actually pending

    # Restart -- the owner comes back later, on a process that was
    # restarted in between, and resolves it.
    from langgraph.types import Command

    graph_after_restart = _graph_on(db_path)
    result2 = graph_after_restart.invoke(Command(resume={"action": "approve"}), config)
    assert result2["approved"] is True
    assert "Sort Code: 000000" in result2["draft_reply"]
