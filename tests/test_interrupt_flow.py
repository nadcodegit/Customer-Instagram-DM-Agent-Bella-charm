"""Tests for the human-review pause (await_owner_approval) at the graph
level -- unlike the other node tests, `interrupt()` only works inside a
real graph run, so these compile the graph and drive it with
`.invoke()` / `.invoke(Command(resume=...))` instead of calling a node
function directly. The LLM is still mocked (via monkeypatching
`graph._llm`), so these stay fast and deterministic.

A confidently in-scope reply (needs_human False -- price/hours/greeting/
payment-proof) sends automatically, with no pause at all. Only a
genuinely out-of-scope one (needs_human True -- classified 'other')
pauses for the owner's decision.
"""

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from bella_charm_agent import graph
from bella_charm_agent.state import new_conversation_state


def _start_conversation(g, thread_id, text, monkeypatch, fake_llm, response):
    """Invoke a fresh conversation and return (result, config)."""
    monkeypatch.setattr(graph, "_llm", fake_llm(response))
    config = {"configurable": {"thread_id": thread_id}}
    state = {**new_conversation_state(thread_id), "messages": [HumanMessage(content=text)]}
    return g.invoke(state, config), config


def test_needs_human_false_sends_automatically_without_pausing(monkeypatch, fake_llm):
    g = graph.build_graph()
    result, _ = _start_conversation(
        g, "t0", "are you open?", monkeypatch, fake_llm, graph.NewRequestClassification(intent="store_hours")
    )
    assert "__interrupt__" not in result
    assert result["approved"] is True
    assert "9:00" in result["draft_reply"]


def test_needs_human_true_pauses_with_the_review_payload(monkeypatch, fake_llm):
    g = graph.build_graph()
    result, _ = _start_conversation(
        g, "t1", "can you make me a custom necklace?", monkeypatch, fake_llm, graph.NewRequestClassification(intent="other")
    )
    payload = result["__interrupt__"][0].value
    assert payload["customer_id"] == "t1"
    assert payload["customer_message"] == "can you make me a custom necklace?"
    assert payload["needs_human"] is True


def test_approve_sends_the_draft_reply_as_is(monkeypatch, fake_llm):
    g = graph.build_graph()
    result, config = _start_conversation(
        g, "t2", "custom request", monkeypatch, fake_llm, graph.NewRequestClassification(intent="other")
    )
    payload = result["__interrupt__"][0].value
    result2 = g.invoke(Command(resume={"action": "approve"}), config)
    assert result2["approved"] is True
    assert result2["draft_reply"] == payload["draft_reply"]


def test_edit_overrides_the_draft_reply(monkeypatch, fake_llm):
    g = graph.build_graph()
    _, config = _start_conversation(
        g, "t3", "custom request", monkeypatch, fake_llm, graph.NewRequestClassification(intent="other")
    )
    result = g.invoke(
        Command(resume={"action": "edit", "text": "Let me check with the owner personally!"}), config
    )
    assert result["approved"] is True
    assert result["draft_reply"] == "Let me check with the owner personally!"


def test_reject_sends_nothing(monkeypatch, fake_llm):
    g = graph.build_graph()
    _, config = _start_conversation(
        g, "t4", "custom request", monkeypatch, fake_llm, graph.NewRequestClassification(intent="other")
    )
    result = g.invoke(Command(resume={"action": "reject"}), config)
    assert result["approved"] is False


def test_an_auto_sent_turn_is_followed_by_a_normal_next_turn(monkeypatch, fake_llm):
    # After an auto-sent reply, the conversation should just continue --
    # no leftover pause, no stale needs_human bleeding into the next turn.
    g = graph.build_graph()
    thread_id = "t5"
    config = {"configurable": {"thread_id": thread_id}}

    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="greeting")))
    state = {**new_conversation_state(thread_id), "messages": [HumanMessage(content="hi")]}
    first = g.invoke(state, config)
    assert "__interrupt__" not in first

    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="store_hours")))
    second = g.invoke({"messages": [HumanMessage(content="are you open?")]}, config)
    assert "__interrupt__" not in second
    assert "9:00" in second["draft_reply"]


def test_needs_human_does_not_leak_from_an_escalated_turn_into_the_next_answer(monkeypatch, fake_llm):
    # Regression guard for the exact bug _finalize_reply exists to
    # prevent: an escalated ("other") turn sets needs_human True, then a
    # completely normal answer right after must NOT inherit that True.
    g = graph.build_graph()
    thread_id = "t6"
    config = {"configurable": {"thread_id": thread_id}}

    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(
            {
                graph.CategoryAnswer: graph.CategoryAnswer(matched=False),
                graph.NewRequestClassification: graph.NewRequestClassification(intent="other"),
            }
        ),
    )
    state = {
        **new_conversation_state(thread_id),
        "current_step": "awaiting_category",
        "messages": [HumanMessage(content="do you deliver internationally by drone?")],
    }
    escalated = g.invoke(state, config)
    payload = escalated["__interrupt__"][0].value
    assert payload["needs_human"] is True
    g.invoke(Command(resume={"action": "approve"}), config)

    monkeypatch.setattr(graph, "_llm", fake_llm(graph.CategoryAnswer(matched=True, category="Bracelet")))
    result = g.invoke({"messages": [HumanMessage(content="bracelet")]}, config)
    assert "__interrupt__" not in result
    assert result["approved"] is True
