"""Feeds a simulated Instagram DM into the graph, and resolves the owner's
review decision once one comes in.

This stands in for the future webhook adapter + review UI: same graph,
same state, just Python function calls instead of an HTTP request from
Meta and a click in some review interface. `submit_customer_message`
covers both outcomes -- almost everything (including the "other"
out-of-scope holding reply) sends automatically now; only the final
purchase confirmation (bank/PayPal details) still pauses for review, see
await_owner_approval in graph.py:

  - Auto-sent: the graph ran straight through -- there's a final
    draft_reply and nothing more to do.
  - Pending review: the graph paused -- `resolve_pending_review` resumes
    it once the owner's decision comes in.

Run this file directly for an interactive terminal chat that plays both
parts (customer and owner) against one fixed customer_id.
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from langgraph.types import Command  # noqa: E402  (import after load_dotenv on purpose)

from .graph import build_graph  # noqa: E402
from .state import new_conversation_state  # noqa: E402

_graph = build_graph()


def submit_customer_message(customer_id: str, text: str) -> dict:
    """Simulates one incoming DM from `customer_id`. Returns either
    {"status": "sent", "draft_reply": ...} if it was confidently in scope
    and went out automatically, or {"status": "pending_review",
    "customer_message": ..., "draft_reply": ..., "needs_human": True} if
    it's paused waiting on the owner (see resolve_pending_review).

    If a review is *already* pending on this thread (the customer sent
    another message before the owner got to the last one), this does
    **not** start an independent turn -- that could race with, or
    silently orphan, the pending one (see the "girlfriend's birthday"
    incident: a second message ran the graph fresh, and the first,
    already-escalated message could never be resumed afterwards). Instead
    it appends the new message to the paused conversation via
    `update_state` (no node runs) and returns the *same* pending review,
    so the owner sees the full context -- not just the first message --
    once she gets to it. The payload is rebuilt from `existing.values`
    (not `existing.tasks[...].interrupts`) because the *first*
    `update_state` call permanently clears the interrupt info from the
    snapshot -- a third message arriving would find nothing left to read.
    """
    config = {"configurable": {"thread_id": customer_id}}
    existing = _graph.get_state(config)

    if existing.next:
        _graph.update_state(config, {"messages": [HumanMessage(content=text)]})
        values = _graph.get_state(config).values
        return {
            "status": "pending_review",
            "customer_id": values["customer_id"],
            "customer_message": values["messages"][-1].content,
            "draft_reply": values["draft_reply"],
            "needs_human": values["needs_human"],
            "queued_messages": [m.content for m in values["messages"]],
        }

    if existing.values:
        update = {"messages": [HumanMessage(content=text)]}
    else:
        update = {
            **new_conversation_state(customer_id),
            "messages": [HumanMessage(content=text)],
        }

    result = _graph.invoke(update, config)
    if "__interrupt__" in result:
        return {"status": "pending_review", **result["__interrupt__"][0].value}
    return {"status": "sent", "draft_reply": result["draft_reply"]}


def resolve_pending_review(customer_id: str, decision: dict) -> dict:
    """Resumes a paused conversation with the owner's decision:
    {"action": "approve"}, {"action": "edit", "text": "..."}, or
    {"action": "reject"}. Returns the final state -- check
    result["approved"] before actually delivering result["draft_reply"]
    anywhere.
    """
    config = {"configurable": {"thread_id": customer_id}}
    return _graph.invoke(Command(resume=decision), config)


if __name__ == "__main__":
    customer_id = "demo_customer"
    print("Chatting as a customer DM'ing Bella Charm London. Ctrl+C to quit.\n")
    while True:
        text = input("You (customer): ")
        review = submit_customer_message(customer_id, text)

        if review["status"] == "sent":
            print(f"[AUTO-SENT]: {review['draft_reply']}\n")
            continue

        print("\n--- pending owner review (needs_human) ---")
        print(f"Customer said: {review['customer_message']}")
        print(f"Draft reply:   {review['draft_reply']}")
        if "queued_messages" in review:
            print(f"(more arrived while pending: {review['queued_messages']})")

        choice = input("Owner: [a]pprove / [e]dit / [r]eject? ").strip().lower()
        if choice == "e":
            new_text = input("New reply text: ")
            decision = {"action": "edit", "text": new_text}
        elif choice == "r":
            decision = {"action": "reject"}
        else:
            decision = {"action": "approve"}

        result = resolve_pending_review(customer_id, decision)
        if result.get("approved"):
            print(f"[SENT TO CUSTOMER]: {result['draft_reply']}\n")
        else:
            print("[Rejected -- nothing sent to customer]\n")
