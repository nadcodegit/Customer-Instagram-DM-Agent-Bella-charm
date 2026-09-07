"""Feeds a simulated Instagram DM into the graph and returns the draft reply.

This stands in for the future webhook adapter: same graph, same state,
just a Python function call instead of an HTTP request from Meta. Swapping
in the real webhook later only changes how `send_message` gets called and
how its return value is delivered -- nothing in graph.py.

Run this file directly for an interactive terminal chat against one fixed
customer_id, useful for manually trying out the flow end to end.
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from .graph import build_graph  # noqa: E402  (import after load_dotenv on purpose)
from .state import new_conversation_state  # noqa: E402

_graph = build_graph()


def send_message(customer_id: str, text: str) -> str:
    """Simulates one incoming DM from `customer_id`, returns the agent's
    draft reply. Never sends anything anywhere -- per the human-review
    decision in customer-conversation-scenarios.md, a person reads and
    sends this manually.
    """
    config = {"configurable": {"thread_id": customer_id}}
    existing = _graph.get_state(config)

    if existing.values:
        update = {"messages": [HumanMessage(content=text)]}
    else:
        update = {
            **new_conversation_state(customer_id),
            "messages": [HumanMessage(content=text)],
        }

    result = _graph.invoke(update, config)
    return result["draft_reply"]


if __name__ == "__main__":
    customer_id = "demo_customer"
    print("Chatting as a customer DM'ing Bella Charm London. Ctrl+C to quit.\n")
    while True:
        text = input("You: ")
        reply = send_message(customer_id, text)
        print(f"Draft reply: {reply}\n")
