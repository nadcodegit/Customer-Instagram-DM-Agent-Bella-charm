"""Shared test fixtures.

The LLM-calling functions in graph.py always go through the module-level
`_llm()` -- these fixtures let a test replace it with a stub that returns a
canned structured-output response, so tests exercise the surrounding Python
logic (routing, state updates, cart math) without hitting the network or
depending on how well a real model happens to answer that day.
"""

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from bella_charm_agent.state import new_conversation_state

# Only matters for the opt-in live_llm tests (see test_llm_quality_manual.py)
# -- harmless for everything else, since they never read GROQ_API_KEY.
load_dotenv()

# web.py refuses to import without these set (the review dashboard must
# never come up unprotected). Force-set (not setdefault) -- unlike
# GROQ_API_KEY, tests have no reason to want whatever's in a dev's local
# .env here; they need one known, fixed value, not "whatever the
# developer happened to put in config/payment_secrets.json's neighbor".
os.environ["DASHBOARD_USERNAME"] = "test"
os.environ["DASHBOARD_PASSWORD"] = "test"


class _FakeStructuredOutput:
    def __init__(self, response):
        self._response = response

    def invoke(self, prompt):
        return self._response


class _FakeLLM:
    """`responses` is either a single canned response (returned regardless
    of which structured-output model is requested), or a dict keyed by the
    model class -- needed when one code path makes more than one
    differently-typed LLM call, e.g. an off-topic match that triggers a
    reclassification call."""

    def __init__(self, responses):
        self._responses = responses

    def with_structured_output(self, model):
        response = self._responses[model] if isinstance(self._responses, dict) else self._responses
        return _FakeStructuredOutput(response)


@pytest.fixture
def fake_llm():
    """fake_llm(responses) -> a zero-arg callable shaped like `_llm`. See
    `_FakeLLM` for what `responses` can be. Use with monkeypatch:
    `monkeypatch.setattr(graph, "_llm", fake_llm(responses))`.
    """

    def _factory(responses):
        return lambda: _FakeLLM(responses)

    return _factory


@pytest.fixture
def make_state():
    """make_state(text=None, **overrides) -> a ConversationState for
    "test_customer", optionally with the customer's latest message set and
    any other fields overridden (e.g. current_step, pending_selection)."""

    def _factory(text: str | None = None, **overrides) -> dict:
        state = new_conversation_state("test_customer")
        if text is not None:
            state["messages"] = [HumanMessage(content=text)]
        state.update(overrides)
        return state

    return _factory
