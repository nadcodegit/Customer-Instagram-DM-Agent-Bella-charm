"""Tests for observability.configure() -- the logging + Sentry setup.

Doesn't touch a real Sentry project: sentry_sdk.init is monkeypatched
to a recording stub, so these just check *whether* it gets called
based on SENTRY_DSN, not that anything is actually reported anywhere.
"""

import sentry_sdk

from bella_charm_agent import observability


def test_configure_skips_sentry_when_dsn_unset(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))

    observability.configure()

    assert calls == []


def test_configure_initializes_sentry_when_dsn_set(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "https://fake@sentry.example/1")
    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))

    observability.configure()

    assert calls == [{"dsn": "https://fake@sentry.example/1"}]
