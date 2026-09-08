"""Tests for payment.load_payment_details()'s two sources: environment
variables (how a real deployment provides secrets) take priority, and
the local gitignored file is only a fallback for local dev.
"""

import json

import pytest

from bella_charm_agent import payment

_ENV_VARS = [
    "PAYMENT_BANK_NAME",
    "PAYMENT_ACCOUNT_NAME",
    "PAYMENT_SORT_CODE",
    "PAYMENT_ACCOUNT_NUMBER",
    "PAYMENT_IBAN",
    "PAYMENT_PAYPAL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Tests must not accidentally read whatever's in the real developer's
    shell environment or local secrets file."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_env_vars_take_priority_over_the_file(monkeypatch, tmp_path):
    # Points at a file that would raise if it were actually read, proving
    # the env-var path never touches it.
    monkeypatch.setattr(payment, "_SECRETS_PATH", tmp_path / "does_not_exist.json")
    monkeypatch.setenv("PAYMENT_BANK_NAME", "Test Bank")
    monkeypatch.setenv("PAYMENT_SORT_CODE", "11-22-33")

    result = payment.load_payment_details()
    assert result == "Bank Name: Test Bank\nSort Code: 11-22-33"


def test_falls_back_to_the_file_when_no_env_vars_are_set(monkeypatch, tmp_path):
    secrets_file = tmp_path / "payment_secrets.json"
    secrets_file.write_text(json.dumps({"bank_name": "File Bank", "sort_code": "44-55-66", "iban": ""}))
    monkeypatch.setattr(payment, "_SECRETS_PATH", secrets_file)

    result = payment.load_payment_details()
    assert result == "Bank Name: File Bank\nSort Code: 44-55-66"


def test_raises_a_helpful_error_when_neither_source_is_available(monkeypatch, tmp_path):
    monkeypatch.setattr(payment, "_SECRETS_PATH", tmp_path / "does_not_exist.json")

    with pytest.raises(FileNotFoundError, match="PAYMENT_"):
        payment.load_payment_details()
