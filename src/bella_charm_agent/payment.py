"""Loads Bella Charm London's bank details for the final-confirmation reply.

Never hardcoded, never committed. Two sources, checked in order:

1. Environment variables (PAYMENT_BANK_NAME, PAYMENT_ACCOUNT_NAME,
   PAYMENT_SORT_CODE, PAYMENT_ACCOUNT_NUMBER, PAYMENT_IBAN,
   PAYMENT_PAYPAL) -- how a real deployment provides them, set in the
   hosting platform's own secrets panel, never in a file that ships with
   the code.
2. A local, gitignored secrets file (config/payment_secrets.json) if none
   of those env vars are set -- for local dev. See
   config/payment_secrets.example.json for its shape; copy it to
   config/payment_secrets.json and fill in the real details.
"""

import json
import os
from pathlib import Path

_SECRETS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "payment_secrets.json"
)

_ENV_FIELDS = {
    "Bank Name": "PAYMENT_BANK_NAME",
    "Account Name": "PAYMENT_ACCOUNT_NAME",
    "Sort Code": "PAYMENT_SORT_CODE",
    "Account Number": "PAYMENT_ACCOUNT_NUMBER",
    "Iban": "PAYMENT_IBAN",
    "Paypal": "PAYMENT_PAYPAL",
}


def _from_env() -> str | None:
    values = {label: os.environ.get(var) for label, var in _ENV_FIELDS.items()}
    if not any(values.values()):
        return None
    return "\n".join(f"{label}: {value}" for label, value in values.items() if value)


def _from_file() -> str:
    if not _SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"{_SECRETS_PATH} not found, and no PAYMENT_* environment "
            "variables are set either. Copy config/payment_secrets.example.json "
            "to config/payment_secrets.json and fill in the real bank details."
        )
    data = json.loads(_SECRETS_PATH.read_text())
    lines = [
        f"{key.replace('_', ' ').title()}: {value}"
        for key, value in data.items()
        if value
    ]
    return "\n".join(lines)


def load_payment_details() -> str:
    return _from_env() or _from_file()
