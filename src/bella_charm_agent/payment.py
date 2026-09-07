"""Loads Bella Charm London's bank details for the final-confirmation reply.

Never hardcoded, never committed: this reads a local, gitignored secrets
file. See config/payment_secrets.example.json for the expected shape --
copy it to config/payment_secrets.json and fill in the real details.
"""

import json
from pathlib import Path

_SECRETS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "payment_secrets.json"
)


def load_payment_details() -> str:
    if not _SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"{_SECRETS_PATH} not found. Copy "
            "config/payment_secrets.example.json to "
            "config/payment_secrets.json and fill in the real bank details."
        )
    data = json.loads(_SECRETS_PATH.read_text())
    lines = [
        f"{key.replace('_', ' ').title()}: {value}"
        for key, value in data.items()
        if value
    ]
    return "\n".join(lines)
