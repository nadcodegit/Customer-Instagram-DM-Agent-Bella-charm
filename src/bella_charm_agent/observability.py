"""Structured logging + error reporting setup for the web app.

Kept as its own module, called once from web.py at import time, so the
two concerns -- what gets written to Railway's log stream, and what
gets reported to Sentry -- have one place to configure instead of
being scattered across every module that might log or raise.

SENTRY_DSN is optional (unlike DASHBOARD_USERNAME/PASSWORD, which are
required): local dev has no reason to need it, so its absence just
means error reporting is off, not a startup failure.
"""

import logging
import os

import sentry_sdk

logger = logging.getLogger("bella_charm_agent")


def configure() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = os.environ.get("SENTRY_DSN")
    if dsn:
        sentry_sdk.init(dsn=dsn)
        logger.info("Sentry error reporting enabled")
    else:
        logger.info("SENTRY_DSN not set -- Sentry error reporting disabled")
