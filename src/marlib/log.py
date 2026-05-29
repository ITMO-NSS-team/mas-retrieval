"""Centralized logger configuration for marlib.

Import the shared logger everywhere instead of using print():

    from marlib.log import logger

    logger.info("message", key=value)

Configuration lives here so it is applied once, on first import.
"""

from __future__ import annotations

import os

from logly import logger

# Level can be overridden via env var, e.g. MARLIB_LOG_LEVEL=DEBUG
_LEVEL = os.environ.get("MARLIB_LOG_LEVEL", "INFO").upper()

logger.configure(level=_LEVEL, color=True, console=True)

__all__ = ["logger"]
