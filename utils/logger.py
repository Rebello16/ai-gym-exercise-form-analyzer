"""Project-wide logging setup."""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; configures the root handler once."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_FORMAT)
    return logging.getLogger(name)
