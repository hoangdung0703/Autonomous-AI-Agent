"""Shared application logger.

Usage: `from app.utils.logger import logger`. Services use this instead of
print() for key events (collection created, embedding failures, connection
errors).
"""

import logging

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.propagate = False
