"""Structured logger utility."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def configure_logger(log_path: Path, name: str = "autotrader") -> logging.Logger:
    """Configure JSON-line logger writing to file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, payload: dict[str, Any]) -> None:
    """Write structured event as one-line JSON."""
    row = {"event": event, **payload}
    logger.info(json.dumps(row, default=str, sort_keys=True))
