"""Centralized rich-formatted logger with retention-focused taxonomy."""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


_INITIALIZED = False


def get_logger(name: str = "yt_os") -> logging.Logger:
    global _INITIALIZED
    logger = logging.getLogger(name)
    if _INITIALIZED:
        return logger

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if _HAS_RICH:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
        formatter = logging.Formatter("%(message)s")
    else:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    _INITIALIZED = True
    return logger


def retention_event(stage: str, **fields: Any) -> None:
    """Structured event log so we can grep retention decisions across a run."""
    log = get_logger("retention")
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    log.info("[bold cyan]%s[/bold cyan] %s", stage, payload)
