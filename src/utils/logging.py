"""Structured logging shared by every component.

Monitoring is a debugging discipline: when a drift alert fires at 3am you need
to reconstruct exactly which encoder version, which judge rubric and which
thresholds produced the decision. Every log line therefore carries the run and
window identifiers once the simulation loop is running.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

_CONFIGURED = False


class _ContextFilter(logging.Filter):
    """Injects run/window identifiers set by the simulation driver."""

    run_id: str = "-"
    window: str = "-"

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        record.window = self.window
        return True


_context = _ContextFilter()


def set_log_context(run_id: str | None = None, window: str | None = None) -> None:
    if run_id is not None:
        _context.run_id = run_id
    if window is not None:
        _context.window = window


def get_log_context() -> dict[str, str]:
    return {"run_id": _context.run_id, "window": _context.window}


def setup_logging(level: str = "INFO", *, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    # Windows consoles default to cp1252, which cannot render the arrows and
    # box characters used in log lines. A UnicodeEncodeError inside the logging
    # handler kills the process mid-run, so the stream is upgraded up front.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # A stream that refuses to reconfigure is not fatal; log anyway.
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s [%(name)s] run=%(run_id)s win=%(window)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(_context)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for noisy in ("urllib3", "httpx", "httpcore", "matplotlib", "PIL", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def log_dict(logger: logging.Logger, message: str, payload: dict[str, Any]) -> None:
    flat = " ".join(f"{k}={v}" for k, v in payload.items())
    logger.info("%s | %s", message, flat)
