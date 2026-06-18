"""Logging setup that always passes records through the secret redactor."""

from __future__ import annotations

import logging
import sys
import os

from .security import redact

TRACE_LOGGER_NAME = "qa_agent.trace"


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return redact(original)


class _TraceFormatter(logging.Formatter):
    """Compact, eye-catching format for the live agent execution trace."""

    def format(self, record: logging.LogRecord) -> str:
        msg = redact(record.getMessage())
        ts = self.formatTime(record, datefmt="%H:%M:%S")
        return f"{ts} | {msg}"


def configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet down chatty libs.
    for noisy in ("httpx", "httpcore", "openai", "playwright"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, root.level))

    # The live execution trace gets its OWN handler attached to its OWN logger.
    # We disable propagation so uvicorn's `disable_existing_loggers` policy and
    # any later root reconfig can't silence it, and so we don't double-print
    # through the root handler. Result: trace lines reliably reach the
    # terminal even when running under `python -m qa_agent.web` (uvicorn).
    trace = logging.getLogger(TRACE_LOGGER_NAME)
    trace.handlers.clear()
    trace_handler = logging.StreamHandler(stream=sys.stderr)
    trace_handler.setFormatter(_TraceFormatter())
    trace.addHandler(trace_handler)
    trace.setLevel(logging.INFO)
    trace.propagate = False
