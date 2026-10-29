"""JSON logs that are useful to an operator and contain nothing a user wrote.

Every line is one JSON object (structlog), from our code and from libraries alike (uvicorn, MLflow
go through the same formatter). Privacy is enforced by an allowlist, not by care: a field whose
name is not in `ALLOWED_FIELDS` is replaced with `[redacted]` before it is rendered, so a future
`log.info("scored", subject=email.subject)` cannot leak anything. Messages of our own events are
fixed strings; values go in fields.

What the allowlist cannot see is free text inside a message or an exception, which is why the
API never logs exception messages or tracebacks, only exception types (`serving/access.py`), and
why a test sends a canary email through every path and searches the logs for it.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any, Final

import structlog

ALLOWED_FIELDS: Final = frozenset(
    {
        # added by the pipeline
        "event", "level", "logger", "timestamp",
        # request
        "request_id", "method", "path", "status", "duration_ms", "request_bytes",
        # scoring: sizes, numbers and identifiers, never content
        "email_bytes", "scoring_ms", "label", "score", "model", "model_version",
        # failures
        "error_type", "detail",
    }
)  # fmt: skip
REDACTED: Final = "[redacted]"
_OURS: Final = "_mailguard_handler"


def allowlist(_: Any, __: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """structlog processor: keep allowlisted fields, redact the value of any other."""
    for key in list(event):
        if key not in ALLOWED_FIELDS and not key.startswith("_"):
            event[key] = REDACTED
    return event


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    """Route structlog and the standard library through one JSON (or console) formatter.

    Idempotent, and it only replaces the handler it installed itself, so test frameworks and
    embedding applications keep theirs.
    """
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        allowlist,
    ]
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    setattr(handler, _OURS, True)

    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, _OURS, False)] + [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []  # let their records reach the root formatter
        logging.getLogger(name).propagate = True
    # The access log of the API replaces uvicorn's, with timings and without client addresses.
    logging.getLogger("uvicorn.access").disabled = True
