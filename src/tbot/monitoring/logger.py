"""
Structured JSON logging for TBot.

configure_logging() — call once at the top of any entry-point script.
get_logger(name)    — use everywhere instead of logging.getLogger().
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(
    log_file: Path | str | None = None,
    level: str = "INFO",
) -> None:
    """
    Wire structlog to emit JSON to stdout (and optionally a log file).

    Args:
        log_file: path to append JSON log lines to (None = stdout only)
        level:    minimum log level — "DEBUG", "INFO", "WARNING", "ERROR"
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()       # prevent duplicate handlers on re-call (tests)
    root.setLevel(numeric_level)

    plain_fmt = logging.Formatter("%(message)s")  # structlog owns formatting

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(plain_fmt)
    root.addHandler(stdout_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(plain_fmt)
        root.addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,   # picks up bind_contextvars() calls
            structlog.stdlib.add_log_level,            # adds "level" key
            structlog.stdlib.add_logger_name,          # adds "logger" key
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,      # serialises exception tracebacks
            structlog.processors.JSONRenderer(),        # final step → JSON string
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,  # False so tests can re-configure
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger. Prefer this over logging.getLogger()."""
    return structlog.get_logger(name)
