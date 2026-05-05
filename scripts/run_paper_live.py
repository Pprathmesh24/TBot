"""
Start the live paper trading loop.

Usage:
    .venv/bin/python scripts/run_paper_live.py

Stops cleanly on Ctrl+C.

Logging:
    - Console: DEBUG level (full visibility while attached).
    - File:    logs/paper_live.log with rotation
               (10 MB per file, 5 backups → ~50 MB cap).
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — configure BEFORE importing tbot so module-level loggers inherit it
# ---------------------------------------------------------------------------

_LOG_DIR  = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "paper_live.log"

_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
)

_root = logging.getLogger()
_root.setLevel(logging.DEBUG)

# Avoid double-handlers if this script is re-imported in tests
for _h in list(_root.handlers):
    _root.removeHandler(_h)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.DEBUG)
_console.setFormatter(_FMT)
_root.addHandler(_console)

_file = RotatingFileHandler(
    _LOG_FILE,
    maxBytes=10 * 1024 * 1024,   # 10 MB
    backupCount=5,               # keep 5 rotated files (~50 MB total)
    encoding="utf-8",
)
_file.setLevel(logging.INFO)     # less chatty on disk than console
_file.setFormatter(_FMT)
_root.addHandler(_file)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

from tbot.live.runner import LiveRunner

try:
    runner = LiveRunner.build()
    runner.start()
except KeyboardInterrupt:
    logger.info("Interrupted by user — exiting.")
except Exception:
    # Last-resort safety net so the crash + traceback ends up in both logs.
    logger.exception("LiveRunner crashed at top level")
    try:
        from tbot.monitoring.alerts import alert_critical_error
        import traceback
        alert_critical_error(traceback.format_exc())
    except Exception:
        pass
    raise
