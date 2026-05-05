"""
TBot Supervisor — unified entry point for unattended/overnight runs.

Starts the watchdog in "owns runner" mode: the watchdog launches
run_paper_live.py as a subprocess, monitors it, and auto-restarts on
stall or crash.

Usage:
    .venv/bin/python scripts/run_supervisor.py

Logs:
    logs/paper_live.log   — runner output (rotated, 10MB x5)
    logs/watchdog.log     — watchdog events
    logs/incidents/       — one JSON file per incident
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_FMT = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

_root = logging.getLogger()
_root.setLevel(logging.DEBUG)
for _h in list(_root.handlers):
    _root.removeHandler(_h)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.DEBUG)
_console.setFormatter(_FMT)
_root.addHandler(_console)

_file_watchdog = RotatingFileHandler(
    _LOG_DIR / "watchdog.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file_watchdog.setLevel(logging.INFO)
_file_watchdog.setFormatter(_FMT)
_root.addHandler(_file_watchdog)

# ---------------------------------------------------------------------------
# Delegate to watchdog (owns_runner=True mode)
# ---------------------------------------------------------------------------

# scripts/ is not a package — add it to path so we can import watchdog directly
sys.path.insert(0, str(Path(__file__).parent))
from watchdog import run_watchdog  # noqa: E402

if __name__ == "__main__":
    logging.getLogger(__name__).info("=== TBot Supervisor starting ===")
    run_watchdog(external_pid=None)   # owns_runner=True — watchdog manages the runner subprocess
    logging.getLogger(__name__).info("=== TBot Supervisor stopped ===")
