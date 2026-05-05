"""
TBot Watchdog — autonomous overnight supervisor.

Monitors the paper trading system and auto-recovers from:
  - Silent stream stalls (no candle close in >6 min)
  - Runner process death
  - OANDA disconnect errors in log

On anomaly:
  1. Sends Slack alert
  2. Restarts the runner
  3. Writes a structured incident report to logs/incidents/

Usage (standalone — manages its own runner subprocess):
    .venv/bin/python scripts/watchdog.py

Or just the watchdog watching an existing runner PID:
    .venv/bin/python scripts/watchdog.py --pid 12345
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  watchdog  %(message)s",
)
logger = logging.getLogger("watchdog")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).parent.parent
LOG_FILE       = PROJECT_ROOT / "logs" / "paper_live.log"
INCIDENT_DIR   = PROJECT_ROOT / "logs" / "incidents"
RUNNER_SCRIPT  = PROJECT_ROOT / "scripts" / "run_paper_live.py"
PYTHON         = PROJECT_ROOT / ".venv" / "bin" / "python"

STALL_THRESHOLD_SEC = 360   # 6 minutes — should see a candle every 5 min
CHECK_INTERVAL_SEC  = 30    # how often watchdog polls

CANDLE_PATTERN  = re.compile(r"Candle closed: (\S+ \S+)")
ERROR_PATTERNS  = [
    "ChunkedEncodingError",
    "Price stream disconnected",
    "LiveRunner crashed",
    "ConnectionError",
    "ReadTimeout",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _send_slack(text: str) -> None:
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        os.chdir(PROJECT_ROOT)
        from tbot.monitoring.alerts import alert_critical_error
        alert_critical_error(text)
    except Exception as exc:
        logger.warning("Slack alert failed: %s", exc)


def _last_candle_time() -> datetime | None:
    """Return the UTC timestamp of the most recent 'Candle closed' log line."""
    if not LOG_FILE.exists():
        return None
    try:
        # Read last 200 lines efficiently
        with open(LOG_FILE, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 8192)
            f.seek(-chunk, 2)
            tail = f.read().decode("utf-8", errors="replace")

        last_match = None
        for match in CANDLE_PATTERN.finditer(tail):
            last_match = match

        if last_match is None:
            return None

        ts_str = last_match.group(1)  # e.g. "2026-05-05 03:40:00+00:00"
        return datetime.fromisoformat(ts_str)
    except Exception as exc:
        logger.warning("Could not parse last candle time: %s", exc)
        return None


def _recent_errors() -> list[str]:
    """Return list of error pattern matches from the last 50 log lines."""
    if not LOG_FILE.exists():
        return []
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 4096)
            f.seek(-chunk, 2)
            tail = f.read().decode("utf-8", errors="replace")
        lines = tail.splitlines()[-50:]
        found = []
        for line in lines:
            for pat in ERROR_PATTERNS:
                if pat in line:
                    found.append(line.strip())
                    break
        return found
    except Exception:
        return []


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_incident(kind: str, detail: str, runner_pid: int | None) -> Path:
    INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now_utc().strftime("%Y-%m-%d_%H-%M-%S")
    path = INCIDENT_DIR / f"{ts}_{kind}.json"
    report = {
        "timestamp_utc": _now_utc().isoformat(),
        "kind":          kind,
        "detail":        detail,
        "runner_pid":    runner_pid,
        "log_tail":      _tail_log(30),
    }
    path.write_text(json.dumps(report, indent=2))
    logger.info("Incident report written → %s", path.name)
    return path


def _tail_log(n: int = 30) -> list[str]:
    if not LOG_FILE.exists():
        return []
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 8192)
            f.seek(-chunk, 2)
            tail = f.read().decode("utf-8", errors="replace")
        return tail.splitlines()[-n:]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Runner management
# ---------------------------------------------------------------------------

def _start_runner() -> subprocess.Popen:
    """Start run_paper_live.py as a subprocess and return the Popen handle."""
    proc = subprocess.Popen(
        [str(PYTHON), str(RUNNER_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,  # runner writes its own log file
        stderr=subprocess.DEVNULL,
    )
    logger.info("Runner started — PID %d", proc.pid)
    return proc


def _stop_runner(proc: subprocess.Popen) -> None:
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _restart_runner(proc: subprocess.Popen | None, reason: str) -> subprocess.Popen:
    logger.warning("Restarting runner — reason: %s", reason)
    _stop_runner(proc)
    time.sleep(2)
    new_proc = _start_runner()
    return new_proc


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthStatus:
    def __init__(self) -> None:
        self.ok          = True
        self.issues: list[str] = []

    def fail(self, msg: str) -> None:
        self.ok = False
        self.issues.append(msg)

    def __str__(self) -> str:
        return " | ".join(self.issues) if self.issues else "OK"


def check_health(runner_pid: int | None) -> HealthStatus:
    status = HealthStatus()
    now = _now_utc()

    # 1. Process alive
    if runner_pid is not None and not _is_alive(runner_pid):
        status.fail(f"runner PID {runner_pid} is dead")

    # 2. Last candle timestamp
    last_ts = _last_candle_time()
    if last_ts is None:
        # No candle yet — only a problem if runner has been up > 10 min
        pass
    else:
        gap_sec = (now - last_ts.replace(tzinfo=timezone.utc) if last_ts.tzinfo is None
                   else now - last_ts)
        gap_sec = gap_sec.total_seconds()
        if gap_sec > STALL_THRESHOLD_SEC:
            status.fail(f"stream stall — last candle {gap_sec:.0f}s ago (>{STALL_THRESHOLD_SEC}s)")

    # 3. Recent error lines
    errors = _recent_errors()
    if errors:
        status.fail(f"{len(errors)} recent error(s): {errors[-1][:80]}")

    return status


# ---------------------------------------------------------------------------
# Main watchdog loop
# ---------------------------------------------------------------------------

def run_watchdog(external_pid: int | None = None) -> None:
    """
    Main loop.

    If external_pid is given, watchdog monitors that process and does NOT
    restart it autonomously (use this when runner was started externally).

    If external_pid is None, watchdog owns the runner subprocess.
    """
    owns_runner = external_pid is None
    runner_proc: subprocess.Popen | None = None
    runner_pid: int | None = external_pid

    if owns_runner:
        runner_proc = _start_runner()
        runner_pid  = runner_proc.pid
        time.sleep(15)  # give runner time to start streaming

    logger.info(
        "Watchdog active — checking every %ds · stall threshold %ds · PID %s",
        CHECK_INTERVAL_SEC, STALL_THRESHOLD_SEC,
        runner_pid or "external",
    )

    consecutive_failures = 0

    try:
        while True:
            time.sleep(CHECK_INTERVAL_SEC)

            # If we own the runner, sync PID in case it was restarted
            if owns_runner and runner_proc is not None:
                runner_pid = runner_proc.pid

            status = check_health(runner_pid)

            if status.ok:
                consecutive_failures = 0
                logger.debug("Health OK — last candle: %s", _last_candle_time())
                continue

            consecutive_failures += 1
            issue_str = str(status)
            logger.warning("Health check FAILED (%dx): %s", consecutive_failures, issue_str)

            # Write incident report
            _write_incident(
                kind="stall" if "stall" in issue_str else "error",
                detail=issue_str,
                runner_pid=runner_pid,
            )

            # Alert Slack
            _send_slack(
                f":rotating_light: *TBot Watchdog Alert*\n"
                f"Issue: {issue_str}\n"
                f"Consecutive failures: {consecutive_failures}\n"
                f"{'Auto-restarting runner…' if owns_runner else 'Manual intervention needed.'}"
            )

            # Auto-restart only if we own the runner
            if owns_runner:
                runner_proc = _restart_runner(runner_proc, issue_str)
                runner_pid  = runner_proc.pid
                consecutive_failures = 0
                time.sleep(15)  # wait for runner to warm up before next check
            else:
                logger.warning("External runner — not auto-restarting. Manual restart needed.")

    except KeyboardInterrupt:
        logger.info("Watchdog stopped by user.")
        if owns_runner:
            _stop_runner(runner_proc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TBot autonomous watchdog")
    parser.add_argument(
        "--pid", type=int, default=None,
        help="Monitor an already-running runner PID (watchdog will NOT restart it)"
    )
    args = parser.parse_args()
    run_watchdog(external_pid=args.pid)


if __name__ == "__main__":
    main()
