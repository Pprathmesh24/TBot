"""
TBot Morning Summary — review overnight incidents and trading activity.

Reads incident reports from logs/incidents/ and recent log activity,
then prints a structured summary of what happened overnight.

Usage:
    .venv/bin/python scripts/morning_summary.py
    .venv/bin/python scripts/morning_summary.py --hours 12   # last 12h only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent
INCIDENT_DIR  = PROJECT_ROOT / "logs" / "incidents"
LOG_FILE      = PROJECT_ROOT / "logs" / "paper_live.log"
WATCHDOG_LOG  = PROJECT_ROOT / "logs" / "watchdog.log"

CANDLE_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*Candle received \[(\d+)/100\]")
SIGNAL_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*signal\(s\) at")
TRADE_RE  = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*Trade Filled|trade.*filled", re.I)
ERROR_RE  = re.compile(r"(ERROR|CRITICAL|disconnected|crashed|stall)", re.I)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_incidents(since: datetime) -> list[dict]:
    if not INCIDENT_DIR.exists():
        return []
    incidents = []
    for f in sorted(INCIDENT_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            ts = datetime.fromisoformat(data["timestamp_utc"])
            if ts >= since:
                incidents.append(data)
        except Exception:
            pass
    return incidents


def _read_log_tail(path: Path, hours: int) -> list[str]:
    if not path.exists():
        return []
    try:
        # Read last 100KB
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 100 * 1024)
            f.seek(-chunk, 2)
            content = f.read().decode("utf-8", errors="replace")
        cutoff = _now() - timedelta(hours=hours)
        lines = []
        for line in content.splitlines():
            # parse timestamp from start of line
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m:
                try:
                    ts = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        lines.append(line)
                except Exception:
                    lines.append(line)
            else:
                lines.append(line)
        return lines
    except Exception:
        return []


def _count_candles(lines: list[str]) -> int:
    max_n = 0
    for line in lines:
        m = CANDLE_RE.search(line)
        if m:
            max_n = max(max_n, int(m.group(2)))
    return max_n


def _count_errors(lines: list[str]) -> list[str]:
    return [l for l in lines if ERROR_RE.search(l)]


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main(hours: int = 8) -> None:
    since = _now() - timedelta(hours=hours)
    print(f"\nTBot Morning Summary — last {hours}h  ({since.strftime('%Y-%m-%d %H:%M')} UTC → now)")

    # ── Incidents ──────────────────────────────────────────────────
    _print_section("Incidents")
    incidents = _load_incidents(since)
    if not incidents:
        print("  ✓ No incidents logged overnight")
    else:
        print(f"  ⚠  {len(incidents)} incident(s) detected:\n")
        for inc in incidents:
            ts = inc.get("timestamp_utc", "?")[:19]
            kind = inc.get("kind", "?").upper()
            detail = inc.get("detail", "?")
            print(f"  [{ts}]  {kind}: {detail}")

    # ── Log activity ───────────────────────────────────────────────
    _print_section("Runner Log Activity")
    runner_lines = _read_log_tail(LOG_FILE, hours)
    if not runner_lines:
        print("  logs/paper_live.log — no recent lines (log file missing or empty)")
    else:
        candle_count = _count_candles(runner_lines)
        error_lines  = _count_errors(runner_lines)
        signal_lines = [l for l in runner_lines if SIGNAL_RE.search(l)]
        trade_lines  = [l for l in runner_lines if TRADE_RE.search(l)]

        status = "✓" if not error_lines else "⚠"
        print(f"  {status} Lines in window:   {len(runner_lines)}")
        print(f"  {'✓' if candle_count > 0 else '⚠'} Max candle warmup:  {candle_count}/100")
        print(f"  {'✓' if not error_lines else '⚠'} Error lines:        {len(error_lines)}")
        print(f"    Signal detections: {len(signal_lines)}")
        print(f"    Trade fills:       {len(trade_lines)}")

        if error_lines:
            print("\n  Recent errors:")
            for l in error_lines[-5:]:
                print(f"    {l[:120]}")

    # ── Watchdog log ───────────────────────────────────────────────
    _print_section("Watchdog Activity")
    watchdog_lines = _read_log_tail(WATCHDOG_LOG, hours)
    if not watchdog_lines:
        print("  logs/watchdog.log — no entries (watchdog may not have been running)")
    else:
        restarts = [l for l in watchdog_lines if "Restarting runner" in l]
        alerts   = [l for l in watchdog_lines if "FAILED" in l]
        print(f"  Health check failures: {len(alerts)}")
        print(f"  Auto-restarts:         {len(restarts)}")
        if restarts:
            print("\n  Restart events:")
            for l in restarts[-3:]:
                print(f"    {l[:120]}")

    # ── Recommendations ────────────────────────────────────────────
    _print_section("Recommendations")
    recs = []

    if incidents:
        recs.append(f"• {len(incidents)} incident(s) overnight — review logs/incidents/ for details")
    if len(_count_errors(runner_lines)) > 3:
        recs.append("• High error rate — consider reviewing OANDA connection stability")
    if not runner_lines:
        recs.append("• Runner log is empty — confirm the paper trading system is running")
    if candle_count < 100 if runner_lines else True:
        recs.append("• Warmup not complete — ML agent not yet scoring signals")

    if not recs:
        print("  ✓ System looks healthy overnight. No action needed.")
    else:
        for r in recs:
            print(f"  {r}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TBot overnight summary")
    parser.add_argument("--hours", type=int, default=8, help="Hours to look back (default 8)")
    args = parser.parse_args()
    main(hours=args.hours)
