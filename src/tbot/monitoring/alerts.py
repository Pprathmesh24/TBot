"""
Slack alert helpers for TBot.

Each public function sends one Slack message and returns immediately.
Failures are logged but never propagate — a Slack outage must not crash the loop.

Usage:
    from tbot.monitoring.alerts import alert_trade_filled, alert_circuit_breaker
    alert_trade_filled(side="BUY", entry=2352.10, units=20, confidence=0.74)
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone

from tbot.config import cfg

logger = logging.getLogger(__name__)

_EMOJI = {
    "BUY":  ":chart_with_upwards_trend:",
    "SELL": ":chart_with_downwards_trend:",
}


# ---------------------------------------------------------------------------
# Public alert functions
# ---------------------------------------------------------------------------

def alert_trade_filled(
    side: str,
    entry: float,
    units: float,
    confidence: float,
    oanda_id: str = "",
) -> None:
    """Send a fill notification when a paper order is placed."""
    emoji = _EMOJI.get(side.upper(), ":money_with_wings:")
    text = (
        f"{emoji} *Trade Filled* — {side} XAU/USD\n"
        f"Entry: `{entry:.3f}` · Units: `{units:.0f}` · "
        f"Conf: `{confidence:.2f}` · OANDA id: `{oanda_id}`"
    )
    _post(text)


def alert_circuit_breaker(consecutive_losses: int, cooldown_hours: int) -> None:
    """Send a warning when the risk circuit breaker fires."""
    text = (
        f":warning: *Circuit Breaker Fired*\n"
        f"{consecutive_losses} consecutive losses — trading paused for {cooldown_hours}h"
    )
    _post(text)


def alert_daily_loss_cap(daily_pnl: float, cap_pct: float) -> None:
    """Send a warning when the daily loss cap is hit."""
    text = (
        f":no_entry: *Daily Loss Cap Hit*\n"
        f"Daily P&L: `{daily_pnl:+.2f}` · Cap: `{cap_pct:.1f}%` — no more trades today"
    )
    _post(text)


def alert_stream_reconnect(attempt: int) -> None:
    """Send an info message when the price stream reconnects."""
    text = f":arrows_counterclockwise: Price stream reconnected (attempt {attempt})"
    _post(text)


def alert_critical_error(error: str) -> None:
    """Send an error alert for unexpected crashes."""
    text = f":rotating_light: *Critical Error*\n```{error[:500]}```"
    _post(text)


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _post(text: str) -> None:
    """POST a plain-text message to the configured Slack webhook. Swallows errors."""
    url = cfg.slack_webhook_url
    if not url:
        logger.debug("SLACK_WEBHOOK_URL not set — alert skipped: %s", text[:80])
        return

    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                logger.warning("Slack returned HTTP %d", resp.status)
            else:
                logger.debug("Slack alert sent: %s", text[:80])
    except urllib.error.URLError as exc:
        logger.warning("Slack alert failed (network): %s", exc)
    except Exception as exc:
        logger.warning("Slack alert failed: %s", exc)
