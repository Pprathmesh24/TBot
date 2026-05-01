"""
Risk manager — gates every signal before it becomes an order.

Two public methods:

    can_trade(equity, recent_trades) → (bool, reason_str)
        Checks three rules:
        1. Circuit breaker: N consecutive losses → cooldown_hours pause
        2. Daily loss cap:  today's P&L < -daily_loss_cap × equity → stop
        3. Equity floor:    equity < min_equity → stop (catastrophic loss)

    position_size(equity, entry, stop) → float (units)
        ATR-based: risk exactly risk_pct of equity per trade.
        units = (equity × risk_pct) / abs(entry − stop)

Usage:
    rm = RiskManager()
    ok, reason = rm.can_trade(equity=10_000, recent_trades=trades)
    if ok:
        units = rm.position_size(equity=10_000, entry=2350.0, stop=2345.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Tuple


@dataclass
class TradeRecord:
    """Minimal trade info needed by RiskManager. Compatible with DB Trade model."""
    net_pnl:   float
    exit_time: datetime


@dataclass
class RiskManager:
    """
    Stateless risk gate — all state is derived from the trades list passed in.
    This makes it trivially testable and avoids hidden mutable state.

    Args:
        risk_pct:               fraction of equity to risk per trade (default 1%)
        max_consecutive_losses: circuit breaker threshold (default 3)
        cooldown_hours:         hours to wait after circuit breaker fires (default 24)
        daily_loss_cap:         max fraction of equity losable per day (default 3%)
        min_equity:             absolute equity floor below which trading halts
        min_position:           minimum trade units (broker minimum lot)
        max_position:           maximum trade units (hard cap)
    """
    risk_pct:               float = 0.01
    max_consecutive_losses: int   = 3
    cooldown_hours:         int   = 24
    daily_loss_cap:         float = 0.03
    min_equity:             float = 1_000.0
    min_position:           float = 0.01
    max_position:           float = 100.0

    def can_trade(
        self,
        equity:       float,
        recent_trades: List[TradeRecord],
        now:          datetime | None = None,
    ) -> Tuple[bool, str]:
        """
        Check whether a new trade is allowed.

        Args:
            equity:        current account equity in USD
            recent_trades: list of completed trades, sorted ascending by exit_time
            now:           current UTC time (injectable for testing; defaults to utcnow)

        Returns:
            (True, "ok") if allowed
            (False, reason_string) if blocked
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # ---- 1. Equity floor ----
        if equity < self.min_equity:
            return False, f"equity ${equity:.0f} below minimum ${self.min_equity:.0f}"

        # ---- 2. Circuit breaker: consecutive losses ----
        consecutive = _count_consecutive_losses(recent_trades)
        if consecutive >= self.max_consecutive_losses:
            # Check if cooldown has expired
            last_trade = recent_trades[-1] if recent_trades else None
            if last_trade is not None:
                last_exit = _ensure_utc(last_trade.exit_time)
                cooldown_end = last_exit + timedelta(hours=self.cooldown_hours)
                if now < cooldown_end:
                    remaining = cooldown_end - now
                    hours_left = remaining.total_seconds() / 3600
                    return False, (
                        f"circuit breaker: {consecutive} consecutive losses — "
                        f"cooldown ends in {hours_left:.1f}h"
                    )

        # ---- 3. Daily loss cap ----
        today_pnl = _daily_pnl(recent_trades, now)
        daily_loss_limit = -self.daily_loss_cap * equity
        if today_pnl < daily_loss_limit:
            return False, (
                f"daily loss cap: today P&L ${today_pnl:.2f} "
                f"exceeds -{self.daily_loss_cap:.0%} of equity"
            )

        return True, "ok"

    def position_size(
        self,
        equity: float,
        entry:  float,
        stop:   float,
    ) -> float:
        """
        Compute trade size so that hitting the stop costs exactly risk_pct of equity.

        Args:
            equity: current account equity in USD
            entry:  planned entry price
            stop:   stop-loss price

        Returns:
            units to trade, clamped to [min_position, max_position]
        """
        risk_dollars = equity * self.risk_pct
        stop_distance = abs(entry - stop)

        if stop_distance <= 0:
            return self.min_position

        units = risk_dollars / stop_distance
        return float(max(self.min_position, min(self.max_position, units)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_consecutive_losses(trades: List[TradeRecord]) -> int:
    """Count unbroken losing trades from the most recent backwards."""
    count = 0
    for trade in reversed(trades):
        if trade.net_pnl < 0:
            count += 1
        else:
            break
    return count


def _daily_pnl(trades: List[TradeRecord], now: datetime) -> float:
    """Sum P&L for trades that closed today (UTC calendar day)."""
    today = now.date()
    return sum(
        t.net_pnl for t in trades
        if _ensure_utc(t.exit_time).date() == today
    )


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
