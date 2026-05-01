"""
DB-backed risk state — thin wrapper around RiskManager that loads trades
from the database so callers don't have to.

Usage:
    # Live trading
    rs = RiskState(session, instrument="XAU_USD")
    ok, reason = rs.can_trade(equity=10_000)
    if ok:
        units = rs.position_size(equity=10_000, entry=2350.0, stop=2345.0)

    # Backtest (scoped to one run)
    rs = RiskState(session, instrument="XAU_USD", backtest_run_id=run.id)
    ok, reason = rs.can_trade(equity=portfolio.equity)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from tbot.db.models import Trade
from tbot.risk.manager import RiskManager, TradeRecord


class RiskState:
    """
    Loads completed trades from the DB and delegates to RiskManager.

    Args:
        session:          SQLAlchemy Session (caller owns the lifecycle)
        instrument:       filter trades by instrument, e.g. "XAU_USD"
        backtest_run_id:  if given, only trades from this run are loaded
                          (backtest mode); if None, loads all completed
                          live trades for the instrument
        rm:               RiskManager instance; defaults to RiskManager()
    """

    def __init__(
        self,
        session:          Session,
        instrument:       str = "XAU_USD",
        backtest_run_id:  int | None = None,
        rm:               RiskManager | None = None,
    ) -> None:
        self._session         = session
        self._instrument      = instrument
        self._backtest_run_id = backtest_run_id
        self._rm              = rm or RiskManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_trade(
        self,
        equity: float,
        now:    datetime | None = None,
    ) -> Tuple[bool, str]:
        """Check whether a new trade is allowed given the current DB state."""
        trades = self._load_trades()
        return self._rm.can_trade(equity=equity, recent_trades=trades, now=now)

    def position_size(
        self,
        equity: float,
        entry:  float,
        stop:   float,
    ) -> float:
        """Compute position size — delegates directly to RiskManager (no DB needed)."""
        return self._rm.position_size(equity=equity, entry=entry, stop=stop)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_trades(self) -> List[TradeRecord]:
        """
        Query the trades table and return completed trades as TradeRecord list,
        sorted ascending by exit_time (required by RiskManager helpers).

        A "completed" trade has both exit_time and net_pnl populated.
        Open/in-flight trades are excluded.
        """
        stmt = (
            select(Trade.net_pnl, Trade.exit_time)
            .where(Trade.instrument == self._instrument)
            .where(Trade.exit_time.is_not(None))
            .where(Trade.net_pnl.is_not(None))
        )

        if self._backtest_run_id is not None:
            stmt = stmt.where(Trade.backtest_run_id == self._backtest_run_id)

        stmt = stmt.order_by(Trade.exit_time)
        rows = self._session.execute(stmt).all()

        return [
            TradeRecord(
                net_pnl=row.net_pnl,
                exit_time=_ensure_utc(row.exit_time),
            )
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
