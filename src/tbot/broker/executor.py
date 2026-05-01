"""
Signal executor — converts an approved signal into an OANDA paper order
and writes a Trade row to the database.

Usage:
    executor = Executor(client, session)
    trade_id = executor.execute(signal, equity)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from tbot.broker.oanda_client import OandaClient
from tbot.db.models import Trade
from tbot.risk.manager import RiskManager

logger = logging.getLogger(__name__)

_INSTRUMENT = "XAU_USD"


class Executor:
    """
    Places paper orders on OANDA and persists Trade rows in the DB.

    Args:
        client:     authenticated OandaClient
        session:    SQLAlchemy Session (caller owns lifecycle)
        rm:         RiskManager used for position sizing
        instrument: trading instrument (default XAU_USD)
    """

    def __init__(
        self,
        client:     OandaClient,
        session:    Session,
        rm:         RiskManager | None = None,
        instrument: str = _INSTRUMENT,
    ) -> None:
        self._client     = client
        self._session    = session
        self._rm         = rm or RiskManager()
        self._instrument = instrument

    def execute(self, signal: dict, equity: float) -> int | None:
        """
        Place a market order for the given signal and write a Trade to DB.

        Args:
            signal: dict with keys: type (BUY/SELL), price, stop_loss,
                    take_profit, confidence, timestamp
            equity: current account NAV (used for position sizing)

        Returns:
            DB Trade.id if successful, None if order placement failed.
        """
        side       = signal.get("type", "BUY")
        entry      = float(signal.get("price", 0.0))
        stop_loss  = float(signal.get("stop_loss", 0.0))
        take_profit = float(signal.get("take_profit", 0.0))

        # Position size from RiskManager
        units = self._rm.position_size(equity=equity, entry=entry, stop=stop_loss)
        if side == "SELL":
            units = -units  # OANDA uses negative units for short orders

        logger.info(
            "Executing %s  entry=%.3f  SL=%.3f  TP=%.3f  units=%.2f",
            side, entry, stop_loss, take_profit, units,
        )

        try:
            oanda_id = self._client.place_market_order(
                instrument  = self._instrument,
                units       = units,
                stop_loss   = stop_loss  if stop_loss  > 0 else None,
                take_profit = take_profit if take_profit > 0 else None,
                comment     = f"tbot:{signal.get('source', 'smc')}",
            )
        except Exception:
            logger.exception("Order placement failed for signal %s", signal.get("timestamp"))
            return None

        logger.info("Order placed → OANDA id=%s", oanda_id)

        # Write Trade row — exit fields filled in later when trade closes
        trade = Trade(
            instrument      = self._instrument,
            side            = "BUY" if units > 0 else "SELL",
            units           = abs(units),
            entry_time      = datetime.now(timezone.utc),
            entry_price     = entry,
            oanda_order_id  = oanda_id,
        )
        self._session.add(trade)
        self._session.flush()

        logger.info("Trade written to DB  id=%d", trade.id)
        return trade.id

    def sync_closed_trades(self) -> int:
        """
        Query OANDA for any trades that have closed since last sync and
        update their exit fields in the DB.

        Returns count of trades updated.
        """
        from sqlalchemy import select

        # Find open DB trades (no exit_time yet)
        stmt = (
            select(Trade)
            .where(Trade.instrument == self._instrument)
            .where(Trade.exit_time.is_(None))
            .where(Trade.oanda_order_id.is_not(None))
        )
        open_trades = self._session.execute(stmt).scalars().all()

        if not open_trades:
            return 0

        # Get currently open trades from OANDA
        try:
            oanda_open = {t["id"] for t in self._client.get_open_trades()}
        except Exception:
            logger.exception("Could not fetch open trades from OANDA")
            return 0

        updated = 0
        for trade in open_trades:
            if trade.oanda_order_id in oanda_open:
                continue  # still open

            # Trade closed on OANDA side — mark exit in DB
            trade.exit_time   = datetime.now(timezone.utc)
            trade.exit_reason = "CLOSED"
            updated += 1
            logger.info("Trade %d closed (OANDA id=%s)", trade.id, trade.oanda_order_id)

        return updated
