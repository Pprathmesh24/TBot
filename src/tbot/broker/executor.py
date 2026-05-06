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

from tbot.broker.oanda_client import MarketClosedError, OandaClient
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

    def execute(self, signal: dict, equity: float, signal_id: int | None = None) -> int | None:
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
        except MarketClosedError:
            logger.info("Market closed — order skipped.")
            return None
        except Exception:
            logger.exception("Order placement failed for signal %s", signal.get("timestamp"))
            return None

        logger.info("Order placed → OANDA id=%s", oanda_id)

        # Write Trade row — exit fields filled in later when trade closes.
        # If the DB write fails (locked / corrupted), the order is already on
        # OANDA — we MUST log loudly so the trade is reconciled manually, but
        # we should NOT crash the live runner.
        try:
            trade = Trade(
                instrument      = self._instrument,
                signal_id       = signal_id,
                side            = "BUY" if units > 0 else "SELL",
                units           = abs(units),
                entry_time      = datetime.now(timezone.utc),
                entry_price     = entry,
                oanda_order_id  = oanda_id,
            )
            self._session.add(trade)
            self._session.flush()
        except Exception:
            logger.exception(
                "DB write FAILED after OANDA fill — orphan trade! "
                "OANDA id=%s side=%s units=%.2f entry=%.3f",
                oanda_id, side, units, entry,
            )
            try:
                from tbot.monitoring.alerts import alert_critical_error
                alert_critical_error(
                    f"DB write failed after OANDA fill (orphan trade): "
                    f"oanda_id={oanda_id} side={side} units={units} entry={entry}"
                )
            except Exception:
                logger.debug("alert_critical_error failed", exc_info=True)
            return None

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

            # Fetch real close price + time from OANDA trade details
            try:
                details = self._client.get_trade_details(trade.oanda_order_id)
                close_price = details.get("averageClosePrice")
                close_time  = details.get("closeTime")
                realized_pl = details.get("realizedPL")

                trade.exit_price  = float(close_price) if close_price else None
                trade.exit_time   = _parse_oanda_time(close_time) or datetime.now(timezone.utc)
                trade.exit_reason = _infer_exit_reason(details)
                if realized_pl is not None:
                    trade.gross_pnl = float(realized_pl)
                    trade.net_pnl   = float(realized_pl)
            except Exception:
                logger.exception("Could not fetch details for OANDA trade %s — using now()", trade.oanda_order_id)
                trade.exit_time   = datetime.now(timezone.utc)
                trade.exit_reason = "CLOSED"

            updated += 1
            logger.info(
                "Trade %d closed  OANDA=%s  exit_price=%s  pnl=%s",
                trade.id, trade.oanda_order_id, trade.exit_price, trade.net_pnl,
            )

        return updated


def _parse_oanda_time(time_str: str | None) -> datetime | None:
    """Parse OANDA ISO timestamp (e.g. '2026-05-05T14:23:00.000000000Z') → UTC datetime."""
    if not time_str:
        return None
    try:
        from datetime import timezone as tz
        # oandapyV20 timestamps end in nanoseconds — truncate to microseconds
        ts = time_str[:26].rstrip("Z").rstrip(".")
        dt = datetime.fromisoformat(ts).replace(tzinfo=tz.utc)
        return dt
    except Exception:
        return None


def _infer_exit_reason(details: dict) -> str:
    """Map OANDA trade state/close reason to our exit_reason vocab."""
    # closingTransactionIDs tells us which transaction(s) closed the trade.
    # The transaction type is not directly in TradeDetails, but realizedPL sign
    # vs stopLoss / takeProfit levels gives us a strong hint.
    stop_loss   = details.get("stopLossOrder", {})
    take_profit = details.get("takeProfitOrder", {})

    if stop_loss.get("state") == "FILLED":
        return "SL"
    if take_profit.get("state") == "FILLED":
        return "TP"
    return "MANUAL"
