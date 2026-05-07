"""
Live trading runner — the single integration point for paper trading.

Chain:
    PriceStream (OANDA ticks → M5 candles)
        → LiveRunner._on_candle()
            → AITradingAgentV2 (SMC + ML scoring)
                → RiskState (circuit breaker / daily cap / equity floor)
                    → Executor (OANDA paper order + DB write)

Usage:
    runner = LiveRunner.build()   # reads credentials from cfg
    runner.start()                # blocks until Ctrl+C
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections import deque
from datetime import datetime, timezone

import pandas as pd

import json

from tbot.broker.executor import Executor
from tbot.broker.oanda_client import MarketClosedError, OandaClient
from tbot.broker.stream import PriceStream
from tbot.config import cfg
from tbot.core.agent_v2 import AITradingAgentV2
from tbot.db.models import RLExperience, Signal as SignalModel, Trade
from tbot.db.session import get_session, init_db
from tbot.risk.manager import RiskManager
from tbot.risk.state import RiskState

logger = logging.getLogger(__name__)

_INSTRUMENT    = "XAU_USD"
_WINDOW_SIZE   = 1_000   # rolling candle window fed to the agent
_MIN_HISTORY   = 100     # candles needed before the agent starts scoring


class LiveRunner:
    """
    Orchestrates the full live paper-trading loop.

    Args:
        client:      authenticated OandaClient
        instrument:  trading instrument (default XAU_USD)
        window_size: rolling candle window size for the agent
        min_confidence: ML score threshold (passed to agent)
    """

    def __init__(
        self,
        client:         OandaClient,
        instrument:     str   = _INSTRUMENT,
        window_size:    int   = _WINDOW_SIZE,
        min_confidence: float = 0.60,
    ) -> None:
        self._client     = client
        self._instrument = instrument
        self._candles:   deque = deque(maxlen=window_size)
        self._agent      = AITradingAgentV2(min_confidence=min_confidence)
        self._stream     = PriceStream(client, instrument=instrument)
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        instrument:     str   = _INSTRUMENT,
        min_confidence: float = 0.60,
    ) -> "LiveRunner":
        """Create a runner from config — reads credentials from .env automatically."""
        init_db()
        client = OandaClient()
        return cls(client=client, instrument=instrument, min_confidence=min_confidence)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the live loop. Blocks until Ctrl+C or stop() is called.

        What happens each M5 bar:
          1. New candle appended to rolling window
          2. Agent runs on the window (SMC + ML scoring)
          3. Only signals at the latest candle timestamp are considered
          4. Each signal gated through RiskState
          5. Approved signals sent to Executor → OANDA paper order
        """
        # Register Ctrl+C handler
        signal.signal(signal.SIGINT,  lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        logger.info("LiveRunner starting — instrument=%s", self._instrument)
        try:
            account = self._client.get_account()
            logger.info(
                "Account %s  NAV=$%.2f  Balance=$%.2f",
                account.account_id, account.nav, account.balance,
            )
        except Exception:
            # Don't abort startup on a transient REST blip — the loop will retry
            # equity fetches per signal anyway.
            logger.exception("Initial account fetch failed — continuing anyway")

        # Backfill warmup window from OANDA REST so a restart doesn't lose
        # the 8-hour warmup. Best-effort: if it fails, we fall back to the
        # old behavior (build up from the live stream).
        self._backfill_warmup()

        # Start price stream in background thread
        stream_thread = self._stream.start_background(self._on_candle)
        logger.info("Price stream started (background thread)")

        # Start background sync loop (closes trades + creates RL experiences)
        sync_thread = threading.Thread(target=self._sync_loop, daemon=True, name="sync-loop")
        sync_thread.start()
        logger.info("Sync loop started (5-min interval)")

        # Block main thread until stop() is called
        try:
            while not self._stop_event.is_set():
                time.sleep(1)
        finally:
            self._shutdown(stream_thread)

    def _shutdown(self, stream_thread: threading.Thread | None) -> None:
        """Best-effort graceful shutdown — never raises."""
        try:
            self._stream.stop()
        except Exception:
            logger.exception("Error stopping price stream")

        if stream_thread is not None:
            try:
                stream_thread.join(timeout=5)
            except Exception:
                logger.exception("Error joining stream thread")

        # Log any open OANDA positions so the operator can reconcile manually.
        # We deliberately do NOT auto-close positions here: stops/take-profits
        # remain attached server-side, and unilaterally flattening on Ctrl+C
        # could realise unintended losses.
        try:
            open_positions = self._client.get_open_positions()
            if open_positions:
                logger.warning(
                    "Shutdown: %d open OANDA position(s) remain (SL/TP still active server-side)",
                    len(open_positions),
                )
                for p in open_positions:
                    logger.warning("  open position: %s", p.get("instrument", "?"))
        except Exception:
            logger.exception("Could not fetch open positions during shutdown")

        logger.info("LiveRunner stopped.")

    def stop(self) -> None:
        """Signal the runner to shut down cleanly."""
        logger.info("Shutdown requested …")
        self._stop_event.set()

    def _backfill_warmup(self) -> None:
        """
        Pre-populate the candle deque with the last `_MIN_HISTORY` completed
        M5 candles from OANDA REST. Without this, every restart loses up to
        8 hours of warmup and the agent sits idle until enough live candles
        accumulate.

        Best-effort: any failure (REST blip, market closed, schema change)
        is caught and the runner continues with an empty deque. The next
        live candle just resumes the old behavior.

        Edge case: if the latest backfilled candle's bar boundary coincides
        with the bar the live stream is currently building, the live stream
        will eventually emit that same bar timestamp — the deque has
        maxlen=_WINDOW_SIZE so duplicates simply shift the window by one;
        the agent's signal de-dup is by timestamp so it won't double-fire.
        """
        try:
            candles = self._client.get_recent_candles(
                instrument  = self._instrument,
                granularity = "M5",
                count       = _MIN_HISTORY,
            )
            if not candles:
                logger.warning("Backfill: REST returned 0 candles — starting cold")
                return

            for c in candles:
                self._candles.append(c)

            logger.info(
                "Backfill: pre-loaded %d candles (oldest=%s, newest=%s) — "
                "agent ready immediately, no warmup wait.",
                len(candles),
                candles[0]["timestamp"],
                candles[-1]["timestamp"],
            )
        except Exception:
            logger.exception(
                "Backfill failed — falling back to live-only warmup "
                "(will need %d live candles before agent activates)",
                _MIN_HISTORY,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_candle(self, candle: dict) -> None:
        """Called by PriceStream once per completed M5 bar.

        This is the top-level callback running in the stream thread.  ANY
        exception here would bubble up into PriceStream's loop (which already
        catches it), but to keep state consistent we also handle it locally so
        a single bad candle / bad model inference does not lose subsequent
        candles' state.
        """
        try:
            self._candles.append(candle)

            n = len(self._candles)
            logger.debug(
                "Candle received [%d/%d]  ts=%s  C=%.3f",
                n, _MIN_HISTORY, candle["timestamp"], candle["close"],
            )

            if n < _MIN_HISTORY:
                return  # not enough history for indicators yet

            df = pd.DataFrame(list(self._candles))

            # Ensure timestamp column is UTC-aware
            if not pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # Run agent on full window — wrap separately so a model/SMC
            # inference bug on one candle doesn't kill the loop.
            try:
                self._agent.run_on_df(df)
            except Exception:
                logger.exception("Agent inference failed on candle %s — skipping", candle.get("timestamp"))
                return

            # Filter to signals at the most recent candle only
            latest_ts = candle["timestamp"]
            if not isinstance(latest_ts, datetime):
                latest_ts = pd.Timestamp(latest_ts).to_pydatetime()
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.replace(tzinfo=timezone.utc)

            all_signals = self._agent.current_signals
            new_signals = [
                s for s in all_signals
                if _ts_matches(s.get("timestamp"), latest_ts)
            ]

            logger.debug(
                "Agent: %d total signal(s) in window, %d at latest bar (%s)",
                len(all_signals), len(new_signals), latest_ts,
            )

            if not new_signals:
                return

            logger.info("%d signal(s) at %s", len(new_signals), latest_ts)
            self._process_signals(new_signals)
        except Exception:
            logger.exception("Unhandled error in _on_candle — continuing")

    def _process_signals(self, signals: list[dict]) -> None:
        """Gate each signal through risk and execute if approved."""
        try:
            equity = self._client.get_equity()
        except MarketClosedError:
            logger.info("Market closed — skipping signals")
            return
        except Exception:
            logger.exception("Could not fetch equity — skipping signals")
            return

        try:
            with get_session() as session:
                risk     = RiskState(session, instrument=self._instrument)
                rm       = RiskManager()
                executor = Executor(self._client, session, rm=rm, instrument=self._instrument)

                for sig in signals:
                    try:
                        ok, reason = risk.can_trade(equity=equity)
                        if not ok:
                            logger.info("Signal blocked: %s", reason)
                            continue

                        logger.info(
                            "Executing signal  side=%s  entry=%.3f  conf=%.3f",
                            sig.get("type"), sig.get("price"), sig.get("confidence", 0),
                        )
                        signal_id = self._write_live_signal(sig, session)
                        trade_id  = executor.execute(sig, equity=equity, signal_id=signal_id)
                        if trade_id:
                            logger.info("Trade DB id=%d  signal_id=%s", trade_id, signal_id)
                    except MarketClosedError:
                        logger.info("Market closed — remaining signals skipped")
                        return
                    except Exception:
                        logger.exception("Failed to process signal %s — continuing", sig.get("timestamp"))
                        continue
        except Exception:
            logger.exception("DB session error in _process_signals — signals dropped")

    def _write_live_signal(self, sig: dict, session) -> int | None:
        """Persist a live signal to DB and return its ID (for Trade.signal_id)."""
        try:
            ts = pd.Timestamp(sig.get("timestamp", datetime.now(timezone.utc)))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            features = sig.get("features")
            db_sig = SignalModel(
                instrument      = self._instrument,
                granularity     = "M5",
                timestamp       = ts.to_pydatetime(),
                side            = sig.get("type", "BUY"),
                entry_price     = float(sig.get("price", 0.0)),
                stop_loss       = float(sig.get("stop_loss", 0.0)),
                take_profit     = float(sig.get("take_profit", 0.0)),
                source_strategy = "smc_v2_live",
                model_score     = float(sig.get("confidence", 0.0)),
                features_json   = json.dumps(features) if features else None,
            )
            session.add(db_sig)
            session.flush()
            return db_sig.id
        except Exception:
            logger.exception("Could not write live signal to DB")
            return None

    def _sync_loop(self) -> None:
        """
        Background thread: every 5 minutes sync closed trades from OANDA
        and create RLExperience rows for newly closed trades.
        """
        from sqlalchemy import exists, not_, select
        from tbot.rl.experience_writer import create_experience

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=300)   # 5-minute interval
            if self._stop_event.is_set():
                break
            try:
                with get_session() as session:
                    executor = Executor(self._client, session, instrument=self._instrument)
                    updated  = executor.sync_closed_trades()
                    if updated:
                        logger.info("Sync: %d trade(s) closed", updated)

                    # Create RL experiences for closed trades that don't have one yet
                    stmt = (
                        select(Trade)
                        .where(Trade.instrument == self._instrument)
                        .where(Trade.exit_time.is_not(None))
                        .where(Trade.exit_price.is_not(None))
                        .where(Trade.net_pnl.is_not(None))
                        .where(not_(exists().where(RLExperience.trade_id == Trade.id)))
                    )
                    pending = session.execute(stmt).scalars().all()
                    for trade in pending:
                        create_experience(trade, session)
            except Exception:
                logger.exception("Error in sync loop — will retry in 5 minutes")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ts_matches(sig_ts, target: datetime) -> bool:
    """Return True if sig_ts refers to the same UTC minute as target."""
    try:
        t = pd.Timestamp(sig_ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.replace(second=0, microsecond=0) == target.replace(second=0, microsecond=0)
    except Exception:
        return False
