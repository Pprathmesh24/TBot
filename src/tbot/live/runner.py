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

from tbot.broker.executor import Executor
from tbot.broker.oanda_client import OandaClient
from tbot.broker.stream import PriceStream
from tbot.config import cfg
from tbot.core.agent_v2 import AITradingAgentV2
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
        account = self._client.get_account()
        logger.info(
            "Account %s  NAV=$%.2f  Balance=$%.2f",
            account.account_id, account.nav, account.balance,
        )

        # Start price stream in background thread
        stream_thread = self._stream.start_background(self._on_candle)
        logger.info("Price stream started (background thread)")

        # Block main thread until stop() is called
        try:
            while not self._stop_event.is_set():
                time.sleep(1)
        finally:
            self._stream.stop()
            stream_thread.join(timeout=5)
            logger.info("LiveRunner stopped.")

    def stop(self) -> None:
        """Signal the runner to shut down cleanly."""
        logger.info("Shutdown requested …")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_candle(self, candle: dict) -> None:
        """Called by PriceStream once per completed M5 bar."""
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

        # Run agent on full window
        self._agent.run_on_df(df)

        # Filter to signals at the most recent candle only
        latest_ts = candle["timestamp"]
        if not isinstance(latest_ts, datetime):
            latest_ts = pd.Timestamp(latest_ts).to_pydatetime()
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=timezone.utc)

        new_signals = [
            s for s in self._agent.current_signals
            if _ts_matches(s.get("timestamp"), latest_ts)
        ]

        if not new_signals:
            return

        logger.info("%d signal(s) at %s", len(new_signals), latest_ts)
        self._process_signals(new_signals)

    def _process_signals(self, signals: list[dict]) -> None:
        """Gate each signal through risk and execute if approved."""
        try:
            equity = self._client.get_equity()
        except Exception:
            logger.exception("Could not fetch equity — skipping signals")
            return

        with get_session() as session:
            risk  = RiskState(session, instrument=self._instrument)
            rm    = RiskManager()
            executor = Executor(self._client, session, rm=rm, instrument=self._instrument)

            for sig in signals:
                ok, reason = risk.can_trade(equity=equity)
                if not ok:
                    logger.info("Signal blocked: %s", reason)
                    continue

                logger.info(
                    "Executing signal  side=%s  entry=%.3f  conf=%.3f",
                    sig.get("type"), sig.get("price"), sig.get("confidence", 0),
                )
                trade_id = executor.execute(sig, equity=equity)
                if trade_id:
                    logger.info("Trade DB id=%d", trade_id)


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
