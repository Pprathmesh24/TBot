"""
OANDA pricing stream → M5 candle aggregator.

Receives tick-level bid/ask prices from OANDA's streaming endpoint and
aggregates them into completed M5 candles that match the format returned
by tbot.data.loader.load_candles().

Usage:
    def on_candle(candle: dict) -> None:
        # called once per completed M5 bar
        agent.process_new_candle(candle)

    stream = PriceStream(client, instrument="XAU_USD")
    stream.start(on_candle)   # blocks until stream.stop() is called
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

import oandapyV20.endpoints.pricing as pricing

from tbot.broker.oanda_client import OandaClient

logger = logging.getLogger(__name__)

_GRANULARITY_MINUTES = 5


# ---------------------------------------------------------------------------
# Candle builder
# ---------------------------------------------------------------------------

class CandleBuilder:
    """
    Accumulates ticks into a single M5 OHLCV candle.

    Bar boundaries are aligned to UTC clock (00:00, 00:05, 00:10 …).
    Volume is the tick count (OANDA streaming doesn't provide real volume).
    """

    def __init__(self, bar_start: datetime, open_price: float) -> None:
        self.bar_start = bar_start
        self.open  = open_price
        self.high  = open_price
        self.low   = open_price
        self.close = open_price
        self.volume = 1

    def update(self, price: float) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += 1

    def to_dict(self) -> dict:
        return {
            "timestamp": self.bar_start,
            "open":      self.open,
            "high":      self.high,
            "low":       self.low,
            "close":     self.close,
            "volume":    self.volume,
            "is_green":  self.close >= self.open,
            "is_red":    self.close < self.open,
        }


def _bar_start(dt: datetime, granularity_minutes: int = _GRANULARITY_MINUTES) -> datetime:
    """Truncate a UTC datetime to the start of its N-minute bar."""
    minute = (dt.minute // granularity_minutes) * granularity_minutes
    return dt.replace(minute=minute, second=0, microsecond=0)


def _mid(bid: str, ask: str) -> float:
    return (float(bid) + float(ask)) / 2.0


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

class PriceStream:
    """
    Streams OANDA pricing ticks for one instrument and calls on_candle()
    each time a complete M5 bar closes.

    Args:
        client:     OandaClient instance (already authenticated)
        instrument: e.g. "XAU_USD"
        granularity_minutes: bar size in minutes (default 5)
    """

    def __init__(
        self,
        client:              OandaClient,
        instrument:          str = "XAU_USD",
        granularity_minutes: int = _GRANULARITY_MINUTES,
    ) -> None:
        self._client      = client
        self._instrument  = instrument
        self._granularity = granularity_minutes
        self._stop_event  = threading.Event()
        self._builder:    CandleBuilder | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self, on_candle: Callable[[dict], None]) -> None:
        """
        Connect to OANDA pricing stream and block until stop() is called.
        Automatically reconnects on disconnect with exponential backoff.

        on_candle is called in this thread once per completed bar.
        Wrap in a background thread if you need non-blocking behaviour.

        Args:
            on_candle: callback receiving a candle dict (same schema as
                       load_candles() rows) when each M5 bar closes.
        """
        self._stop_event.clear()
        logger.info("Starting price stream for %s …", self._instrument)

        retry_delay = 5  # seconds, doubles on each failure up to 60s
        attempt     = 0

        while not self._stop_event.is_set():
            req = pricing.PricingStream(
                accountID=self._client.account_id,
                params={"instruments": self._instrument},
            )

            received_any = False  # if we got at least one msg, reset backoff afterwards

            try:
                for msg in self._client._client.request(req):
                    if self._stop_event.is_set():
                        break

                    # Reset retry delay once we know the connection works.
                    if not received_any:
                        received_any = True
                        if attempt > 0:
                            logger.info("Price stream reconnected (attempt %d)", attempt)
                            try:
                                from tbot.monitoring.alerts import alert_stream_reconnect
                                alert_stream_reconnect(attempt)
                            except Exception:
                                logger.debug("alert_stream_reconnect failed", exc_info=True)
                        retry_delay = 5
                        attempt = 0

                    msg_type = msg.get("type")

                    if msg_type == "HEARTBEAT":
                        continue

                    if msg_type != "PRICE":
                        continue

                    bid = msg.get("bids", [{}])[0].get("price", "0")
                    ask = msg.get("asks", [{}])[0].get("price", "0")
                    mid = _mid(bid, ask)

                    raw_time = msg.get("time", "")
                    try:
                        tick_time = datetime.fromisoformat(
                            raw_time.replace("Z", "+00:00")
                        ).replace(tzinfo=timezone.utc)
                    except (ValueError, AttributeError):
                        tick_time = datetime.now(timezone.utc)

                    bar_ts = _bar_start(tick_time, self._granularity)

                    if self._builder is None:
                        self._builder = CandleBuilder(bar_ts, mid)
                    elif bar_ts > self._builder.bar_start:
                        completed = self._builder.to_dict()
                        logger.debug("Candle closed: %s  C=%.3f", bar_ts, completed["close"])
                        # No try/except here: LiveRunner._on_candle wraps its
                        # own logic in try/except already, so any exception
                        # surfacing here is a bug in the framing or in
                        # LiveRunner's OWN exception handler — letting it
                        # propagate triggers the outer stream-loop's reconnect
                        # path, which logs it loudly. Double-catching would
                        # mask genuine bugs.
                        on_candle(completed)
                        self._builder = CandleBuilder(bar_ts, mid)
                    else:
                        self._builder.update(mid)

                # Clean exit via stop()
                if self._stop_event.is_set():
                    break

                # Stream ended without stop() — treat as disconnect
                logger.warning("Stream ended unexpectedly — reconnecting in %ds …", retry_delay)

            except Exception:
                if self._stop_event.is_set():
                    break
                logger.exception("Price stream disconnected — reconnecting in %ds …", retry_delay)

            # Wait before retrying, but stay responsive to stop()
            attempt += 1
            if self._stop_event.wait(timeout=retry_delay):
                break
            retry_delay = min(retry_delay * 2, 60)

        logger.info("Price stream stopped.")

    def stop(self) -> None:
        """Signal the stream loop to exit after the next message."""
        self._stop_event.set()

    def start_background(self, on_candle: Callable[[dict], None]) -> threading.Thread:
        """
        Start the stream in a daemon thread and return it.
        Call stop() to cleanly shut it down.
        """
        t = threading.Thread(target=self.start, args=(on_candle,), daemon=True)
        t.start()
        return t
