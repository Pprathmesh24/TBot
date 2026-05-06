"""
Build the 30-candle OHLCV observation vector used as RL state.

Reads from the historical parquet file (primary) or the live candles table
(fallback for timestamps beyond the parquet range).

Output: flat list of 150 floats [lookback * 5] — prices normalised by the
entry-candle close, volume normalised by its own window mean.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PARQUET_PATH = Path("data/raw/XAU_USD_M5_2020_2025.parquet")
_LOOKBACK     = 30

_df_cache: pd.DataFrame | None = None


def _load_parquet() -> pd.DataFrame | None:
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    if not _PARQUET_PATH.exists():
        logger.warning("Parquet not found at %s — observation builder unavailable", _PARQUET_PATH)
        return None
    df = pd.read_parquet(_PARQUET_PATH)
    if not pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    _df_cache = df
    logger.info("Parquet loaded: %d candles (%s → %s)", len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return _df_cache


def build_observation(
    timestamp: datetime | str | pd.Timestamp,
    lookback:  int = _LOOKBACK,
    session=None,        # SQLAlchemy session — used if parquet misses the timestamp
) -> list[float] | None:
    """
    Return a flat list of `lookback * 5` normalised floats (OHLCV) ending at
    `timestamp`, or None if there aren't enough candles.
    """
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    window = _get_window_from_parquet(ts, lookback)

    if window is None and session is not None:
        window = _get_window_from_db(ts, lookback, session)

    if window is None:
        return None

    return _normalise(window)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_window_from_parquet(ts: pd.Timestamp, lookback: int) -> np.ndarray | None:
    df = _load_parquet()
    if df is None:
        return None

    mask   = df["timestamp"] <= ts
    subset = df.loc[mask, ["open", "high", "low", "close", "volume"]]
    if len(subset) < lookback:
        return None

    return subset.tail(lookback).values.astype(float)


def _get_window_from_db(ts: pd.Timestamp, lookback: int, session) -> np.ndarray | None:
    try:
        from sqlalchemy import select
        from tbot.db.models import Candle

        rows = (
            session.execute(
                select(Candle)
                .where(Candle.timestamp <= ts)
                .order_by(Candle.timestamp.desc())
                .limit(lookback)
            )
            .scalars()
            .all()
        )
        if len(rows) < lookback:
            return None

        rows = sorted(rows, key=lambda c: c.timestamp)
        return np.array([[c.open, c.high, c.low, c.close, c.volume] for c in rows], dtype=float)
    except Exception:
        logger.exception("DB candle window query failed")
        return None


def _normalise(ohlcv: np.ndarray) -> list[float]:
    base_price = ohlcv[-1, 3]          # last close
    if base_price == 0:
        base_price = 1.0
    ohlcv = ohlcv.copy()
    ohlcv[:, :4] /= base_price         # prices → fraction of entry close
    vol_mean = ohlcv[:, 4].mean()
    if vol_mean > 0:
        ohlcv[:, 4] /= vol_mean        # volume → relative to window mean
    return ohlcv.flatten().tolist()
