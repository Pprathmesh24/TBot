"""
Canonical loader for TBot OHLCV data.

Every phase that needs candle data calls load_candles() — never pd.read_parquet() directly.
This ensures schema, types, and timestamp integrity are always validated in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tbot.config import cfg

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
DEFAULT_PATH = Path("data/raw/XAU_USD_M5_2020_2025.parquet")


def load_candles(
    path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV candles from parquet, validate schema, and return a clean DataFrame.

    Args:
        path:  parquet file path; defaults to data/raw/XAU_USD_M5_2020_2025.parquet
        start: optional ISO date string to slice from, e.g. "2023-01-01"
        end:   optional ISO date string to slice to,   e.g. "2024-01-01"

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume],
        timestamp as UTC-aware datetime, sorted ascending, no duplicates.

    Raises:
        FileNotFoundError: if the parquet file doesn't exist
        ValueError:        if required columns are missing or timestamps are invalid
    """
    path = Path(path) if path else DEFAULT_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run: python -m tbot.data.fetch_oanda_history"
        )

    df = pd.read_parquet(path)

    # --- schema validation ---
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Parquet file is missing columns: {missing}")

    # --- type coercion ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype("float64")
    df["volume"] = df["volume"].astype("int64")

    # --- integrity checks ---
    dupes = df["timestamp"].duplicated().sum()
    if dupes:
        df = df.drop_duplicates(subset="timestamp")

    df = df.sort_values("timestamp").reset_index(drop=True)

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonically increasing after sort — data is corrupt.")

    # --- optional date slice ---
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]

    return df.reset_index(drop=True)


def candles_to_dict_list(df: pd.DataFrame) -> list[dict]:
    """
    Convert a loader DataFrame to the list-of-dicts format MarketStructureAnalyzer expects.
    Bridges the new loader with the existing core/market_structure.py.
    """
    records = []
    for row in df.itertuples(index=True):
        records.append({
            "index":     row.Index,
            "timestamp": row.timestamp,
            "open":      row.open,
            "high":      row.high,
            "low":       row.low,
            "close":     row.close,
            "volume":    row.volume,
            "is_green":  row.close > row.open,
            "is_red":    row.close < row.open,
        })
    return records
