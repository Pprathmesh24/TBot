"""
DXY (US Dollar Index) daily data loader.

Fetches DX-Y.NYB from Yahoo Finance and saves to data/raw/macro/dxy.parquet.
DXY has a strong negative correlation with XAU/USD (~-0.7 historically),
so its direction is a first-order macro feature for gold signals.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

_TICKER   = "DX-Y.NYB"          # Yahoo Finance symbol for DXY
_OUT_PATH = Path("data/raw/macro/dxy.parquet")


def fetch_dxy(
    start: str = "2019-01-01",
    end:   str | None = None,
    out:   Path = _OUT_PATH,
) -> pd.DataFrame:
    """
    Download DXY daily OHLCV from Yahoo Finance and save to parquet.

    Args:
        start: ISO date string, start of history
        end:   ISO date string, defaults to today
        out:   output parquet path

    Returns:
        DataFrame with columns [open, high, low, close, volume], DatetimeIndex UTC
    """
    raw = yf.download(_TICKER, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {_TICKER}")

    # Flatten MultiIndex columns if present (yfinance >= 0.2.40 sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index   = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df.sort_index()

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def load_dxy(path: Path = _OUT_PATH) -> pd.DataFrame:
    """Load DXY parquet. Returns DataFrame with UTC DatetimeIndex."""
    return pd.read_parquet(path)
