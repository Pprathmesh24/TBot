"""
VIX (CBOE Volatility Index) daily data loader.

Fetches ^VIX from Yahoo Finance and saves to data/raw/macro/vix.parquet.
VIX measures expected 30-day S&P 500 volatility. Spikes in VIX drive
safe-haven flows into gold — it's a risk-off/risk-on proxy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

_TICKER   = "^VIX"
_OUT_PATH = Path("data/raw/macro/vix.parquet")


def fetch_vix(
    start: str = "2019-01-01",
    end:   str | None = None,
    out:   Path = _OUT_PATH,
) -> pd.DataFrame:
    """
    Download VIX daily close from Yahoo Finance and save to parquet.

    Returns:
        DataFrame with columns [close], DatetimeIndex UTC
    """
    raw = yf.download(_TICKER, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {_TICKER}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    df.columns = ["close"]
    df.index   = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df.sort_index()

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def load_vix(path: Path = _OUT_PATH) -> pd.DataFrame:
    """Load VIX parquet. Returns DataFrame with UTC DatetimeIndex."""
    return pd.read_parquet(path)
