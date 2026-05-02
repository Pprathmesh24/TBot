"""
10-Year TIPS Real Yield daily data loader (FRED series DFII10).

Real yield = nominal yield - inflation expectations.
This is the single strongest macro driver for gold:
  real yield rises  → gold falls  (bonds pay more in real terms)
  real yield falls  → gold rises  (gold attractive vs negative-real bonds)

Values can be negative — they hit -1.1% in 2021, which drove gold's 2020-2021 rally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fredapi import Fred

from tbot.config import cfg

_SERIES   = "DFII10"   # 10-Year Treasury Inflation-Indexed Security, Constant Maturity
_OUT_PATH = Path("data/raw/macro/yields.parquet")


def fetch_yields(
    start: str = "2019-01-01",
    end:   str | None = None,
    out:   Path = _OUT_PATH,
) -> pd.DataFrame:
    """
    Download 10Y TIPS real yield from FRED and save to parquet.

    Requires FRED_API_KEY in .env (free at fred.stlouisfed.org).

    Returns:
        DataFrame with column [yield_pct], DatetimeIndex UTC
    """
    api_key = cfg.fred_api_key
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in .env — get a free key at fred.stlouisfed.org")

    fred   = Fred(api_key=api_key)
    series = fred.get_series(_SERIES, observation_start=start, observation_end=end)

    if series.empty:
        raise RuntimeError(f"FRED returned no data for {_SERIES}")

    df = series.rename("yield_pct").to_frame()
    # FRED already returns % (e.g. 1.5 = 1.50%)
    df.index      = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df.dropna().sort_index()

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def load_yields(path: Path = _OUT_PATH) -> pd.DataFrame:
    """Load yields parquet. Returns DataFrame with UTC DatetimeIndex."""
    return pd.read_parquet(path)
