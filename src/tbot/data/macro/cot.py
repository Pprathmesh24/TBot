"""
CFTC Commitment of Traders (COT) — Gold futures data loader.

Downloads the public Disaggregated COT report (futures only) from cftc.gov.
Extracts Managed Money net position for Gold (COMEX contract 088691).

Published weekly on Fridays, reflecting positions as of the prior Tuesday.
Managed Money net long = hedge funds bullish on gold.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

_GOLD_CONTRACT = "088691"   # COMEX Gold futures CFTC contract code
_OUT_PATH      = Path("data/raw/macro/cot_gold.parquet")

# CFTC publishes one zip per year; columns we need are stable across years
_COT_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
_COLS = {
    "Market_and_Exchange_Names":   "market",
    "Report_Date_as_YYYY-MM-DD":   "date_raw",   # ISO format, easier to parse than YYMMDD
    "CFTC_Contract_Market_Code":   "contract_code",
    "M_Money_Positions_Long_All":  "mm_long",
    "M_Money_Positions_Short_All": "mm_short",
}


def fetch_cot(
    start_year: int = 2019,
    end_year:   int | None = None,
    out:        Path = _OUT_PATH,
) -> pd.DataFrame:
    """
    Download CFTC disaggregated COT for Gold and save to parquet.

    Args:
        start_year: first year to fetch (inclusive)
        end_year:   last year to fetch (inclusive, defaults to current year)
        out:        output parquet path

    Returns:
        DataFrame with columns [mm_long, mm_short, mm_net, mm_net_pct],
        DatetimeIndex UTC (weekly, Tuesdays)
    """
    import datetime
    if end_year is None:
        end_year = datetime.date.today().year

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        url = _COT_URL.format(year=year)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  Warning: could not fetch COT for {year}: {exc}")
            continue

        try:
            df_year = pd.read_csv(
                io.BytesIO(resp.content),
                compression="zip",
                usecols=list(_COLS.keys()),
                low_memory=False,
            )
        except Exception as exc:
            print(f"  Warning: could not parse COT for {year}: {exc}")
            continue

        df_year = df_year.rename(columns=_COLS)
        df_year = df_year[df_year["contract_code"].str.strip() == _GOLD_CONTRACT]
        frames.append(df_year)

    if not frames:
        raise RuntimeError("No COT data fetched — check CFTC URL or network connection")

    df = pd.concat(frames, ignore_index=True)

    # Parse date: YYYY-MM-DD → datetime
    df["timestamp"] = pd.to_datetime(df["date_raw"].astype(str), format="%Y-%m-%d", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    df["mm_long"]  = pd.to_numeric(df["mm_long"],  errors="coerce")
    df["mm_short"] = pd.to_numeric(df["mm_short"], errors="coerce")
    df["mm_net"]   = df["mm_long"] - df["mm_short"]
    total = df["mm_long"] + df["mm_short"]
    df["mm_net_pct"] = df["mm_net"] / total.replace(0, float("nan"))  # -1 to +1

    result = df[["mm_long", "mm_short", "mm_net", "mm_net_pct"]].dropna()

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out)
    return result


def load_cot(path: Path = _OUT_PATH) -> pd.DataFrame:
    """Load COT parquet. Returns weekly DataFrame with UTC DatetimeIndex."""
    return pd.read_parquet(path)
