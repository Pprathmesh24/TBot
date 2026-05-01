"""
Fetch XAU_USD M5 historical candles from OANDA and save as parquet.

Usage:
    python -m tbot.data.fetch_oanda_history

Reads credentials from .env (OANDA__API_TOKEN, OANDA__ACCOUNT_ID).
Saves to data/raw/XAU_USD_M5_2020_2025.parquet.

OANDA limits each request to 5 000 candles, so we paginate by advancing
the `from` timestamp after every batch (~74 requests for 5 years of M5).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

from tbot.config import cfg

load_dotenv()

INSTRUMENT = "XAU_USD"
GRANULARITY = "M5"
BATCH_SIZE = 5000
START_DATE = "2020-01-01T00:00:00Z"
OUT_PATH = Path("data/raw/XAU_USD_M5_2020_2025.parquet")


def _candles_to_rows(candles: list[dict]) -> list[dict]:
    rows = []
    for c in candles:
        if c.get("complete") is False:
            continue  # skip the in-progress candle at the end of each batch
        mid = c["mid"]
        rows.append({
            "timestamp": pd.Timestamp(c["time"]),
            "open":   float(mid["o"]),
            "high":   float(mid["h"]),
            "low":    float(mid["l"]),
            "close":  float(mid["c"]),
            "volume": int(c["volume"]),
        })
    return rows


def fetch(token: str, from_dt: str = START_DATE) -> pd.DataFrame:
    client = oandapyV20.API(access_token=token, environment=cfg.oanda.environment)

    all_rows: list[dict] = []
    current_from = from_dt
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    batch_num = 0

    print(f"Fetching {INSTRUMENT} {GRANULARITY} from {current_from} …")

    while current_from < now_str:
        params = {
            "granularity": GRANULARITY,
            "count": BATCH_SIZE,
            "from": current_from,
            "price": "M",  # mid prices
        }
        req = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
        client.request(req)
        candles = req.response.get("candles", [])

        if not candles:
            break

        rows = _candles_to_rows(candles)
        all_rows.extend(rows)
        batch_num += 1

        # Advance past the last candle we received
        last_ts = candles[-1]["time"]
        current_from = last_ts  # OANDA `from` is inclusive, so next batch starts here;
                                # duplicate is dropped by dedup below

        print(f"  batch {batch_num:>3}: {len(rows):>5} candles  "
              f"last={last_ts[:16]}  total={len(all_rows):>7}")

        if len(candles) < BATCH_SIZE:
            break  # reached the present

        time.sleep(0.2)  # be polite to the API

    if not all_rows:
        raise RuntimeError("No candles returned — check your API token and account.")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} candles → {OUT_PATH}")
    print(f"Date range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    return df


if __name__ == "__main__":
    token = cfg.oanda.api_token
    if not token:
        raise SystemExit("OANDA__API_TOKEN not set — copy .env.example to .env and fill it in.")
    fetch(token)
