"""
Phase 4 Chunk 3 — Build the ML training dataset.

Runs the V2 SMC agent on the full 5-year candle history, labels every
signal with triple-barrier labeling, builds 32 features per signal,
and writes the result to data/features/signals_labeled.parquet.

Usage:
    uv run python scripts/build_training_dataset.py
    uv run python scripts/build_training_dataset.py --timeout 72  # wider barrier

Output:
    data/features/signals_labeled.parquet
    data/features/dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from tbot.core.smc.structure_v2 import EnrichedMarketStructureAnalyzer
from tbot.data.loader import load_candles
from tbot.features.builder import build_features_fast, precompute_indicators
from tbot.features.labeler import label_all
from tbot.features.macro_features import get_macro_features_batch


def build_dataset(timeout_candles: int = 48) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Load candles
    # ------------------------------------------------------------------ #
    parquet_path = Path("data/raw/XAU_USD_M5_2020_2025.parquet")
    print(f"Loading candles from {parquet_path} …")
    df = load_candles(parquet_path)
    print(f"  {len(df):,} candles  ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")

    # ------------------------------------------------------------------ #
    # 2. Run SMC analyzer — detects FVGs, OBs, sweeps, generates signals
    # ------------------------------------------------------------------ #
    print("Running EnrichedMarketStructureAnalyzer …  (this takes ~20 min on 448k candles — not stuck)")
    t0 = time.time()
    analyzer = EnrichedMarketStructureAnalyzer(df)
    raw_signals = analyzer.get_entry_signals()
    elapsed = time.time() - t0
    print(f"  {len(raw_signals):,} raw signals  ({elapsed:.1f}s)")

    fvgs   = analyzer._fvgs
    obs    = analyzer._obs
    sweeps = analyzer._sweeps
    print(f"  FVGs={len(fvgs)}  OBs={len(obs)}  Sweeps={len(sweeps)}")

    # ------------------------------------------------------------------ #
    # 3. Triple-barrier labeling
    # ------------------------------------------------------------------ #
    print(f"Labeling with triple-barrier (timeout={timeout_candles} candles = {timeout_candles*5/60:.0f}h) …")
    labeled = label_all(df, raw_signals, timeout_candles=timeout_candles)
    print(f"  {len(labeled):,} signals matched to candle index")

    # ------------------------------------------------------------------ #
    # 4. Build features for each labeled signal
    # ------------------------------------------------------------------ #
    print("Pre-computing indicators (one pass) …")
    pc = precompute_indicators(df)

    print("Building features …")
    rows: list[dict] = []
    skipped_no_history = 0

    for sig in labeled:
        idx   = sig["signal_idx"]
        feats = build_features_fast(df, pc, idx, sig, fvgs, obs, sweeps, use_macro=False)

        if not feats:        # idx < 50 — not enough history
            skipped_no_history += 1
            continue

        row = {**feats, "label": sig["label"], "timestamp": df["timestamp"].iloc[idx]}
        rows.append(row)

    print(f"  {len(rows):,} feature rows built  (skipped {skipped_no_history} for insufficient history)")

    # ------------------------------------------------------------------ #
    # 4b. Join macro features (vectorized — one pass via merge_asof)
    # ------------------------------------------------------------------ #
    print("Joining macro features (vectorized) …")
    dataset_tmp  = pd.DataFrame(rows)
    macro_df     = get_macro_features_batch(dataset_tmp["timestamp"])
    macro_df.index = dataset_tmp.index
    dataset_tmp  = pd.concat([dataset_tmp, macro_df], axis=1)
    rows = dataset_tmp.to_dict("records")
    print(f"  Macro features joined ({len(macro_df.columns)} cols)")

    # ------------------------------------------------------------------ #
    # 5. Assemble DataFrame and save
    # ------------------------------------------------------------------ #
    dataset = pd.DataFrame(rows)

    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "signals_labeled.parquet"
    dataset.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}  ({len(dataset):,} rows × {len(dataset.columns)} cols)")

    # ------------------------------------------------------------------ #
    # 6. Class balance + quick stats
    # ------------------------------------------------------------------ #
    counts = dataset["label"].value_counts()
    total  = len(dataset)

    print("\n=== Label distribution ===")
    for label, n in counts.items():
        print(f"  {label:8s}  {n:6,}  ({100*n/total:.1f}%)")

    win_rate = counts.get("WIN", 0) / total

    stats = {
        "total_rows":         total,
        "n_features":         len(dataset.columns) - 2,   # excl. label + timestamp
        "label_counts":       counts.to_dict(),
        "win_rate":           round(win_rate, 4),
        "timeout_candles":    timeout_candles,
        "date_range":         [
            str(dataset["timestamp"].min().date()),
            str(dataset["timestamp"].max().date()),
        ],
    }

    stats_path = out_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nStats saved → {stats_path}")

    # Sanity check: class balance (30–50% WIN is healthy)
    if not 0.20 <= win_rate <= 0.60:
        print(f"\n  WARNING: win_rate={win_rate:.1%} outside 20–60% range — labeler may be misconfigured")
    else:
        print(f"\n  OK: win_rate={win_rate:.1%} is within healthy range (20–60%)")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build triple-barrier labeled training dataset")
    parser.add_argument("--timeout", type=int, default=48,
                        help="Candles to look forward for barrier (default 48 = 4h on M5)")
    args = parser.parse_args()

    build_dataset(timeout_candles=args.timeout)


if __name__ == "__main__":
    main()
