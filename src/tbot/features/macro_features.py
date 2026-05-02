"""
Macro feature lookup for ML model.

Two APIs:
  get_macro_features(ts)          — single timestamp (used in live/backtest runner)
  get_macro_features_batch(tss)   — vectorized, uses pd.merge_asof for 100x speedup
                                    (used in build_training_dataset for 82k signals)

At each signal candle (UTC timestamp), returns the most-recent available
value for each macro series using as-of / forward-fill logic — zero look-ahead.

Series frequencies:
  DXY, VIX, yields — daily
  COT              — weekly (Tuesday publish)
"""

from __future__ import annotations

import pandas as pd

from tbot.data.macro.dxy    import load_dxy
from tbot.data.macro.vix    import load_vix
from tbot.data.macro.yields import load_yields
from tbot.data.macro.cot    import load_cot

_dxy:    pd.DataFrame | None = None
_vix:    pd.DataFrame | None = None
_yields: pd.DataFrame | None = None
_cot:    pd.DataFrame | None = None


def _load_all() -> None:
    global _dxy, _vix, _yields, _cot
    if _dxy is None:
        _dxy    = load_dxy()
        _vix    = load_vix()
        _yields = load_yields()
        _cot    = load_cot()


# ---------------------------------------------------------------------------
# Batch API (fast) — use this when processing many signals
# ---------------------------------------------------------------------------

def get_macro_features_batch(timestamps: pd.Series) -> pd.DataFrame:
    """
    Vectorized macro feature lookup for a Series of UTC timestamps.

    Returns a DataFrame with 10 macro columns, one row per timestamp.
    Uses pd.merge_asof — O(n log n) single pass vs O(n*m) per-signal loop.

    Args:
        timestamps: pd.Series of UTC Timestamps (signal candle times)

    Returns:
        DataFrame indexed 0..n-1 with the 10 macro feature columns.
    """
    _load_all()

    ts_df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True)}).sort_values("timestamp")

    # --- DXY ---
    dxy = _dxy.reset_index()[["timestamp", "close"]].rename(columns={"close": "dxy_close"}).sort_values("timestamp")
    dxy["dxy_prev"] = dxy["dxy_close"].shift(1)
    merged = pd.merge_asof(ts_df, dxy, on="timestamp")
    merged["dxy_1d_pct"] = (merged["dxy_close"] - merged["dxy_prev"]) / merged["dxy_prev"].replace(0, float("nan"))
    merged["dxy_close"]  = merged["dxy_close"].fillna(0.0)
    merged["dxy_1d_pct"] = merged["dxy_1d_pct"].fillna(0.0)

    # --- VIX ---
    vix = _vix.reset_index()[["timestamp", "close"]].rename(columns={"close": "vix_close"}).sort_values("timestamp")
    merged = pd.merge_asof(merged, vix, on="timestamp")
    merged["vix_close"] = merged["vix_close"].fillna(0.0)

    # VIX percentile: fraction of trailing 252 VIX closes below current — computed per-row
    vix_vals = _vix["close"].dropna()
    def _vix_pct(row_vix, row_ts):
        window = vix_vals[vix_vals.index <= row_ts].tail(252)
        return float((window < row_vix).sum() / len(window)) if len(window) else 0.5
    merged["vix_pct"] = [_vix_pct(v, t) for v, t in zip(merged["vix_close"], merged["timestamp"])]

    # --- 10Y TIPS yields ---
    yld = _yields.reset_index()[["timestamp", "yield_pct"]].sort_values("timestamp")
    yld["yield_prev"] = yld["yield_pct"].shift(1)
    merged = pd.merge_asof(merged, yld, on="timestamp")
    merged["yield_1d_chg"] = (merged["yield_pct"] - merged["yield_prev"]).fillna(0.0)
    merged["yield_pct"]    = merged["yield_pct"].fillna(0.0)

    # --- COT ---
    cot = _cot.reset_index()[["timestamp", "mm_net", "mm_net_pct"]].sort_values("timestamp")
    merged = pd.merge_asof(merged, cot, on="timestamp")
    merged["cot_mm_net"]     = merged["mm_net"].fillna(0.0)
    merged["cot_mm_net_pct"] = merged["mm_net_pct"].fillna(0.0)

    # --- Composite regime flags ---
    merged["macro_risk_on"]  = ((merged["vix_close"] < 20) & (merged["dxy_1d_pct"] < 0)).astype(float)
    merged["macro_risk_off"] = ((merged["vix_close"] > 30) | (merged["yield_1d_chg"] > 0.05)).astype(float)

    cols = [
        "dxy_close", "dxy_1d_pct",
        "vix_close",  "vix_pct",
        "yield_pct",  "yield_1d_chg",
        "cot_mm_net", "cot_mm_net_pct",
        "macro_risk_on", "macro_risk_off",
    ]
    # Restore original order
    result = merged.set_index(merged.index)[cols]
    result.index = ts_df.index  # align back to caller's index
    return result.reindex(pd.RangeIndex(len(timestamps))).fillna(0.0)


# ---------------------------------------------------------------------------
# Single-timestamp API (used in live runner, test scripts)
# ---------------------------------------------------------------------------

def _asof(series: pd.Series, ts: pd.Timestamp) -> float | None:
    subset = series[series.index <= ts].dropna()
    return float(subset.iloc[-1]) if len(subset) else None


def get_macro_features(ts: pd.Timestamp) -> dict[str, float]:
    """
    Return macro feature dict for a signal at timestamp ts (UTC).

    For batch use (training), call get_macro_features_batch() instead.
    """
    _load_all()
    ts = pd.Timestamp(ts, tz="UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    dxy_close = _asof(_dxy["close"], ts) or 0.0
    dxy_prev  = _asof(_dxy["close"], ts - pd.Timedelta(days=2)) or dxy_close
    dxy_1d_pct = (dxy_close - dxy_prev) / dxy_prev if dxy_prev else 0.0

    vix_close = _asof(_vix["close"], ts) or 0.0
    vix_series = _vix["close"]
    window    = vix_series[vix_series.index <= ts].tail(252).dropna()
    vix_pct   = float((window < vix_close).sum() / len(window)) if len(window) else 0.5

    yield_now  = _asof(_yields["yield_pct"], ts) or 0.0
    yield_prev = _asof(_yields["yield_pct"], ts - pd.Timedelta(days=2)) or yield_now
    yield_1d_chg = yield_now - yield_prev

    cot_mm_net     = _asof(_cot["mm_net"],     ts) or 0.0
    cot_mm_net_pct = _asof(_cot["mm_net_pct"], ts) or 0.0

    macro_risk_on  = int(vix_close < 20 and dxy_1d_pct < 0)
    macro_risk_off = int(vix_close > 30 or yield_1d_chg > 0.05)

    return {
        "dxy_close":       dxy_close,
        "dxy_1d_pct":      dxy_1d_pct,
        "vix_close":       vix_close,
        "vix_pct":         vix_pct,
        "yield_pct":       yield_now,
        "yield_1d_chg":    yield_1d_chg,
        "cot_mm_net":      cot_mm_net,
        "cot_mm_net_pct":  cot_mm_net_pct,
        "macro_risk_on":   float(macro_risk_on),
        "macro_risk_off":  float(macro_risk_off),
    }
