"""
Feature builder — snapshot market state at each signal candle.

build_features(df, idx, signal, fvgs, obs, sweeps) → dict of ~30 floats.
These dicts become the rows of the XGBoost training dataset in Phase 5.

Feature groups:
    volatility  — ATR, ATR percentile, Bollinger bandwidth
    momentum    — RSI, EMA slopes, MACD histogram, close-vs-EMA20 z-score
    smc_context — distances to nearest FVG/OB, recent sweep flag
    time        — hour, day-of-week, session flags
    signal_meta — direction, confidence, trigger type
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from tbot.core.smc.fvg import FVG
from tbot.core.smc.liquidity import LiquiditySweep
from tbot.core.smc.order_blocks import OrderBlock

# How many candles back to look for a "recent" sweep
_SWEEP_LOOKBACK = 10
# Rolling window for ATR percentile
_ATR_PCT_WINDOW = 500

# Trigger type encoding
_TRIGGER_ENCODE = {"FVG Retracement": 0, "OB Retracement": 1, "Liquidity Sweep": 2}


def build_features(
    df: pd.DataFrame,
    idx: int,
    signal: dict,
    fvgs: List[FVG],
    obs: List[OrderBlock],
    sweeps: List[LiquiditySweep],
) -> dict:
    """
    Snapshot all features at candle index `idx`.

    Args:
        df:      full candle DataFrame (load_candles output)
        idx:     integer position of the signal candle in df
        signal:  signal dict from AITradingAgentV2
        fvgs:    full FVG list for the dataset
        obs:     full OB list for the dataset
        sweeps:  full sweep list for the dataset

    Returns:
        Flat dict of feature_name → float.  All values are finite scalars.
    """
    if idx < 50:
        return {}   # not enough history for reliable indicators

    window = df.iloc[: idx + 1]   # candles up to and including signal candle
    close  = window["close"]
    high   = window["high"]
    low    = window["low"]
    ts     = df["timestamp"].iloc[idx]

    feats: dict = {}

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------
    atr      = _atr(window)
    atr_val  = float(atr.iloc[-1])
    price    = float(close.iloc[-1])

    feats["atr_abs"]   = atr_val
    feats["atr_pct"]   = atr_val / price if price > 0 else 0.0
    feats["atr_rank"]  = _rolling_rank(atr, _ATR_PCT_WINDOW)   # 0–1 percentile

    bb_upper, bb_mid, bb_lower = _bollinger(close)
    bb_width = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
    feats["bb_bandwidth"] = (bb_width / float(bb_mid.iloc[-1])) if bb_mid.iloc[-1] > 0 else 0.0
    feats["bb_position"]  = _clip(
        (price - float(bb_lower.iloc[-1])) / bb_width if bb_width > 0 else 0.5
    )

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    ema5  = _ema(close, 5)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)

    # Slopes: (current - 3 bars ago) / ATR
    feats["ema5_slope"]  = _slope(ema5,  3, atr_val)
    feats["ema20_slope"] = _slope(ema20, 5, atr_val)
    feats["ema50_slope"] = _slope(ema50, 10, atr_val)

    # EMA alignment: 1 = bull stack (5>20>50), -1 = bear, 0 = mixed
    feats["ema_alignment"] = _ema_alignment(ema5, ema20, ema50)

    # Close vs EMA20 z-score (rolling 50 bars)
    diff = close - ema20
    std50 = diff.rolling(50, min_periods=10).std().iloc[-1]
    feats["close_ema20_zscore"] = _clip(float(diff.iloc[-1]) / std50 if std50 > 0 else 0.0, 5)

    feats["rsi"] = _rsi(close, 14)

    macd_line, macd_signal = _macd(close)
    feats["macd_hist"]       = _clip(float((macd_line - macd_signal).iloc[-1]) / atr_val if atr_val > 0 else 0.0)
    feats["macd_cross"]      = _macd_cross(macd_line, macd_signal)

    # ------------------------------------------------------------------
    # SMC context
    # ------------------------------------------------------------------
    active_bull_fvgs = [f for f in fvgs if f.direction == "bull" and not f.is_filled and f.index < idx]
    active_bear_fvgs = [f for f in fvgs if f.direction == "bear" and not f.is_filled and f.index < idx]
    active_bull_obs  = [o for o in obs  if o.direction == "bull" and not o.is_mitigated and o.index < idx]
    active_bear_obs  = [o for o in obs  if o.direction == "bear" and not o.is_mitigated and o.index < idx]

    feats["dist_bull_fvg"] = _nearest_zone_dist(price, active_bull_fvgs, atr_val)
    feats["dist_bear_fvg"] = _nearest_zone_dist(price, active_bear_fvgs, atr_val)
    feats["dist_bull_ob"]  = _nearest_zone_dist(price, active_bull_obs,  atr_val)
    feats["dist_bear_ob"]  = _nearest_zone_dist(price, active_bear_obs,  atr_val)
    feats["n_active_bull_fvg"] = min(len(active_bull_fvgs), 10)
    feats["n_active_bear_fvg"] = min(len(active_bear_fvgs), 10)

    recent_sweeps = [s for s in sweeps if idx - _SWEEP_LOOKBACK <= s.index < idx]
    feats["recent_bull_sweep"] = int(any(s.direction == "bull" for s in recent_sweeps))
    feats["recent_bear_sweep"] = int(any(s.direction == "bear" for s in recent_sweeps))

    # ------------------------------------------------------------------
    # Candle structure
    # ------------------------------------------------------------------
    c = df.iloc[idx]
    body = abs(float(c["close"]) - float(c["open"]))
    rng  = float(c["high"]) - float(c["low"])
    feats["candle_body_ratio"] = (body / rng) if rng > 0 else 0.0
    feats["upper_wick_ratio"]  = (float(c["high"]) - max(float(c["open"]), float(c["close"]))) / rng if rng > 0 else 0.0
    feats["lower_wick_ratio"]  = (min(float(c["open"]), float(c["close"])) - float(c["low"])) / rng if rng > 0 else 0.0

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    feats["hour"]            = ts.hour
    feats["day_of_week"]     = ts.dayofweek    # 0=Mon, 4=Fri
    feats["london_session"]  = int(7  <= ts.hour < 16)
    feats["ny_session"]      = int(13 <= ts.hour < 21)
    feats["overlap_session"] = int(13 <= ts.hour < 16)  # London+NY overlap — highest volume

    # ------------------------------------------------------------------
    # Signal metadata
    # ------------------------------------------------------------------
    feats["signal_direction"] = 1.0 if signal.get("type") == "BUY" else -1.0
    feats["signal_confidence"] = float(signal.get("confidence", 0.0))
    feats["signal_trigger"]    = float(_TRIGGER_ENCODE.get(signal.get("reason", ""), -1))

    return {k: float(v) for k, v in feats.items()}


# ---------------------------------------------------------------------------
# Indicator helpers (all operate on pd.Series, return scalar or Series)
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid   = close.rolling(period, min_periods=1).mean()
    std   = close.rolling(period, min_periods=1).std().fillna(0)
    return mid + std_mult * std, mid, mid - std_mult * std


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs    = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100.0
    return float(100 - 100 / (1 + rs))


def _macd(close: pd.Series):
    line   = _ema(close, 12) - _ema(close, 26)
    signal = _ema(line, 9)
    return line, signal


def _slope(series: pd.Series, lookback: int, atr_val: float) -> float:
    if len(series) <= lookback or atr_val <= 0:
        return 0.0
    return _clip(float(series.iloc[-1] - series.iloc[-lookback - 1]) / atr_val)


def _ema_alignment(ema5, ema20, ema50) -> float:
    e5, e20, e50 = float(ema5.iloc[-1]), float(ema20.iloc[-1]), float(ema50.iloc[-1])
    if e5 > e20 > e50:
        return 1.0
    if e5 < e20 < e50:
        return -1.0
    return 0.0


def _macd_cross(line, signal) -> float:
    if len(line) < 2:
        return 0.0
    prev = float(line.iloc[-2]) - float(signal.iloc[-2])
    curr = float(line.iloc[-1]) - float(signal.iloc[-1])
    if prev <= 0 < curr:
        return 1.0   # bullish cross
    if prev >= 0 > curr:
        return -1.0  # bearish cross
    return 0.0


def _rolling_rank(series: pd.Series, window: int) -> float:
    tail = series.iloc[-window:].dropna()
    if len(tail) < 2:
        return 0.5
    val = float(series.iloc[-1])
    return float((tail <= val).mean())


def _nearest_zone_dist(price: float, zones, atr_val: float) -> float:
    """ATR-normalized distance from price to the nearest zone midpoint."""
    if not zones or atr_val <= 0:
        return 10.0   # far away sentinel
    mid_dists = [abs(price - (z.zone_high + z.zone_low) / 2) / atr_val for z in zones]
    return _clip(min(mid_dists), 10.0)


def _clip(val: float, limit: float = 10.0) -> float:
    return float(np.clip(val, -limit, limit))


# ---------------------------------------------------------------------------
# Batch-optimised API: pre-compute indicators once, then read per-index
# ---------------------------------------------------------------------------

class Precomputed:
    """Holds all indicator series computed once over the full DataFrame."""
    __slots__ = (
        "close", "high", "low", "open_",
        "atr", "atr_rank",
        "ema5", "ema20", "ema50",
        "bb_upper", "bb_mid", "bb_lower",
        "macd_line", "macd_signal",
        "rsi", "diff_ema20", "std50",
    )


def precompute_indicators(df: pd.DataFrame) -> Precomputed:
    """
    Compute every rolling indicator once on the full DataFrame.
    Pass the result to build_features_fast() for each signal instead of
    calling build_features() (which recomputes from scratch every time).
    """
    close = df["close"]

    atr            = _atr(df)
    # pandas rolling rank (pct=True) — O(n log n) once vs O(n²) per signal
    atr_rank       = atr.rolling(_ATR_PCT_WINDOW, min_periods=2).rank(pct=True).fillna(0.5)
    ema5           = _ema(close, 5)
    ema20          = _ema(close, 20)
    ema50          = _ema(close, 50)
    bb_upper, bb_mid, bb_lower = _bollinger(close)
    macd_line, macd_signal     = _macd(close)

    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss_s  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs      = gain / loss_s.replace(0.0, float("nan"))
    rsi_s   = (100 - 100 / (1 + rs)).fillna(100.0)

    diff  = close - ema20
    std50 = diff.rolling(50, min_periods=10).std()

    pc = Precomputed()
    pc.close       = close
    pc.high        = df["high"]
    pc.low         = df["low"]
    pc.open_       = df["open"]
    pc.atr         = atr
    pc.atr_rank    = atr_rank
    pc.ema5        = ema5
    pc.ema20       = ema20
    pc.ema50       = ema50
    pc.bb_upper    = bb_upper
    pc.bb_mid      = bb_mid
    pc.bb_lower    = bb_lower
    pc.macd_line   = macd_line
    pc.macd_signal = macd_signal
    pc.rsi         = rsi_s
    pc.diff_ema20  = diff
    pc.std50       = std50
    return pc


def build_features_fast(
    df: pd.DataFrame,
    pc: Precomputed,
    idx: int,
    signal: dict,
    fvgs: List[FVG],
    obs: List[OrderBlock],
    sweeps: List[LiquiditySweep],
    use_macro: bool = True,
) -> dict:
    """
    Same feature set as build_features() but reads from pre-computed series.
    Orders of magnitude faster for batch processing.
    """
    if idx < 50:
        return {}

    feats: dict = {}
    ts    = df["timestamp"].iloc[idx]

    atr_val = float(pc.atr.iloc[idx])
    price   = float(pc.close.iloc[idx])

    # --- Volatility ---
    feats["atr_abs"]  = atr_val
    feats["atr_pct"]  = atr_val / price if price > 0 else 0.0
    feats["atr_rank"] = float(pc.atr_rank.iloc[idx])

    bb_w = float(pc.bb_upper.iloc[idx] - pc.bb_lower.iloc[idx])
    bb_m = float(pc.bb_mid.iloc[idx])
    feats["bb_bandwidth"] = (bb_w / bb_m) if bb_m > 0 else 0.0
    feats["bb_position"]  = _clip((price - float(pc.bb_lower.iloc[idx])) / bb_w if bb_w > 0 else 0.5)

    # --- Momentum ---
    def slope_at(series, lookback):
        if idx < lookback + 1 or atr_val <= 0:
            return 0.0
        return _clip(float(series.iloc[idx] - series.iloc[idx - lookback]) / atr_val)

    feats["ema5_slope"]  = slope_at(pc.ema5,  3)
    feats["ema20_slope"] = slope_at(pc.ema20, 5)
    feats["ema50_slope"] = slope_at(pc.ema50, 10)

    e5, e20, e50 = float(pc.ema5.iloc[idx]), float(pc.ema20.iloc[idx]), float(pc.ema50.iloc[idx])
    if e5 > e20 > e50:
        feats["ema_alignment"] = 1.0
    elif e5 < e20 < e50:
        feats["ema_alignment"] = -1.0
    else:
        feats["ema_alignment"] = 0.0

    std50_val = pc.std50.iloc[idx]
    feats["close_ema20_zscore"] = _clip(
        float(pc.diff_ema20.iloc[idx]) / float(std50_val) if std50_val and std50_val > 0 else 0.0, 5
    )

    feats["rsi"] = float(pc.rsi.iloc[idx])

    ml = float(pc.macd_line.iloc[idx])
    ms = float(pc.macd_signal.iloc[idx])
    feats["macd_hist"] = _clip((ml - ms) / atr_val if atr_val > 0 else 0.0)

    if idx >= 1:
        prev_diff = float(pc.macd_line.iloc[idx - 1]) - float(pc.macd_signal.iloc[idx - 1])
        curr_diff = ml - ms
        if prev_diff <= 0 < curr_diff:
            feats["macd_cross"] = 1.0
        elif prev_diff >= 0 > curr_diff:
            feats["macd_cross"] = -1.0
        else:
            feats["macd_cross"] = 0.0
    else:
        feats["macd_cross"] = 0.0

    # --- SMC context ---
    active_bull_fvgs = [f for f in fvgs if f.direction == "bull" and not f.is_filled and f.index < idx]
    active_bear_fvgs = [f for f in fvgs if f.direction == "bear" and not f.is_filled and f.index < idx]
    active_bull_obs  = [o for o in obs  if o.direction == "bull" and not o.is_mitigated and o.index < idx]
    active_bear_obs  = [o for o in obs  if o.direction == "bear" and not o.is_mitigated and o.index < idx]

    feats["dist_bull_fvg"]    = _nearest_zone_dist(price, active_bull_fvgs, atr_val)
    feats["dist_bear_fvg"]    = _nearest_zone_dist(price, active_bear_fvgs, atr_val)
    feats["dist_bull_ob"]     = _nearest_zone_dist(price, active_bull_obs,  atr_val)
    feats["dist_bear_ob"]     = _nearest_zone_dist(price, active_bear_obs,  atr_val)
    feats["n_active_bull_fvg"] = min(len(active_bull_fvgs), 10)
    feats["n_active_bear_fvg"] = min(len(active_bear_fvgs), 10)

    recent_sweeps = [s for s in sweeps if idx - _SWEEP_LOOKBACK <= s.index < idx]
    feats["recent_bull_sweep"] = int(any(s.direction == "bull" for s in recent_sweeps))
    feats["recent_bear_sweep"] = int(any(s.direction == "bear" for s in recent_sweeps))

    # --- Candle structure ---
    c_open  = float(pc.open_.iloc[idx])
    c_close = float(pc.close.iloc[idx])
    c_high  = float(pc.high.iloc[idx])
    c_low   = float(pc.low.iloc[idx])
    body = abs(c_close - c_open)
    rng  = c_high - c_low
    feats["candle_body_ratio"] = (body / rng) if rng > 0 else 0.0
    feats["upper_wick_ratio"]  = (c_high - max(c_open, c_close)) / rng if rng > 0 else 0.0
    feats["lower_wick_ratio"]  = (min(c_open, c_close) - c_low) / rng if rng > 0 else 0.0

    # --- Time ---
    feats["hour"]            = ts.hour
    feats["day_of_week"]     = ts.dayofweek
    feats["london_session"]  = int(7  <= ts.hour < 16)
    feats["ny_session"]      = int(13 <= ts.hour < 21)
    feats["overlap_session"] = int(13 <= ts.hour < 16)

    # --- Signal metadata ---
    feats["signal_direction"]  = 1.0 if signal.get("type") == "BUY" else -1.0
    feats["signal_confidence"] = float(signal.get("confidence", 0.0))
    feats["signal_trigger"]    = float(_TRIGGER_ENCODE.get(signal.get("reason", ""), -1))

    # --- Macro (DXY, VIX, 10Y TIPS yield, COT) ---
    if use_macro:
        try:
            from tbot.features.macro_features import get_macro_features
            feats.update(get_macro_features(ts))
        except Exception:
            pass  # macro parquets not yet fetched — skip gracefully

    return {k: float(v) for k, v in feats.items()}
