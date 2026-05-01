"""
Triple-barrier labeler (López de Prado, "Advances in Financial Machine Learning").

For each signal, scan the next `timeout_candles` candles and assign:
    WIN     — take-profit hit before stop-loss and before timeout
    LOSS    — stop-loss hit before take-profit and before timeout
    NEUTRAL — neither barrier hit within timeout (trade skipped by model)

Rules when both barriers are hit on the same candle:
    - Conservatively assign LOSS (worst-case assumption).

Usage:
    from tbot.features.labeler import label_signal, label_all

    label = label_signal(df, idx, signal)          # one signal
    labeled = label_all(df, signals, timeout=48)   # list of signals
"""

from __future__ import annotations

from typing import List

import pandas as pd


def label_signal(
    df: pd.DataFrame,
    signal_idx: int,
    signal: dict,
    timeout_candles: int = 48,
) -> str:
    """
    Label a single signal as WIN, LOSS, or NEUTRAL.

    Args:
        df:              full candle DataFrame (load_candles output)
        signal_idx:      integer row index of the signal candle in df
        signal:          signal dict with keys: type, stop_loss, take_profit
        timeout_candles: number of candles to look forward (default 48 = 4h on M5)

    Returns:
        "WIN" | "LOSS" | "NEUTRAL"
    """
    side = signal.get("type", "BUY")
    sl   = float(signal.get("stop_loss",   0.0))
    tp   = float(signal.get("take_profit", 0.0))

    if sl == 0.0 or tp == 0.0:
        return "NEUTRAL"

    start = signal_idx + 1
    end   = min(start + timeout_candles, len(df))

    if start >= len(df):
        return "NEUTRAL"

    future = df.iloc[start:end]

    for _, candle in future.iterrows():
        c_high = float(candle["high"])
        c_low  = float(candle["low"])

        if side == "BUY":
            hit_tp = c_high >= tp
            hit_sl = c_low  <= sl
        else:  # SELL
            hit_tp = c_low  <= tp
            hit_sl = c_high >= sl

        if hit_tp and hit_sl:
            return "LOSS"   # conservative: assume SL hit first on ambiguous candle
        if hit_tp:
            return "WIN"
        if hit_sl:
            return "LOSS"

    return "NEUTRAL"


def label_all(
    df: pd.DataFrame,
    signals: List[dict],
    timeout_candles: int = 48,
) -> List[dict]:
    """
    Label a list of signals, adding a 'label' key to each dict.

    Args:
        df:              full candle DataFrame
        signals:         list of signal dicts (must have 'timestamp' key)
        timeout_candles: forward-looking window in candles

    Returns:
        List of signal dicts, each with 'label': "WIN" | "LOSS" | "NEUTRAL"
        and 'signal_idx': the integer position in df.
        Signals whose timestamp isn't found in df are dropped.
    """
    ts_to_idx = {ts: i for i, ts in enumerate(df["timestamp"])}
    labeled: List[dict] = []

    for sig in signals:
        raw_ts = sig.get("timestamp")
        if raw_ts is None:
            continue
        ts = pd.Timestamp(raw_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        idx = ts_to_idx.get(ts)
        if idx is None:
            continue

        sig_copy = dict(sig)
        sig_copy["signal_idx"] = idx
        sig_copy["label"]      = label_signal(df, idx, sig, timeout_candles)
        labeled.append(sig_copy)

    return labeled
