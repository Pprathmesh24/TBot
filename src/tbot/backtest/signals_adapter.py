"""
Converts AITradingAgent signal dicts → aligned NumPy arrays for vectorbt.

Input:  list of signal dicts from AITradingAgent._generate_trading_signals()
        + the candle DataFrame (timestamp index)

Output: SignalArrays — four arrays of length == len(df), aligned to candle timestamps
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalArrays:
    """Vectorbt-ready arrays aligned to the candle DataFrame index."""
    entries:  np.ndarray   # bool  — True = enter long at this candle
    exits:    np.ndarray   # bool  — True = exit  long at this candle
    sl_stop:  np.ndarray   # float — stop-loss  price (0.0 where no signal)
    tp_stop:  np.ndarray   # float — take-profit price (0.0 where no signal)
    n_signals: int         # total BUY signals found


def adapt(
    signals: list[dict],
    candles: pd.DataFrame,
    min_confidence: float = 0.0,
) -> SignalArrays:
    """
    Align agent signal dicts to the candle DataFrame index.

    Args:
        signals:        output of AITradingAgent._generate_trading_signals()
        candles:        DataFrame from load_candles() — must have a 'timestamp' column
        min_confidence: drop signals below this threshold (default 0 = keep all)

    Returns:
        SignalArrays with four bool/float arrays of length len(candles)
    """
    n = len(candles)
    entries = np.zeros(n, dtype=bool)
    exits   = np.zeros(n, dtype=bool)
    sl_stop = np.zeros(n, dtype=float)
    tp_stop = np.zeros(n, dtype=float)

    # Build a fast timestamp → integer index lookup
    ts_to_idx: dict[pd.Timestamp, int] = {
        ts: i for i, ts in enumerate(candles["timestamp"])
    }

    buy_count = 0
    for sig in signals:
        if sig.get("confidence", 1.0) < min_confidence:
            continue

        sig_ts = pd.Timestamp(sig["timestamp"])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.tz_localize("UTC")

        idx = ts_to_idx.get(sig_ts)
        if idx is None:
            continue  # signal timestamp not in the candle window

        if sig.get("type") == "BUY":
            entries[idx] = True
            sl_stop[idx] = float(sig.get("stop_loss",   0.0))
            tp_stop[idx] = float(sig.get("take_profit", 0.0))
            buy_count += 1
        elif sig.get("type") == "SELL":
            exits[idx] = True

    return SignalArrays(
        entries=entries,
        exits=exits,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        n_signals=buy_count,
    )
