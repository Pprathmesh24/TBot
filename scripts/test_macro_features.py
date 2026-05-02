"""Phase 9 Chunk 4 — verify macro_features lookup."""
import pandas as pd
from tbot.features.macro_features import get_macro_features

# Three test timestamps spread across the dataset
timestamps = [
    pd.Timestamp("2020-03-20 14:00", tz="UTC"),   # COVID crash — VIX ~80, yields negative
    pd.Timestamp("2022-06-15 10:00", tz="UTC"),   # Fed hiking — yields rising fast
    pd.Timestamp("2024-09-10 16:00", tz="UTC"),   # Recent — should have all series
]

for ts in timestamps:
    f = get_macro_features(ts)
    print(f"\n{ts.date()}")
    for k, v in f.items():
        print(f"  {k:<20} {v:>10.4f}")

# Basic sanity checks
f_covid = get_macro_features(timestamps[0])
assert f_covid["vix_close"] > 30, "COVID VIX should be elevated"

f_recent = get_macro_features(timestamps[2])
assert all(isinstance(v, float) for v in f_recent.values()), "All values should be floats"
assert f_recent["cot_mm_net_pct"] >= -1 and f_recent["cot_mm_net_pct"] <= 1, "mm_net_pct out of range"
assert f_recent["macro_risk_on"] in (0.0, 1.0), "regime flag must be 0 or 1"
assert f_recent["macro_risk_off"] in (0.0, 1.0), "regime flag must be 0 or 1"

print("\n✓ Phase 9 Chunk 4 passed — macro_features lookup OK")
