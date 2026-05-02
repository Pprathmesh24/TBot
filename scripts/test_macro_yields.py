"""Phase 9 Chunk 2 — verify 10Y TIPS real yield loader (FRED)."""
from pathlib import Path
from tbot.data.macro.yields import fetch_yields, load_yields

print("Fetching 10Y TIPS real yield from FRED (DFII10, 2019-01-01 → today)...")
df = fetch_yields(start="2019-01-01")
print(f"  rows={len(df)}  range={df.index[0].date()} → {df.index[-1].date()}")
print(f"  yield_pct min={df['yield_pct'].min():.2f}%  max={df['yield_pct'].max():.2f}%")

# Real yields went negative in 2020-2021 (hit ~-1.1%) and peaked ~2.5% in 2023
assert df["yield_pct"].min() < 0, "Expected some negative real yields (2020-2021 period)"
assert df["yield_pct"].max() < 4.0, "Real yield above 4% is unexpected — check series"
assert len(df) > 1500, "Expected >1500 trading days"
assert Path("data/raw/macro/yields.parquet").exists()

df2 = load_yields()
assert len(df2) == len(df)
print("  ✓ yields.parquet saved + round-trip OK")
print(f"\n  Note: negative values (e.g. {df['yield_pct'].min():.2f}%) = 2020-2021 stimulus era")
print("✓ Phase 9 Chunk 2 passed")
