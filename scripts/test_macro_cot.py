"""Phase 9 Chunk 3 — verify CFTC COT Gold loader."""
from pathlib import Path
from tbot.data.macro.cot import fetch_cot, load_cot

print("Fetching CFTC COT Gold (2019 → today) from cftc.gov...")
df = fetch_cot(start_year=2019)
print(f"  rows={len(df)}  range={df.index[0].date()} → {df.index[-1].date()}")
print(f"  mm_net  min={df['mm_net'].min():,.0f}  max={df['mm_net'].max():,.0f}")
print(f"  mm_net_pct  min={df['mm_net_pct'].min():.2f}  max={df['mm_net_pct'].max():.2f}")

assert len(df) > 100, "Expected >100 weekly COT reports"
assert df["mm_net_pct"].between(-1, 1).all(), "mm_net_pct should be in [-1, 1]"
assert Path("data/raw/macro/cot_gold.parquet").exists()

df2 = load_cot()
assert len(df2) == len(df)
print("  ✓ cot_gold.parquet saved + round-trip OK")
print("\n✓ Phase 9 Chunk 3 passed")
