"""Phase 9 Chunk 1 — verify DXY + VIX loaders."""
from pathlib import Path
from tbot.data.macro.dxy import fetch_dxy, load_dxy
from tbot.data.macro.vix import fetch_vix, load_vix

print("Fetching DXY (2019-01-01 → today)...")
dxy = fetch_dxy(start="2019-01-01")
print(f"  rows={len(dxy)}  range={dxy.index[0].date()} → {dxy.index[-1].date()}")
print(f"  close min={dxy['close'].min():.2f}  max={dxy['close'].max():.2f}")
assert len(dxy) > 1000, "Expected >1000 trading days"
assert Path("data/raw/macro/dxy.parquet").exists()
print("  ✓ dxy.parquet saved")

print("\nFetching VIX (2019-01-01 → today)...")
vix = fetch_vix(start="2019-01-01")
print(f"  rows={len(vix)}  range={vix.index[0].date()} → {vix.index[-1].date()}")
print(f"  close min={vix['close'].min():.2f}  max={vix['close'].max():.2f}")
assert len(vix) > 1000, "Expected >1000 trading days"
assert Path("data/raw/macro/vix.parquet").exists()
print("  ✓ vix.parquet saved")

print("\nRound-trip load test...")
dxy2 = load_dxy()
vix2 = load_vix()
assert len(dxy2) == len(dxy)
assert len(vix2) == len(vix)
print("  ✓ parquet round-trip OK")

print("\n✓ Phase 9 Chunk 1 passed")
