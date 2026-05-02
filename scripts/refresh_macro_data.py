"""
Refresh all macro data sources.

Run daily (e.g. 00:30 UTC after markets close) to keep parquets current:
    .venv/bin/python scripts/refresh_macro_data.py

Each source is fetched independently — one failure does not block the others.
"""

import datetime
import sys

START = "2019-01-01"


def _report(name: str, df) -> None:
    print(f"  {name:<10} {len(df):>5} rows  {df.index[0].date()} → {df.index[-1].date()}")


def main() -> int:
    errors: list[str] = []
    today = datetime.date.today().isoformat()
    print(f"Refreshing macro data  (start={START}  today={today})")

    try:
        from tbot.data.macro.dxy import fetch_dxy
        df = fetch_dxy(start=START)
        _report("DXY", df)
    except Exception as exc:
        print(f"  DXY        ERROR: {exc}")
        errors.append("DXY")

    try:
        from tbot.data.macro.vix import fetch_vix
        df = fetch_vix(start=START)
        _report("VIX", df)
    except Exception as exc:
        print(f"  VIX        ERROR: {exc}")
        errors.append("VIX")

    try:
        from tbot.data.macro.yields import fetch_yields
        df = fetch_yields(start=START)
        _report("yields", df)
    except Exception as exc:
        print(f"  yields     ERROR: {exc}")
        errors.append("yields")

    try:
        from tbot.data.macro.cot import fetch_cot
        df = fetch_cot(start_year=2019)
        _report("COT", df)
    except Exception as exc:
        print(f"  COT        ERROR: {exc}")
        errors.append("COT")

    if errors:
        print(f"\n  {len(errors)} source(s) failed: {', '.join(errors)}")
        return 1

    print("\n✓ All macro data refreshed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
