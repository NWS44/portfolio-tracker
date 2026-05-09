"""One-shot job: sync holdings, fetch latest prices, recompute daily totals.

Schedule this with Windows Task Scheduler after market close.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from compute_totals import compute_daily_totals  # noqa: E402
from db import init_db, load_holdings, replace_daily_totals  # noqa: E402
from fetch_prices import (  # noqa: E402
    fetch_for_tickers,
    fetch_fx_rates,
    sync_holdings_from_yaml,
)


def main() -> int:
    init_db()
    n = sync_holdings_from_yaml()
    print(f"[holdings] synced {n} rows")

    holdings = load_holdings()
    if holdings.empty:
        print("[abort] no holdings configured")
        return 1

    rows = fetch_for_tickers(holdings["ticker"].tolist())
    print(f"[prices] {rows} new rows")

    fx_rows = fetch_fx_rates()
    print(f"[fx]     {fx_rows} new rows")

    df = compute_daily_totals()
    n = replace_daily_totals(df)
    print(f"[totals] {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
