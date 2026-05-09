"""Compute per-ticker and per-day portfolio value (close_price * shares)."""
from __future__ import annotations

import argparse

import pandas as pd

from db import (
    init_db,
    load_holdings,
    load_prices,
    replace_daily_totals,
)


def compute_daily_totals() -> pd.DataFrame:
    holdings = load_holdings()
    if holdings.empty:
        return pd.DataFrame(columns=["date", "ticker", "close_price", "shares", "value", "currency"])

    prices = load_prices(tickers=holdings["ticker"].tolist())
    if prices.empty:
        return pd.DataFrame(columns=["date", "ticker", "close_price", "shares", "value", "currency"])

    merged = prices.merge(
        holdings[["ticker", "shares", "currency"]],
        on="ticker",
        how="inner",
    )
    merged = merged.rename(columns={"close": "close_price"})
    merged = merged[merged["close_price"].notna() & (merged["close_price"] > 0)]
    merged["value"] = merged["close_price"] * merged["shares"]
    return merged[["date", "ticker", "close_price", "shares", "value", "currency"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute daily_totals from prices x holdings")
    parser.parse_args()

    init_db()
    df = compute_daily_totals()
    n = replace_daily_totals(df)
    print(f"Wrote {n} daily_totals rows.")
    if not df.empty:
        print(f"Range: {df['date'].min()} → {df['date'].max()}")


if __name__ == "__main__":
    main()
