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


EMPTY_COLUMNS = ["date", "ticker", "close_price", "shares", "value", "currency"]


def shares_held_on_dates(
    transactions: list[dict], prices: pd.DataFrame
) -> pd.DataFrame:
    """Return a (ticker, date, shares) frame: the net shares held as of each
    price date, given a dated transaction log.

    Shares step up on buy dates and down on sell dates. For every price date we
    take the cumulative net position as of the most recent transaction on or
    before that date (a step function). Dates before a ticker's first
    transaction, and any date where the running position is <= 0, are omitted —
    so a ticker only contributes to the portfolio from its first purchase.
    """
    if not transactions or prices.empty:
        return pd.DataFrame(columns=["ticker", "date", "shares"])

    tx = pd.DataFrame(transactions)
    tx["delta"] = tx["shares"].where(tx["action"] == "buy", -tx["shares"])
    deltas = (
        tx.groupby(["ticker", "date"], as_index=False)["delta"].sum()
    )
    deltas["_d"] = pd.to_datetime(deltas["date"])
    deltas = deltas.sort_values(["ticker", "_d"])
    deltas["shares"] = deltas.groupby("ticker")["delta"].cumsum()

    price_dates = prices[["ticker", "date"]].drop_duplicates().copy()
    price_dates["_d"] = pd.to_datetime(price_dates["date"])
    price_dates = price_dates.sort_values("_d")

    # For each price date, carry the last known net position forward (a share
    # count only changes on transaction dates). merge_asof needs the join key
    # sorted and datetime-typed, hence the temporary `_d` column.
    merged = pd.merge_asof(
        price_dates,
        deltas[["ticker", "_d", "shares"]].sort_values("_d"),
        on="_d",
        by="ticker",
        direction="backward",
    )
    merged = merged[merged["shares"].notna() & (merged["shares"] > 0)]
    return merged[["ticker", "date", "shares"]]


def compute_daily_totals(
    holdings: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    transactions: list[dict] | None = None,
) -> pd.DataFrame:
    """Compute close_price * shares per ticker per day.

    `holdings` and `prices` can be passed in for an in-memory, per-session
    computation (used by the multi-user web app so nothing is read from the
    shared DB). When omitted they fall back to the shared DB tables, which is
    what the CLI scripts rely on.

    When `transactions` (a dated buy/sell log) is supplied, the share count is
    date-aware: each ticker contributes shares held *as of that date*, so a
    holding only adds to the portfolio value from its buy date onward (and
    reflects later top-ups / partial sells). Without it, the current net
    `shares` from `holdings` is applied across the whole price history (the
    legacy "held throughout" behaviour, used for YAML/paste input).
    """
    if holdings is None:
        holdings = load_holdings()
    if holdings.empty:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    if prices is None:
        prices = load_prices(tickers=holdings["ticker"].tolist())
    if prices.empty:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    prices = prices.rename(columns={"close": "close_price"})
    prices = prices[prices["close_price"].notna() & (prices["close_price"] > 0)]

    if transactions:
        shares_ts = shares_held_on_dates(transactions, prices)
        merged = prices.merge(shares_ts, on=["ticker", "date"], how="inner")
        currency = holdings[["ticker", "currency"]].drop_duplicates()
        merged = merged.merge(currency, on="ticker", how="left")
    else:
        merged = prices.merge(
            holdings[["ticker", "shares", "currency"]],
            on="ticker",
            how="inner",
        )

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
