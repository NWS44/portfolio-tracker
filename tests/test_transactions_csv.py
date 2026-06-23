"""Tests for the CSV transaction log parser and aggregator added to support
the new transaction-based holdings format.

Format: ticker, market, shares, price, action (buy/sell), date, avg_buy_price.
avg_buy_price is required on sell rows and lets us compute realized P&L
without resorting to FIFO bookkeeping.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fetch_prices import aggregate_transactions, parse_transactions_csv  # noqa: E402


CSV_OK = """ticker,market,shares,price,action,date,avg_buy_price
VTI,US,5,240.00,buy,2024-01-15,
VTI,US,5,260.00,buy,2024-06-01,
VTI,US,3,280.00,sell,2024-09-15,250.00
TSLA,US,2,200.00,buy,2024-03-10,
0050.TW,TW,1000,100.00,buy,2023-08-01,
"""


def test_parses_all_rows():
    txs = parse_transactions_csv(CSV_OK)
    assert len(txs) == 5
    by_action = {t["action"] for t in txs}
    assert by_action == {"buy", "sell"}


def test_buy_only_holdings():
    csv = "ticker,market,shares,price,action,date,avg_buy_price\nVTI,US,10,250,buy,2024-01-01,\n"
    h, r = aggregate_transactions(parse_transactions_csv(csv))
    assert len(h) == 1
    assert h[0]["ticker"] == "VTI"
    assert h[0]["shares"] == 10
    assert h[0]["cost_basis"] == 250
    assert h[0]["currency"] == "USD"
    assert r == []


def test_weighted_cost_basis_after_partial_sell():
    h, r = aggregate_transactions(parse_transactions_csv(CSV_OK))
    vti = next(x for x in h if x["ticker"] == "VTI")
    # 5*240 + 5*260 = 2500 bought; 3*250 = 750 cost of sold; 1750 / 7 = 250
    assert vti["shares"] == pytest.approx(7.0)
    assert vti["cost_basis"] == pytest.approx(250.0)

    vti_real = next(x for x in r if x["ticker"] == "VTI")
    # 3 shares × ($280 - $250) = $90
    assert vti_real["realized_pl"] == pytest.approx(90.0)


def test_sold_out_ticker_omitted_from_holdings_but_kept_in_realized():
    csv = """ticker,market,shares,price,action,date,avg_buy_price
ALL,US,10,50,buy,2024-01-01,
ALL,US,10,75,sale,2024-09-01,50.00
"""
    h, r = aggregate_transactions(parse_transactions_csv(csv))
    assert h == []  # nothing left to hold
    assert len(r) == 1
    assert r[0]["realized_pl"] == pytest.approx(250.0)  # 10 × $25


def test_sell_without_avg_buy_price_raises():
    bad = "ticker,market,shares,price,action,date,avg_buy_price\nVTI,US,5,300,sell,2024-09-15,\n"
    with pytest.raises(ValueError, match="avg_buy_price"):
        parse_transactions_csv(bad)


def test_missing_required_column_raises():
    bad = "ticker,market,shares,price,action\nVTI,US,5,300,buy\n"  # no date
    with pytest.raises(ValueError, match="missing required columns"):
        parse_transactions_csv(bad)


def test_invalid_market_raises():
    bad = "ticker,market,shares,price,action,date,avg_buy_price\nVTI,JP,5,300,buy,2024-01-01,\n"
    with pytest.raises(ValueError, match="market"):
        parse_transactions_csv(bad)


def test_action_aliases_accepted():
    csv = """ticker,market,shares,price,action,date,avg_buy_price
A,US,1,100,B,2024-01-01,
A,US,1,150,sale,2024-02-01,100
"""
    txs = parse_transactions_csv(csv)
    assert txs[0]["action"] == "buy"
    assert txs[1]["action"] == "sell"


def test_currency_inferred_from_market():
    csv = """ticker,market,shares,price,action,date,avg_buy_price
A,US,1,100,buy,2024-01-01,
B,TW,1,100,buy,2024-01-01,
"""
    txs = parse_transactions_csv(csv)
    assert txs[0]["currency"] == "USD"
    assert txs[1]["currency"] == "TWD"


def test_empty_csv_returns_empty_list():
    assert parse_transactions_csv("") == []
