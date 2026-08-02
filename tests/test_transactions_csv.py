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

import pandas as pd  # noqa: E402

from compute_totals import compute_daily_totals, shares_held_on_dates  # noqa: E402
from fetch_prices import aggregate_transactions, parse_transactions_csv  # noqa: E402


def _prices(rows):
    """rows: list of (ticker, date, close)."""
    return pd.DataFrame(
        [{"ticker": t, "date": d, "close": c} for t, d, c in rows]
    )


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


def test_zero_price_rsu_grant_accepted():
    """Free RSU / stock grants have a $0 cost — price 0 must be allowed and
    should drag the weighted-average cost basis down accordingly."""
    csv = """ticker,market,shares,price,action,date,avg_buy_price
SNPS,US,40,0.0,buy,2022-05-20,
SNPS,US,10,300.00,buy,2022-06-01,
"""
    txs = parse_transactions_csv(csv)
    assert len(txs) == 2
    assert txs[0]["price"] == 0.0

    h, _ = aggregate_transactions(txs)
    snps = next(x for x in h if x["ticker"] == "SNPS")
    assert snps["shares"] == pytest.approx(50.0)
    # (40*0 + 10*300) / 50 = 60
    assert snps["cost_basis"] == pytest.approx(60.0)


def test_all_zero_price_grant_has_zero_cost_basis():
    csv = "ticker,market,shares,price,action,date,avg_buy_price\nSNPS,US,100,0.0,buy,2020-09-03,\n"
    h, _ = aggregate_transactions(parse_transactions_csv(csv))
    assert h[0]["cost_basis"] == pytest.approx(0.0)


def test_negative_price_raises():
    bad = "ticker,market,shares,price,action,date,avg_buy_price\nVTI,US,5,-10,buy,2024-01-01,\n"
    with pytest.raises(ValueError, match="price cannot be negative"):
        parse_transactions_csv(bad)


def test_zero_or_negative_shares_raises():
    bad = "ticker,market,shares,price,action,date,avg_buy_price\nVTI,US,0,100,buy,2024-01-01,\n"
    with pytest.raises(ValueError, match="shares must be positive"):
        parse_transactions_csv(bad)


def test_empty_csv_returns_empty_list():
    assert parse_transactions_csv("") == []


# --------------------------------------------------------------------------
# Date-aware daily totals — a holding contributes only from its buy date, and
# reflects later top-ups. This is what makes the daily portfolio value track
# actual accumulation instead of applying today's share count to all history.
# --------------------------------------------------------------------------

PRICE_DATES = [
    ("VTI", "2024-01-01", 100.0),
    ("VTI", "2024-01-02", 110.0),
    ("VTI", "2024-01-03", 120.0),
    ("VTI", "2024-01-04", 130.0),
]


def test_shares_held_steps_on_buy_dates():
    txs = [
        {"ticker": "VTI", "shares": 10, "action": "buy", "date": "2024-01-02"},
        {"ticker": "VTI", "shares": 5, "action": "buy", "date": "2024-01-04"},
    ]
    held = shares_held_on_dates(txs, _prices(PRICE_DATES))
    by_date = dict(zip(held["date"], held["shares"]))
    assert "2024-01-01" not in by_date          # before first buy → absent
    assert by_date["2024-01-02"] == 10          # first buy
    assert by_date["2024-01-03"] == 10          # carried forward
    assert by_date["2024-01-04"] == 15          # top-up


def test_date_aware_value_excludes_pre_buy_days():
    holdings = pd.DataFrame(
        [{"ticker": "VTI", "market": "US", "shares": 15, "cost_basis": 0, "currency": "USD"}]
    )
    txs = [
        {"ticker": "VTI", "shares": 10, "action": "buy", "date": "2024-01-02"},
        {"ticker": "VTI", "shares": 5, "action": "buy", "date": "2024-01-04"},
    ]
    df = compute_daily_totals(holdings=holdings, prices=_prices(PRICE_DATES), transactions=txs)
    vals = dict(zip(df["date"], df["value"]))
    assert "2024-01-01" not in vals             # not held yet → no value
    assert vals["2024-01-02"] == pytest.approx(10 * 110.0)
    assert vals["2024-01-03"] == pytest.approx(10 * 120.0)
    assert vals["2024-01-04"] == pytest.approx(15 * 130.0)


def test_legacy_path_applies_net_shares_to_all_history():
    """Without transactions, the net share count is applied across all price
    dates (the pre-existing 'held throughout' behaviour, unchanged)."""
    holdings = pd.DataFrame(
        [{"ticker": "VTI", "market": "US", "shares": 15, "cost_basis": 0, "currency": "USD"}]
    )
    df = compute_daily_totals(holdings=holdings, prices=_prices(PRICE_DATES))
    assert len(df) == 4                          # every price date present
    assert df["shares"].eq(15).all()             # constant net shares
    vals = dict(zip(df["date"], df["value"]))
    assert vals["2024-01-01"] == pytest.approx(15 * 100.0)
