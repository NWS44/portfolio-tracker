"""Tests covering the bug where switching holdings (e.g. loading the
TEMPLATE_YAML in the dashboard) left stale rows in `daily_totals`,
causing wrong totals on the dashboard.

The fix lives in src/app.py:_ensure_prices_loaded — when holdings change
or when daily_totals contains tickers no longer in holdings, the totals
are recomputed via compute_daily_totals() + replace_daily_totals().

These tests verify the underlying compute / replace primitives behave
correctly, plus parse_holdings_yaml's contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from compute_totals import compute_daily_totals  # noqa: E402
from db import (  # noqa: E402
    init_db,
    load_daily_totals,
    load_holdings,
    replace_daily_totals,
    replace_holdings,
    upsert_prices,
)
from fetch_prices import parse_holdings_yaml  # noqa: E402


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point all db helpers at a fresh sqlite file so tests are isolated."""
    db_path = tmp_path / "test.db"
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    init_db(db_path=db_path)
    return db_path


def _seed_prices(db_path):
    """Insert a couple of trading days for VTI, 0050.TW, plus stale tickers."""
    prices = pd.DataFrame(
        [
            # VTI (USD)
            {"ticker": "VTI", "date": "2026-05-07", "open": 360, "high": 363, "low": 359, "close": 362.0, "volume": 1000},
            {"ticker": "VTI", "date": "2026-05-08", "open": 362, "high": 365, "low": 361, "close": 363.0, "volume": 1100},
            # 0050.TW (TWD)
            {"ticker": "0050.TW", "date": "2026-05-07", "open": 96, "high": 98, "low": 95, "close": 97.7, "volume": 200},
            {"ticker": "0050.TW", "date": "2026-05-08", "open": 96, "high": 97, "low": 95, "close": 97.0, "volume": 250},
            # Stale ticker — was held previously, no longer in holdings
            {"ticker": "PLTR", "date": "2026-05-07", "open": 130, "high": 140, "low": 128, "close": 137.0, "volume": 500},
            {"ticker": "PLTR", "date": "2026-05-08", "open": 137, "high": 142, "low": 135, "close": 137.8, "volume": 520},
        ]
    )
    upsert_prices(prices, db_path=db_path)


# --------------------------------------------------------------------------
# parse_holdings_yaml — input contract
# --------------------------------------------------------------------------

def test_parse_holdings_yaml_basic():
    yaml = """
holdings:
  - ticker: VTI
    market: US
    shares: 10
    cost_basis: 250.00
    currency: USD
"""
    rows = parse_holdings_yaml(yaml)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "VTI"
    assert rows[0]["shares"] == 10
    assert rows[0]["currency"] == "USD"


def test_parse_holdings_yaml_defaults_currency_from_market():
    yaml = """
holdings:
  - ticker: 2330.TW
    market: TW
    shares: 100
  - ticker: AAPL
    market: US
    shares: 5
"""
    rows = parse_holdings_yaml(yaml)
    assert rows[0]["currency"] == "TWD"
    assert rows[1]["currency"] == "USD"


def test_parse_holdings_yaml_defaults_cost_basis_to_none():
    yaml = """
holdings:
  - ticker: VTI
    market: US
    shares: 10
"""
    rows = parse_holdings_yaml(yaml)
    assert rows[0]["cost_basis"] is None


def test_parse_holdings_yaml_empty_returns_empty_list():
    assert parse_holdings_yaml("") == []
    assert parse_holdings_yaml("holdings: []") == []


# --------------------------------------------------------------------------
# compute_daily_totals — only emits rows for current holdings, with their
# current `shares`. This is the core safety net that makes a holdings
# change clean up stale data on recompute.
# --------------------------------------------------------------------------

def test_compute_daily_totals_uses_only_current_holdings(tmp_db):
    """The bug: when switching from 6 real holdings to 2 template holdings,
    compute_daily_totals must only return rows for the 2 current holdings,
    even if prices for the old 6 are still in the prices table.
    """
    _seed_prices(tmp_db)

    # Apply the template holdings (2 tickers)
    replace_holdings([
        {"ticker": "VTI", "market": "US", "shares": 10, "cost_basis": 250.0, "currency": "USD"},
        {"ticker": "0050.TW", "market": "TW", "shares": 1000, "cost_basis": 100.0, "currency": "TWD"},
    ])

    df = compute_daily_totals()

    # Stale ticker (PLTR) must be absent
    assert "PLTR" not in df["ticker"].unique()
    # Only the current holdings appear
    assert set(df["ticker"].unique()) == {"VTI", "0050.TW"}


def test_compute_daily_totals_uses_new_share_counts(tmp_db):
    """If the user changes the share count (e.g. from 94 → 10 on VTI), the
    recomputed totals must use the new value, not the historical one.
    """
    _seed_prices(tmp_db)

    # First commit: 94 shares
    replace_holdings([{"ticker": "VTI", "market": "US", "shares": 94, "cost_basis": 340, "currency": "USD"}])
    first = compute_daily_totals()
    vti_first = first[(first["ticker"] == "VTI") & (first["date"] == "2026-05-08")].iloc[0]
    assert vti_first["shares"] == 94
    assert vti_first["value"] == pytest.approx(94 * 363.0)

    # Switch to template: 10 shares
    replace_holdings([{"ticker": "VTI", "market": "US", "shares": 10, "cost_basis": 250, "currency": "USD"}])
    second = compute_daily_totals()
    vti_second = second[(second["ticker"] == "VTI") & (second["date"] == "2026-05-08")].iloc[0]
    assert vti_second["shares"] == 10
    assert vti_second["value"] == pytest.approx(10 * 363.0)


def test_compute_daily_totals_value_equals_shares_times_close(tmp_db):
    """Sanity check: value column = close_price × shares."""
    _seed_prices(tmp_db)
    replace_holdings([
        {"ticker": "VTI", "market": "US", "shares": 10, "cost_basis": 250, "currency": "USD"},
        {"ticker": "0050.TW", "market": "TW", "shares": 1000, "cost_basis": 100, "currency": "TWD"},
    ])

    df = compute_daily_totals()
    df["expected"] = df["close_price"] * df["shares"]
    pd.testing.assert_series_equal(
        df["value"].reset_index(drop=True),
        df["expected"].reset_index(drop=True),
        check_names=False,
    )


def test_compute_daily_totals_drops_zero_close_rows(tmp_db):
    """Rows where close_price is 0 or NaN (market holiday, bad data) are
    excluded so they don't render as a drop-to-zero in charts.
    """
    _seed_prices(tmp_db)
    upsert_prices(
        pd.DataFrame([{
            "ticker": "VTI", "date": "2026-05-09",
            "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0,
        }]),
        db_path=tmp_db,
    )
    replace_holdings([{"ticker": "VTI", "market": "US", "shares": 10, "cost_basis": 250, "currency": "USD"}])

    df = compute_daily_totals()
    assert "2026-05-09" not in df["date"].unique()


# --------------------------------------------------------------------------
# replace_daily_totals — wipes the table before inserting, so stale rows
# from a previous holdings set disappear.
# --------------------------------------------------------------------------

def test_replace_daily_totals_wipes_stale_rows(tmp_db):
    """Direct test of the cleanup primitive: after replace_daily_totals,
    the table must contain ONLY the rows passed in.
    """
    _seed_prices(tmp_db)

    # Pretend user had 3 holdings: write totals for all 3
    replace_holdings([
        {"ticker": "VTI", "market": "US", "shares": 94, "cost_basis": 340, "currency": "USD"},
        {"ticker": "0050.TW", "market": "TW", "shares": 262000, "cost_basis": 41, "currency": "TWD"},
        {"ticker": "PLTR", "market": "US", "shares": 85, "cost_basis": 185, "currency": "USD"},
    ])
    replace_daily_totals(compute_daily_totals())
    before = load_daily_totals()
    assert set(before["ticker"].unique()) == {"VTI", "0050.TW", "PLTR"}

    # Switch to template (2 holdings) and recompute
    replace_holdings([
        {"ticker": "VTI", "market": "US", "shares": 10, "cost_basis": 250, "currency": "USD"},
        {"ticker": "0050.TW", "market": "TW", "shares": 1000, "cost_basis": 100, "currency": "TWD"},
    ])
    replace_daily_totals(compute_daily_totals())
    after = load_daily_totals()

    # PLTR is gone
    assert "PLTR" not in after["ticker"].unique()
    # And VTI's shares are now 10, not 94
    vti_latest = after[(after["ticker"] == "VTI") & (after["date"] == "2026-05-08")].iloc[0]
    assert vti_latest["shares"] == 10


def test_replace_daily_totals_with_empty_df_clears_table(tmp_db):
    """Empty input wipes the table (no rows for empty holdings)."""
    _seed_prices(tmp_db)
    replace_holdings([{"ticker": "VTI", "market": "US", "shares": 10, "cost_basis": 250, "currency": "USD"}])
    replace_daily_totals(compute_daily_totals())
    assert not load_daily_totals().empty

    replace_daily_totals(pd.DataFrame())
    assert load_daily_totals().empty


# --------------------------------------------------------------------------
# Integration: end-to-end "load template" simulation
# --------------------------------------------------------------------------

def test_full_template_load_scenario(tmp_db):
    """Reproduces the user-reported bug end to end:

    1. User has 3 real holdings; daily_totals reflects them.
    2. User pastes the template (2 holdings) and applies.
    3. After sync + recompute, the dashboard data must show ONLY the 2
       template holdings with the template's share counts.
    """
    _seed_prices(tmp_db)

    # --- Initial state: 3 holdings ---
    real_holdings = [
        {"ticker": "VTI", "market": "US", "shares": 94, "cost_basis": 340, "currency": "USD"},
        {"ticker": "0050.TW", "market": "TW", "shares": 262000, "cost_basis": 41, "currency": "TWD"},
        {"ticker": "PLTR", "market": "US", "shares": 85, "cost_basis": 185, "currency": "USD"},
    ]
    replace_holdings(real_holdings)
    replace_daily_totals(compute_daily_totals())

    # --- User loads the template ---
    template_yaml = """
holdings:
  - ticker: VTI
    market: US
    shares: 10
    cost_basis: 250.00
    currency: USD
  - ticker: 0050.TW
    market: TW
    shares: 1000
    cost_basis: 100.00
    currency: TWD
"""
    template_rows = parse_holdings_yaml(template_yaml)
    replace_holdings(template_rows)

    # Recompute step (this is what the fix in app.py now triggers)
    replace_daily_totals(compute_daily_totals())

    # --- Verify the dashboard would now show clean data ---
    holdings_after = load_holdings()
    totals_after = load_daily_totals()

    # Holdings table reflects only the template
    assert set(holdings_after["ticker"]) == {"VTI", "0050.TW"}

    # daily_totals has no stale tickers
    assert "PLTR" not in totals_after["ticker"].unique()

    # Share counts match the template
    vti = totals_after[(totals_after["ticker"] == "VTI") & (totals_after["date"] == "2026-05-08")].iloc[0]
    assert vti["shares"] == 10
    assert vti["value"] == pytest.approx(10 * 363.0)  # 3,630 USD

    tw = totals_after[(totals_after["ticker"] == "0050.TW") & (totals_after["date"] == "2026-05-08")].iloc[0]
    assert tw["shares"] == 1000
    assert tw["value"] == pytest.approx(1000 * 97.0)  # 97,000 TWD
