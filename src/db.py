"""SQLite schema and read/write helpers for portfolio data."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "portfolio.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    ticker      TEXT PRIMARY KEY,
    market      TEXT NOT NULL,
    shares      REAL NOT NULL,
    cost_basis  REAL,
    currency    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prices (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS daily_totals (
    date         TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    close_price  REAL NOT NULL,
    shares       REAL NOT NULL,
    value        REAL NOT NULL,
    currency     TEXT NOT NULL,
    PRIMARY KEY (date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_daily_totals_date ON daily_totals(date);

CREATE TABLE IF NOT EXISTS fx_rates (
    pair  TEXT NOT NULL,
    date  TEXT NOT NULL,
    rate  REAL NOT NULL,
    PRIMARY KEY (pair, date)
);
CREATE INDEX IF NOT EXISTS idx_fx_rates_date ON fx_rates(date);

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


def _resolve(db_path: Path | None) -> Path:
    """Pick the effective DB path. Reading the module-level DB_PATH at call
    time (instead of as a default arg) lets tests monkeypatch it.
    """
    return db_path if db_path is not None else DB_PATH


@contextmanager
def connect(db_path: Path | None = None):
    db_path = _resolve(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_holdings(rows: Iterable[dict], db_path: Path | None = None) -> int:
    sql = """
    INSERT INTO holdings (ticker, market, shares, cost_basis, currency)
    VALUES (:ticker, :market, :shares, :cost_basis, :currency)
    ON CONFLICT(ticker) DO UPDATE SET
      market = excluded.market,
      shares = excluded.shares,
      cost_basis = excluded.cost_basis,
      currency = excluded.currency
    """
    rows = list(rows)
    with connect(db_path) as conn:
        conn.executemany(sql, rows)
    return len(rows)


def replace_holdings(rows: Iterable[dict], db_path: Path | None = None) -> int:
    """Replace the holdings table to mirror the YAML exactly."""
    rows = list(rows)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM holdings")
        conn.executemany(
            "INSERT INTO holdings (ticker, market, shares, cost_basis, currency) "
            "VALUES (:ticker, :market, :shares, :cost_basis, :currency)",
            rows,
        )
    return len(rows)


def upsert_prices(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """df columns: ticker, date (YYYY-MM-DD str), open, high, low, close, volume"""
    if df.empty:
        return 0
    sql = """
    INSERT INTO prices (ticker, date, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(ticker, date) DO UPDATE SET
      open = excluded.open,
      high = excluded.high,
      low = excluded.low,
      close = excluded.close,
      volume = excluded.volume
    """
    records = df[["ticker", "date", "open", "high", "low", "close", "volume"]].to_records(index=False)
    with connect(db_path) as conn:
        conn.executemany(sql, list(records))
    return len(df)


def upsert_daily_totals(df: pd.DataFrame, db_path: Path | None = None) -> int:
    if df.empty:
        return 0
    sql = """
    INSERT INTO daily_totals (date, ticker, close_price, shares, value, currency)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(date, ticker) DO UPDATE SET
      close_price = excluded.close_price,
      shares = excluded.shares,
      value = excluded.value,
      currency = excluded.currency
    """
    records = df[["date", "ticker", "close_price", "shares", "value", "currency"]].to_records(index=False)
    with connect(db_path) as conn:
        conn.executemany(sql, list(records))
    return len(df)


def replace_daily_totals(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """Wipe and rewrite daily_totals so removed holdings stop contributing."""
    with connect(db_path) as conn:
        conn.execute("DELETE FROM daily_totals")
        if df.empty:
            return 0
        records = df[["date", "ticker", "close_price", "shares", "value", "currency"]].to_records(index=False)
        conn.executemany(
            "INSERT INTO daily_totals (date, ticker, close_price, shares, value, currency) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            list(records),
        )
    return len(df)


def load_holdings(db_path: Path | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM holdings ORDER BY market, ticker", conn)


def load_prices(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM prices WHERE 1=1"
    params: list = []
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        sql += f" AND ticker IN ({placeholders})"
        params.extend(tickers)
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date, ticker"
    with connect(db_path) as conn:
        return pd.read_sql(sql, conn, params=params)


def load_daily_totals(
    start: str | None = None,
    end: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM daily_totals WHERE 1=1"
    params: list = []
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date, ticker"
    with connect(db_path) as conn:
        return pd.read_sql(sql, conn, params=params)


def latest_price_date(ticker: str, db_path: Path | None = None) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(date) AS d FROM prices WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row["d"] if row and row["d"] else None


def upsert_fx_rates(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """df columns: pair, date, rate"""
    if df.empty:
        return 0
    sql = """
    INSERT INTO fx_rates (pair, date, rate)
    VALUES (?, ?, ?)
    ON CONFLICT(pair, date) DO UPDATE SET rate = excluded.rate
    """
    records = df[["pair", "date", "rate"]].to_records(index=False)
    with connect(db_path) as conn:
        conn.executemany(sql, list(records))
    return len(df)


def load_fx_rates(
    pair: str | None = None,
    start: str | None = None,
    end: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM fx_rates WHERE 1=1"
    params: list = []
    if pair:
        sql += " AND pair = ?"
        params.append(pair)
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date"
    with connect(db_path) as conn:
        return pd.read_sql(sql, conn, params=params)


def latest_fx_date(pair: str, db_path: Path | None = None) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(date) AS d FROM fx_rates WHERE pair = ?", (pair,)
        ).fetchone()
    return row["d"] if row and row["d"] else None


def set_meta(key: str, value: str, db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_meta(key: str, db_path: Path | None = None) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None
