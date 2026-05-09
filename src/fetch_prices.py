"""Fetch daily OHLC from Yahoo Finance for US and TW tickers."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from db import (
    DB_PATH,
    init_db,
    latest_fx_date,
    latest_price_date,
    load_holdings,
    replace_holdings,
    upsert_fx_rates,
    upsert_prices,
)

FX_TICKERS = {
    "USDTWD": "TWD=X",
}

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOLDINGS = ROOT / "config" / "holdings.yaml"


def parse_holdings_yaml(content: str) -> list[dict]:
    """Parse holdings.yaml content (string) into a list of dicts."""
    data = yaml.safe_load(content) or {}
    rows = data.get("holdings", [])
    for r in rows:
        r.setdefault("cost_basis", None)
        r.setdefault("currency", "USD" if r.get("market") == "US" else "TWD")
    return rows


def load_holdings_yaml(path: Path = DEFAULT_HOLDINGS) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_holdings_yaml(f.read())


def sync_holdings_from_yaml(path: Path = DEFAULT_HOLDINGS) -> int:
    rows = load_holdings_yaml(path)
    return replace_holdings(rows)


def sync_holdings_from_rows(rows: list[dict]) -> int:
    """Sync holdings from a parsed list of dicts (e.g. from uploaded YAML)."""
    return replace_holdings(rows)


def _latest_quote_row(ticker: str) -> pd.DataFrame:
    """Fetch the latest quote (current/last price) via the quote endpoint.

    yfinance's historical OHLC API can lag a few hours after market close.
    This pulls regularMarketPrice + regularMarketTime to fill the gap.
    Returns an empty DataFrame if no live quote is available.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return pd.DataFrame()

    price = info.get("regularMarketPrice")
    market_time = info.get("regularMarketTime")
    if not price or price <= 0 or not market_time:
        return pd.DataFrame()

    tz_name = info.get("exchangeTimezoneName")
    tz = None
    if tz_name and ZoneInfo is not None:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = None
    if tz is None:
        tz = timezone.utc

    dt = datetime.fromtimestamp(market_time, tz=tz)
    return pd.DataFrame([
        {
            "ticker": ticker,
            "date": dt.strftime("%Y-%m-%d"),
            "open": info.get("regularMarketOpen") or price,
            "high": info.get("regularMarketDayHigh") or price,
            "low": info.get("regularMarketDayLow") or price,
            "close": float(price),
            "volume": info.get("regularMarketVolume") or 0,
        }
    ])


def _download_one(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Download one ticker; returns long-form DataFrame ready for DB.
    Use either (start, end) for a date range or `period` (e.g. 'max') for full history.
    Drops rows with null or zero close price (market holidays / bad data).
    """
    kwargs = dict(progress=False, auto_adjust=False, actions=False)
    if period:
        df = yf.download(ticker, period=period, **kwargs)
    else:
        df = yf.download(ticker, start=start, end=end, **kwargs)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    keep = ["ticker", "date", "open", "high", "low", "close", "volume"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[keep]
    df = df[df["close"].notna() & (df["close"] > 0)]
    return df


def fetch_fx_rates(
    pairs: dict[str, str] | None = None,
    full_refresh: bool = False,
    max_history: bool = False,
) -> int:
    """Pull FX rates (default USDTWD via TWD=X). Stores daily close as the rate.
    First fetch (or max_history) pulls period='max'; otherwise incremental.
    """
    init_db()
    pairs = pairs or FX_TICKERS
    today = date.today()
    end = (today + timedelta(days=1)).isoformat()
    total = 0

    for pair, yf_symbol in pairs.items():
        period: str | None = None
        start: str | None = None

        if max_history:
            period = "max"
        else:
            last = latest_fx_date(pair)
            if last is None:
                period = "max"
            elif full_refresh:
                period = "max"
            else:
                start = datetime.strptime(last, "%Y-%m-%d").date().isoformat()
                if start > today.isoformat():
                    print(f"[skip] FX {pair} — already current ({last})")
                    continue

        try:
            df = _download_one(yf_symbol, start=start, end=end if start else None, period=period)
        except Exception as e:
            print(f"[error] FX {pair}: {e}")
            continue

        if df.empty:
            label = f"period={period}" if period else f"{start} → {end}"
            print(f"[empty] FX {pair} — no rows ({label})")
            continue

        out = pd.DataFrame({
            "pair": pair,
            "date": df["date"],
            "rate": df["close"],
        }).dropna(subset=["rate"])
        out = out[out["rate"] > 0]
        n = upsert_fx_rates(out)
        total += n
        print(f"[ok]   FX {pair}: {n} rows ({out['date'].min()} → {out['date'].max()})")

    return total


def fetch_for_tickers(
    tickers: list[str],
    full_refresh: bool = False,
    max_history: bool = False,
) -> int:
    """Pull prices for given tickers. By default:
    - first time we see a ticker → period='max' (all history Yahoo has)
    - subsequent runs → incremental from the last stored date
    full_refresh re-pulls the existing range; max_history forces period='max' for everyone.
    """
    init_db()
    today = date.today()
    end = (today + timedelta(days=1)).isoformat()
    total = 0

    for t in tickers:
        period: str | None = None
        start: str | None = None

        if max_history:
            period = "max"
        else:
            last = latest_price_date(t)
            if last is None:
                period = "max"
            elif full_refresh:
                period = "max"
            else:
                start = datetime.strptime(last, "%Y-%m-%d").date().isoformat()
                if start > today.isoformat():
                    print(f"[skip] {t} — already current ({last})")
                    continue

        try:
            df = _download_one(t, start=start, end=end if start else None, period=period)
        except Exception as e:
            print(f"[error] {t}: {e}")
            continue

        # Supplement with latest quote if historical API is behind
        try:
            quote_df = _latest_quote_row(t)
            if not quote_df.empty:
                quote_date = quote_df["date"].iloc[0]
                if df.empty or quote_date > df["date"].max():
                    df = pd.concat([df, quote_df], ignore_index=True)
                    print(f"[live] {t} — added live quote for {quote_date} (close={quote_df['close'].iloc[0]})")
        except Exception as e:
            print(f"[warn] {t} quote supplement failed: {e}")

        if df.empty:
            label = f"period={period}" if period else f"{start} → {end}"
            print(f"[empty] {t} — no rows ({label})")
            continue

        n = upsert_prices(df)
        total += n
        print(f"[ok]   {t}: {n} rows ({df['date'].min()} → {df['date'].max()})")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch stock prices for holdings")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_HOLDINGS,
        help="Path to holdings.yaml",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Re-download full history (period='max') even if data exists",
    )
    parser.add_argument(
        "--max-history",
        action="store_true",
        help="Force period='max' for every ticker (alias of --full-refresh)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Only fetch these tickers (default: all from holdings.yaml)",
    )
    args = parser.parse_args()

    init_db()
    n = sync_holdings_from_yaml(args.config)
    print(f"Synced {n} holdings from {args.config}")

    holdings = load_holdings()
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = holdings["ticker"].tolist()
    if not tickers:
        print("No holdings found — edit config/holdings.yaml first.")
        return

    full = args.full_refresh or args.max_history
    rows = fetch_for_tickers(tickers, full_refresh=full, max_history=args.max_history)
    fx_rows = fetch_fx_rates(full_refresh=full, max_history=args.max_history)
    print(f"Done — wrote {rows} price rows and {fx_rows} FX rows to {DB_PATH}")


if __name__ == "__main__":
    main()
