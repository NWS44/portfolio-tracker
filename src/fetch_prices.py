"""Fetch daily OHLC from Yahoo Finance for US and TW tickers."""
from __future__ import annotations

import argparse
import csv
import io
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

# yfinance (and its native dep curl_cffi) is only needed when we actually hit
# Yahoo to fetch prices. Importing it eagerly means an install/ABI problem on
# the host (e.g. Streamlit Cloud bumping its Python) takes down the ENTIRE app
# at import time — even though parsing holdings and rendering cached prices
# don't need it. So we import it lazily via `_get_yf()` and surface a clear
# error only when a fetch is attempted.
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _get_yf():
    """Import yfinance on demand. Raises a clear RuntimeError if it (or its
    native dependency curl_cffi) can't be imported in this environment."""
    try:
        import yfinance as yf
    except Exception as e:  # ImportError, or curl_cffi ABI errors, etc.
        raise RuntimeError(
            "yfinance is unavailable in this environment, so prices can't be "
            f"fetched right now ({type(e).__name__}: {e}). Cached prices in the "
            "database still display normally."
        ) from e
    return yf

from db import (
    DB_PATH,
    init_db,
    latest_fx_date,
    latest_price_date,
    load_holdings,
    replace_holdings,
    set_meta,
    upsert_fx_rates,
    upsert_prices,
)

FX_TICKERS = {
    "USDTWD": "TWD=X",
}

# Re-fetch a small window before the last stored date so late corrections
# (split/dividend adjustments, late prints) overwrite stale rows.
INCREMENTAL_LOOKBACK_DAYS = 3


def _taipei_now_str() -> str:
    """Return current Taipei time as 'YYYY-MM-DD HH:MM' (24hr)."""
    tz = ZoneInfo("Asia/Taipei") if ZoneInfo is not None else timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M")

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


TX_REQUIRED_COLS = ("ticker", "market", "shares", "price", "action", "date")


def parse_transactions_csv(content: str) -> list[dict]:
    """Parse a CSV transaction log into a list of transaction dicts.

    Required columns (case-insensitive, comma-separated):
      ticker, market, shares, price, action, date

    Optional column:
      avg_buy_price — required on rows where action is 'sell'/'sale'.

    `action` accepts 'buy', 'b', 'sell', 'sale', or 's'. `date` is YYYY-MM-DD.
    Currency is inferred from market (US→USD, TW→TWD).
    """
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        return []

    headers = {h.strip().lower(): h for h in reader.fieldnames if h}
    missing = [c for c in TX_REQUIRED_COLS if c not in headers]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(missing)}. "
            f"Expected: {', '.join(TX_REQUIRED_COLS)}, [avg_buy_price]."
        )

    transactions: list[dict] = []
    for row_num, raw in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        if not row.get("ticker"):
            continue  # silently skip empty rows

        ticker = row["ticker"].upper()
        market = row.get("market", "").upper()
        if market not in ("US", "TW"):
            raise ValueError(f"Row {row_num}: market must be US or TW (got {market!r}).")

        try:
            shares = float(row["shares"])
            price = float(row["price"])
        except ValueError as e:
            raise ValueError(f"Row {row_num}: invalid shares/price — {e}")
        if shares <= 0:
            raise ValueError(f"Row {row_num}: shares must be positive.")
        # price == 0 is allowed (e.g. free RSU / stock grant with zero cost);
        # only a negative price is invalid.
        if price < 0:
            raise ValueError(f"Row {row_num}: price cannot be negative.")

        action_raw = row.get("action", "").lower()
        if action_raw in ("buy", "b"):
            action = "buy"
        elif action_raw in ("sell", "sale", "s"):
            action = "sell"
        else:
            raise ValueError(f"Row {row_num}: action must be buy/sell (got {action_raw!r}).")

        try:
            tx_date = datetime.strptime(row["date"], "%Y-%m-%d").date().isoformat()
        except ValueError:
            raise ValueError(f"Row {row_num}: date must be YYYY-MM-DD (got {row.get('date')!r}).")

        avg_buy_price = None
        if action == "sell":
            avg_str = row.get("avg_buy_price", "")
            if not avg_str:
                raise ValueError(
                    f"Row {row_num}: sell rows require an avg_buy_price column "
                    f"(the average cost of the shares being sold)."
                )
            try:
                avg_buy_price = float(avg_str)
            except ValueError:
                raise ValueError(f"Row {row_num}: invalid avg_buy_price {avg_str!r}.")
            if avg_buy_price < 0:
                raise ValueError(f"Row {row_num}: avg_buy_price cannot be negative.")

        transactions.append({
            "ticker": ticker,
            "market": market,
            "shares": shares,
            "price": price,
            "action": action,
            "date": tx_date,
            "avg_buy_price": avg_buy_price,
            "currency": "USD" if market == "US" else "TWD",
        })

    return transactions


def aggregate_transactions(transactions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Roll up a transaction log into current holdings + realized P&L per ticker.

    Net shares = buys − sells. Cost basis of remaining shares is the weighted
    average of buys minus the cost already removed by sells (sell_shares ×
    avg_buy_price). Tickers that net to zero are omitted from holdings but
    still appear in the realized P&L list.

    Returns (holdings_rows, realized_summary).
    """
    by_ticker: dict[str, dict] = {}
    for tx in sorted(transactions, key=lambda t: (t["ticker"], t["date"])):
        agg = by_ticker.setdefault(tx["ticker"], {
            "ticker": tx["ticker"],
            "market": tx["market"],
            "currency": tx["currency"],
            "bought_shares": 0.0,
            "buy_cost": 0.0,
            "sold_shares": 0.0,
            "sold_cost": 0.0,  # Σ sold_shares × avg_buy_price
            "proceeds": 0.0,   # Σ sold_shares × sale_price
        })
        if tx["action"] == "buy":
            agg["bought_shares"] += tx["shares"]
            agg["buy_cost"] += tx["shares"] * tx["price"]
        else:
            agg["sold_shares"] += tx["shares"]
            agg["sold_cost"] += tx["shares"] * (tx["avg_buy_price"] or 0.0)
            agg["proceeds"] += tx["shares"] * tx["price"]

    holdings: list[dict] = []
    realized: list[dict] = []
    for agg in by_ticker.values():
        net_shares = agg["bought_shares"] - agg["sold_shares"]
        if net_shares > 1e-9:
            remaining_cost = agg["buy_cost"] - agg["sold_cost"]
            cost_basis = remaining_cost / net_shares if net_shares > 0 else 0.0
            holdings.append({
                "ticker": agg["ticker"],
                "market": agg["market"],
                "shares": net_shares,
                "cost_basis": cost_basis,
                "currency": agg["currency"],
            })
        if agg["sold_shares"] > 0:
            realized.append({
                "ticker": agg["ticker"],
                "currency": agg["currency"],
                "shares_sold": agg["sold_shares"],
                "proceeds": agg["proceeds"],
                "cost": agg["sold_cost"],
                "realized_pl": agg["proceeds"] - agg["sold_cost"],
            })

    return holdings, realized


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
        yf = _get_yf()
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
    yf = _get_yf()
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
                last_d = datetime.strptime(last, "%Y-%m-%d").date()
                if last_d.isoformat() > today.isoformat():
                    print(f"[skip] FX {pair} — already current ({last})")
                    continue
                start = (last_d - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).isoformat()

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

    set_meta("last_fx_fetch", _taipei_now_str())
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
                last_d = datetime.strptime(last, "%Y-%m-%d").date()
                if last_d.isoformat() > today.isoformat():
                    print(f"[skip] {t} — already current ({last})")
                    continue
                start = (last_d - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).isoformat()

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

    set_meta("last_price_fetch", _taipei_now_str())
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
