"""Verify database prices against yfinance."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from db import init_db, load_holdings, load_prices


def verify_ticker(ticker: str) -> dict:
    """Compare DB prices with yfinance for a ticker."""
    # Fetch from yfinance
    yf_data = yf.download(ticker, period="max", progress=False, auto_adjust=False, actions=False)

    if yf_data is None or yf_data.empty:
        return {"ticker": ticker, "status": "error", "message": "yfinance returned no data"}

    if isinstance(yf_data.columns, pd.MultiIndex):
        yf_data.columns = yf_data.columns.get_level_values(0)

    yf_data = yf_data.reset_index()
    yf_data.columns = [str(c).lower() for c in yf_data.columns]
    yf_data["date"] = pd.to_datetime(yf_data["date"]).dt.strftime("%Y-%m-%d")
    yf_data = yf_data[["date", "close"]].dropna()
    yf_data = yf_data[yf_data["close"] > 0]
    yf_data = yf_data.rename(columns={"close": "yf_close"})

    # Load from DB
    db_data = load_prices(tickers=[ticker])
    db_data = db_data[["date", "close"]].dropna()
    db_data = db_data.rename(columns={"close": "db_close"})

    # Merge and compare
    merged = yf_data.merge(db_data, on="date", how="outer", indicator=True)

    yf_only = merged[merged["_merge"] == "left_only"]
    db_only = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"]

    # Check for differences in overlapping dates
    diffs = None
    if not both.empty:
        both["diff"] = abs(both["yf_close"] - both["db_close"])
        both["pct_diff"] = (both["diff"] / both["yf_close"] * 100).round(2)
        diffs = both[both["diff"] > 0.01]  # Tolerance for float precision

    return {
        "ticker": ticker,
        "yf_rows": len(yf_data),
        "db_rows": len(db_data),
        "common_rows": len(both),
        "yf_only": len(yf_only),
        "db_only": len(db_only),
        "differences": len(diffs) if diffs is not None else 0,
        "yf_date_range": f"{yf_data['date'].min()} to {yf_data['date'].max()}" if not yf_data.empty else "N/A",
        "db_date_range": f"{db_data['date'].min()} to {db_data['date'].max()}" if not db_data.empty else "N/A",
        "sample_diffs": diffs[["date", "yf_close", "db_close", "pct_diff"]].head().to_dict() if diffs is not None and not diffs.empty else None,
    }


def main() -> None:
    init_db()
    holdings = load_holdings()

    if holdings.empty:
        print("No holdings found")
        return

    tickers = holdings["ticker"].tolist()
    print(f"Verifying {len(tickers)} tickers...\n")

    issues = []
    for ticker in tickers:
        result = verify_ticker(ticker)
        print(f"{ticker:12} | YF: {result['yf_rows']:5} rows | DB: {result['db_rows']:5} rows | Diffs: {result['differences']}")

        if result["differences"] > 0 or result["yf_only"] > 0:
            issues.append(result)

        if result.get("sample_diffs"):
            print(f"             ↳ Sample differences: {result['sample_diffs']}")

    if issues:
        print(f"\n WARNING Found {len(issues)} ticker(s) with discrepancies:")
        for issue in issues:
            print(f"\n{issue['ticker']}:")
            print(f"  YF range: {issue['yf_date_range']}")
            print(f"  DB range: {issue['db_date_range']}")
            print(f"  Missing in DB: {issue['yf_only']} rows")
            print(f"  Price differences: {issue['differences']}")
    else:
        print("\n[OK] All prices match!")


if __name__ == "__main__":
    main()
