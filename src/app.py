"""Streamlit dashboard for the stock-price portfolio tracker."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from compute_totals import compute_daily_totals  # noqa: E402
from db import (  # noqa: E402
    init_db,
    load_daily_totals,
    load_fx_rates,
    load_holdings,
    load_prices,
    replace_daily_totals,
    replace_holdings,
)
from fetch_prices import (  # noqa: E402
    fetch_for_tickers,
    fetch_fx_rates,
    parse_holdings_yaml,
    sync_holdings_from_rows,
)

LOCAL_HOLDINGS_PATH = ROOT / "config" / "holdings.yaml"

st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="📈",
    layout="wide",
)


# Cache TTLs: prices/totals/fx cached for 6 hours since they only change after market close.
# Holdings cached briefly so uploads take effect quickly.
@st.cache_data(ttl=60)
def get_holdings() -> pd.DataFrame:
    return load_holdings()


@st.cache_data(ttl=21600)  # 6 hours
def get_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return load_prices(tickers=list(tickers), start=start, end=end)


@st.cache_data(ttl=21600)  # 6 hours
def get_daily_totals(start: str, end: str) -> pd.DataFrame:
    return load_daily_totals(start=start, end=end)


@st.cache_data(ttl=21600)  # 6 hours
def get_fx_rates(pair: str = "USDTWD") -> pd.DataFrame:
    return load_fx_rates(pair=pair)


TEMPLATE_YAML = (
    "holdings:\n"
    "  - ticker: VTI\n"
    "    market: US\n"
    "    shares: 10\n"
    "    cost_basis: 250.00\n"
    "    currency: USD\n"
    "  - ticker: 0050.TW\n"
    "    market: TW\n"
    "    shares: 1000\n"
    "    cost_basis: 100.00\n"
    "    currency: TWD\n"
)


def _ensure_holdings_loaded() -> bool:
    """Resolve holdings source: paste / upload → session → local file.
    Returns True if holdings are loaded into the DB.
    Mobile-friendly: file uploaders are flaky on iOS Safari, so a paste
    text-area is provided as the primary input.
    """
    with st.sidebar:
        st.subheader("Your holdings")

        rows = None
        source = None

        tab_paste, tab_upload = st.tabs(["📋 Paste", "📁 Upload"])

        with tab_paste:
            st.caption("Template (tap the copy icon → paste below → edit values):")
            st.code(TEMPLATE_YAML, language="yaml")

            text = st.text_area(
                "Paste your YAML here",
                height=220,
                help="Works on any device including iOS / Android browsers.",
                key="holdings_text_input",
            )

            col_apply, col_load = st.columns(2)
            with col_apply:
                apply_paste = st.button("Apply", use_container_width=True, type="primary")
            with col_load:
                # Use on_click so the callback runs BEFORE the next rerun renders
                # the textarea widget; setting session_state in the same run
                # would conflict with the already-rendered widget.
                st.button(
                    "Load template",
                    use_container_width=True,
                    on_click=lambda: st.session_state.update(holdings_text_input=TEMPLATE_YAML),
                )

            if apply_paste and text.strip():
                try:
                    rows = parse_holdings_yaml(text)
                    if not rows:
                        st.error("YAML parsed but no `holdings:` entries found.")
                        rows = None
                    else:
                        st.session_state["holdings_rows"] = rows
                        source = "pasted"
                except Exception as e:
                    st.error(f"Could not parse YAML: {e}")
                    return False

        with tab_upload:
            # Allow broader extensions; iOS often saves YAML as .txt or has no
            # MIME type for unknown extensions, so accepting anything text-y helps.
            uploaded = st.file_uploader(
                "Upload holdings.yaml / .yml / .txt",
                type=["yaml", "yml", "txt", "yamlk"],
                accept_multiple_files=False,
                help="If your phone won't show .yaml files, switch to the Paste tab.",
            )
            if rows is None and uploaded is not None:
                try:
                    content = uploaded.getvalue().decode("utf-8", errors="replace")
                    parsed = parse_holdings_yaml(content)
                    if not parsed:
                        st.error("File parsed but no `holdings:` entries found.")
                    else:
                        rows = parsed
                        st.session_state["holdings_rows"] = rows
                        st.session_state["holdings_text"] = content
                        source = "uploaded"
                except Exception as e:
                    st.error(f"Could not parse uploaded file: {e}")
                    return False

        # Fall back to existing session, then the DB (persists across F5
        # because Streamlit Cloud keeps the container's filesystem alive),
        # then the local dev YAML.
        if rows is None and "holdings_rows" in st.session_state:
            rows = st.session_state["holdings_rows"]
            source = "session (loaded earlier)"
        if rows is None:
            db_holdings = load_holdings()
            if not db_holdings.empty:
                rows = [
                    {
                        "ticker": r["ticker"],
                        "market": r["market"],
                        "shares": r["shares"],
                        "cost_basis": r["cost_basis"],
                        "currency": r["currency"],
                    }
                    for _, r in db_holdings.iterrows()
                ]
                # Cache back into session_state so subsequent checks short-circuit
                st.session_state["holdings_rows"] = rows
                source = "database (from previous visit)"
        if rows is None and LOCAL_HOLDINGS_PATH.exists():
            try:
                with open(LOCAL_HOLDINGS_PATH, "r", encoding="utf-8") as f:
                    rows = parse_holdings_yaml(f.read())
                source = "local config/holdings.yaml"
            except Exception as e:
                st.error(f"Could not read local holdings.yaml: {e}")
                return False

        if not rows:
            st.info("👆 Paste your YAML in the **Paste** tab and click **Apply**.")
            return False

        # Sync + recompute only if holdings actually changed (signature includes
        # ticker and shares so changes to either trigger a recompute).
        rows_sig = repr(sorted([(r.get("ticker"), r.get("shares")) for r in rows]))
        if st.session_state.get("holdings_sig") != rows_sig:
            sync_holdings_from_rows(rows)
            st.session_state["holdings_sig"] = rows_sig
            st.session_state["holdings_changed"] = True
            get_holdings.clear()

        st.caption(f"Source: {source} · {len(rows)} tickers")
        return True


def _ensure_prices_loaded(tickers: list[str]) -> None:
    """Make sure prices and daily_totals match the current holdings.

    1. Fetch yfinance data for any ticker that has no prices in the DB.
    2. Recompute daily_totals if (a) holdings changed since last run, or
       (b) any ticker was missing prices. This wipes stale rows for
       previously-held tickers and updates `shares` to current values.
    """
    needs_recompute = st.session_state.pop("holdings_changed", False)

    prices_df = load_prices(tickers=tickers)
    have_tickers = set(prices_df["ticker"].unique()) if not prices_df.empty else set()
    missing = [t for t in tickers if t not in have_tickers]

    if missing:
        needs_recompute = True
        with st.spinner(f"Fetching prices for {len(missing)} ticker(s) from Yahoo Finance..."):
            fetch_for_tickers(missing)
            fetch_fx_rates()

    # Also recompute if daily_totals has tickers not in current holdings
    # (left over from a previous holdings set).
    totals = load_daily_totals()
    if not totals.empty:
        stale_tickers = set(totals["ticker"].unique()) - set(tickers)
        if stale_tickers:
            needs_recompute = True

    if needs_recompute:
        with st.spinner("Recomputing daily totals..."):
            df = compute_daily_totals()
            replace_daily_totals(df)
            get_prices.clear()
            get_daily_totals.clear()
            get_fx_rates.clear()


def attach_twd_value(totals: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Add a `value_twd` column. Uses USDTWD rate (forward-filled across the date range).
    TWD rows pass through; USD rows are multiplied by that day's rate.
    Drops any rows with zero/NaN value (treat as market holiday).
    """
    if totals.empty:
        out = totals.copy()
        out["value_twd"] = []
        out["fx_rate"] = []
        return out

    out = totals.copy()
    out = out[out["value"].notna() & (out["value"] > 0)]

    if fx.empty:
        out = out.copy()
        out["fx_rate"] = float("nan")
        out["value_twd"] = out["value"].where(out["currency"] == "TWD")
        return out

    fx = fx[["date", "rate"]].drop_duplicates("date").sort_values("date")
    all_dates = pd.Index(sorted(set(out["date"].tolist()) | set(fx["date"].tolist())))
    fx_full = (
        fx.set_index("date")
        .reindex(all_dates)
        .ffill()
        .bfill()
        .reset_index()
        .rename(columns={"index": "date"})
    )

    out = out.merge(fx_full, on="date", how="left").rename(columns={"rate": "fx_rate"})
    out["value_twd"] = out.apply(
        lambda r: r["value"] if r["currency"] == "TWD" else r["value"] * r["fx_rate"],
        axis=1,
    )
    return out


def _ffill_pivot(totals_twd: pd.DataFrame) -> pd.DataFrame:
    """date × ticker pivot of value_twd, forward-filled across one-market holidays.
    Pre-IPO cells (before a ticker's first trade) are filled with 0 so they don't
    appear in the stack."""
    if totals_twd.empty:
        return pd.DataFrame()
    return (
        totals_twd.pivot_table(
            index="date", columns="ticker", values="value_twd", aggfunc="last"
        )
        .sort_index()
        .ffill()
        .fillna(0)
    )


def combined_twd_series(totals_twd: pd.DataFrame) -> pd.DataFrame:
    """Per-day combined portfolio value in TWD.
    Forward-fills each ticker's value so a one-market holiday (US closed but TW
    open, or vice versa) does not drop the combined line. Starts from the date
    every currently-held ticker has at least one observation."""
    pivot = _ffill_pivot(totals_twd)
    if pivot.empty:
        return pd.DataFrame(columns=["date", "value_twd"])
    first_valid = pivot.replace(0, pd.NA).apply(lambda s: s.first_valid_index()).max()
    if first_valid is not None:
        pivot = pivot.loc[first_valid:]
    out = pivot.sum(axis=1).reset_index()
    out.columns = ["date", "value_twd"]
    return out


def stacked_twd_long(totals_twd: pd.DataFrame) -> pd.DataFrame:
    """Long-form table for the per-ticker stacked area chart.
    Forward-fills each ticker's value across holidays so the stack is continuous;
    pre-IPO contributions stay at 0."""
    pivot = _ffill_pivot(totals_twd)
    if pivot.empty:
        return pd.DataFrame(columns=["date", "ticker", "value_twd"])
    pivot = pivot[pivot.sum(axis=1) > 0]
    return pivot.reset_index().melt(
        id_vars="date", var_name="ticker", value_name="value_twd"
    )


def refresh_all() -> str:
    holdings = load_holdings()
    if holdings.empty:
        return "No holdings to fetch."
    fetch_for_tickers(holdings["ticker"].tolist())
    fetch_fx_rates()
    df = compute_daily_totals()
    replace_daily_totals(df)
    st.cache_data.clear()
    return f"Refreshed {len(holdings)} holdings, {len(df)} total rows."


def _wipe_all_holdings_data() -> None:
    """Clear everything tied to the user's holdings: session state, DB tables,
    and the in-memory cache. Used by the explicit Clear button and by the
    fresh-page-load guard so refreshes leave no traces.
    """
    for key in (
        "holdings_rows",
        "holdings_sig",
        "holdings_text_input",
        "holdings_changed",
    ):
        st.session_state.pop(key, None)
    replace_holdings([])
    replace_daily_totals(pd.DataFrame())
    st.cache_data.clear()


def main() -> None:
    init_db()

    st.title("📈 Portfolio Tracker")
    st.caption("US + TW daily prices · holdings × close → daily portfolio value (TWD combined)")

    # Reserve a slot at the very top of the sidebar for the Controls block.
    # We populate it AFTER _ensure_holdings_loaded() runs so we know whether
    # holdings exist (after a fresh Apply click, the session state is only
    # updated mid-run inside that helper).
    with st.sidebar:
        controls_slot = st.empty()

    # Holdings input renders below the reserved slot.
    if not _ensure_holdings_loaded():
        return

    with st.sidebar:
        st.caption("Holdings stay in your browser session only.")

    today = date.today()
    default_start = today - timedelta(days=365)
    start_d, end_d = default_start, today

    # Now fill the reserved slot at the top — holdings exist at this point.
    with controls_slot.container():
        st.header("Controls")
        if st.button("🔄 Refresh prices", use_container_width=True):
            with st.spinner("Fetching from Yahoo Finance..."):
                msg = refresh_all()
            st.success(msg)

        if st.button("🗑️ Clear holdings", use_container_width=True, type="secondary"):
            _wipe_all_holdings_data()
            st.rerun()

        date_range = st.date_input(
            "Date range",
            value=(default_start, today),
            max_value=today,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range

        st.divider()

    holdings = get_holdings()
    if holdings.empty:
        st.warning("No holdings loaded. Upload your `holdings.yaml` from the sidebar.")
        return

    # Auto-fetch prices if the DB is empty (e.g. first run on Streamlit Cloud).
    _ensure_prices_loaded(holdings["ticker"].tolist())

    start_s = start_d.isoformat()
    end_s = end_d.isoformat()
    totals = get_daily_totals(start_s, end_s)
    prices = get_prices(tuple(holdings["ticker"].tolist()), start_s, end_s)
    fx = get_fx_rates("USDTWD")

    if totals.empty:
        st.info("No price data yet. Click **🔄 Refresh prices** in the sidebar to fetch.")
        return

    totals_twd = attach_twd_value(totals, fx)

    def _pivot_ffill(col: str) -> pd.DataFrame:
        return (
            totals_twd.pivot_table(
                index="date", columns="ticker", values=col, aggfunc="last"
            )
            .sort_index()
            .ffill()
        )

    twd_pivot = _pivot_ffill("value_twd")
    native_pivot = _pivot_ffill("value")

    market_map = dict(zip(holdings["ticker"], holdings["market"]))
    tw_tickers = [t for t, m in market_map.items() if m == "TW" and t in twd_pivot.columns]
    us_tickers = [t for t, m in market_map.items() if m == "US" and t in twd_pivot.columns]

    def _last_two_with_data(cols: list[str]) -> tuple[str | None, str | None]:
        if not cols:
            return None, None
        sub = totals_twd[totals_twd["ticker"].isin(cols)]
        dates = sorted(sub["date"].unique())
        return (
            dates[-1] if dates else None,
            dates[-2] if len(dates) >= 2 else None,
        )

    tw_today_date, tw_prev_date = _last_two_with_data(tw_tickers)
    us_today_date, us_prev_date = _last_two_with_data(us_tickers)

    def _sum_at(pivot: pd.DataFrame, cols: list[str], date_str: str | None) -> float:
        if not cols or not date_str or date_str not in pivot.index:
            return 0.0
        return float(pivot.loc[date_str, cols].fillna(0).sum())

    tw_today_twd = _sum_at(twd_pivot, tw_tickers, tw_today_date)
    tw_prev_twd = _sum_at(twd_pivot, tw_tickers, tw_prev_date)
    us_today_twd = _sum_at(twd_pivot, us_tickers, us_today_date)
    us_prev_twd = _sum_at(twd_pivot, us_tickers, us_prev_date)
    us_today_usd = _sum_at(native_pivot, us_tickers, us_today_date)
    us_prev_usd = _sum_at(native_pivot, us_tickers, us_prev_date)

    combined_today = tw_today_twd + us_today_twd
    combined_prev = tw_prev_twd + us_prev_twd if (tw_prev_date or us_prev_date) else None
    combined_delta = (combined_today - combined_prev) if combined_prev else None
    combined_delta_pct = (combined_delta / combined_prev * 100.0) if combined_prev else None

    fx_today = None
    fx_prev = None
    if us_today_date and not fx.empty:
        fx_on = fx[fx["date"] <= us_today_date]
        if not fx_on.empty:
            fx_today = float(fx_on.iloc[-1]["rate"])
            if len(fx_on) >= 2:
                fx_prev = float(fx_on.iloc[-2]["rate"])

    st.subheader("Summary")
    if fx_today:
        if fx_prev:
            fx_delta = fx_today - fx_prev
            fx_pct = fx_delta / fx_prev * 100.0
            color = "#16a34a" if fx_delta >= 0 else "#dc2626"
            st.markdown(
                f"<small>USD→TWD rate used: <b>{fx_today:,.4f}</b>  "
                f"<span style='color:{color}'>"
                f"({fx_delta:+.4f}, {fx_pct:+.2f}%)</span></small>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"USD→TWD rate used: **{fx_today:,.4f}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Combined Total (TWD)",
            value=f"NT$ {combined_today:,.0f}",
            delta=(
                f"{combined_delta:+,.0f}  ({combined_delta_pct:+.2f}%)"
                if combined_delta is not None and combined_prev
                else None
            ),
        )
        as_of_combined = max(d for d in [tw_today_date, us_today_date] if d) if (tw_today_date or us_today_date) else "—"
        st.caption(f"as of {as_of_combined}")
    with c2:
        tw_gain_twd = tw_today_twd - tw_prev_twd if tw_prev_twd else 0.0
        tw_gain_pct = (tw_gain_twd / tw_prev_twd * 100.0) if tw_prev_twd else None
        st.metric(
            label="TW stocks (TWD)",
            value=f"NT$ {tw_today_twd:,.0f}",
            delta=(
                f"{tw_gain_twd:+,.0f}  ({tw_gain_pct:+.2f}%)"
                if tw_gain_pct is not None
                else None
            ),
        )
        st.caption(f"as of {tw_today_date or '—'}")
    with c3:
        us_gain_usd = us_today_usd - us_prev_usd if us_prev_usd else 0.0
        us_gain_twd = us_today_twd - us_prev_twd if us_prev_twd else 0.0
        us_gain_pct = (us_gain_usd / us_prev_usd * 100.0) if us_prev_usd else None
        us_gain_twd_pct = (us_gain_twd / us_prev_twd * 100.0) if us_prev_twd else None
        st.metric(
            label="US stocks (USD)",
            value=f"US$ {us_today_usd:,.2f}",
            delta=(
                f"{us_gain_usd:+,.2f}  ({us_gain_pct:+.2f}%)"
                if us_gain_pct is not None
                else None
            ),
        )
        st.metric(
            label="US stocks (TWD equiv.)",
            value=f"NT$ {us_today_twd:,.0f}",
            delta=(
                f"{us_gain_twd:+,.0f}  ({us_gain_twd_pct:+.2f}%)"
                if us_gain_twd_pct is not None
                else None
            ),
        )
        st.caption(f"as of {us_today_date or '—'}")

    tab_holdings, tab_history, tab_portfolio = st.tabs(
        ["Holdings", "Price history", "Daily portfolio total"]
    )

    with tab_holdings:
        sorted_t = totals_twd.sort_values(["ticker", "date"]).copy()
        sorted_t["prev_close"] = sorted_t.groupby("ticker")["close_price"].shift(1)
        last_prices = (
            sorted_t.groupby("ticker", as_index=False)
            .tail(1)[["ticker", "date", "close_price", "prev_close", "fx_rate"]]
            .rename(columns={"date": "as_of", "close_price": "last_close"})
        )
        view = holdings.merge(last_prices, on="ticker", how="left")
        view["market_value"] = view["last_close"] * view["shares"]
        view["cost_value"] = view["cost_basis"] * view["shares"]
        view["unrealized_pl"] = view["market_value"] - view["cost_value"]
        view["unrealized_pl_pct"] = (
            (view["last_close"] / view["cost_basis"] - 1.0) * 100.0
        ).where(view["cost_basis"].notna() & (view["cost_basis"] != 0))
        view["daily_gain"] = (view["last_close"] - view["prev_close"]) * view["shares"]
        view["daily_gain_pct"] = (
            (view["last_close"] / view["prev_close"] - 1.0) * 100.0
        ).where(view["prev_close"].notna() & (view["prev_close"] != 0))
        view["daily_gain_twd"] = view.apply(
            lambda r: r["daily_gain"] if r["currency"] == "TWD"
            else (r["daily_gain"] * r["fx_rate"]) if pd.notna(r["fx_rate"]) else None,
            axis=1,
        )
        view["market_value_twd"] = view.apply(
            lambda r: r["market_value"] if r["currency"] == "TWD" else r["market_value"] * (r["fx_rate"] or 0),
            axis=1,
        )
        # Build a Styler so gain/loss columns render green/red.
        # NumberColumn formats don't support per-cell color, but Styler
        # via st.dataframe does — at the cost of losing column_config
        # formats, so we apply formats inside the Styler instead.
        gain_cols = [
            "daily_gain", "daily_gain_pct", "daily_gain_twd",
            "unrealized_pl", "unrealized_pl_pct",
        ]

        def _color_sign(v):
            if pd.isna(v):
                return ""
            return "color:#16a34a" if v >= 0 else "color:#dc2626"

        view_table = view[
            [
                "ticker", "market", "currency", "shares",
                "cost_basis", "last_close", "as_of",
                "daily_gain", "daily_gain_pct", "daily_gain_twd",
                "market_value", "market_value_twd",
                "cost_value", "unrealized_pl", "unrealized_pl_pct",
            ]
        ].rename(
            columns={
                "daily_gain": "daily_gain (native)",
                "daily_gain_pct": "daily_gain %",
                "daily_gain_twd": "daily_gain (TWD)",
                "market_value": "market_value (native)",
                "market_value_twd": "market_value (TWD)",
            }
        )

        styled = view_table.style.format({
            "shares": "{:,.2f}",
            "cost_basis": "{:,.4f}",
            "last_close": "{:,.2f}",
            "daily_gain (native)": "{:+,.2f}",
            "daily_gain %": "{:+.2f}%",
            "daily_gain (TWD)": "{:+,.0f}",
            "market_value (native)": "{:,.2f}",
            "market_value (TWD)": "{:,.0f}",
            "cost_value": "{:,.2f}",
            "unrealized_pl": "{:+,.2f}",
            "unrealized_pl_pct": "{:+.2f}%",
        }, na_rep="—").map(
            _color_sign,
            subset=[
                "daily_gain (native)", "daily_gain %", "daily_gain (TWD)",
                "unrealized_pl", "unrealized_pl_pct",
            ],
        )

        st.dataframe(styled, use_container_width=True, hide_index=True)

        pie_df = view[view["market_value_twd"].notna() & (view["market_value_twd"] > 0)].copy()
        if not pie_df.empty:
            pie_df = pie_df.sort_values("market_value_twd", ascending=False)
            total_twd = pie_df["market_value_twd"].sum()
            col_a, col_b = st.columns([1, 1])
            with col_a:
                fig_pie = px.pie(
                    pie_df,
                    names="ticker",
                    values="market_value_twd",
                    title=f"Holdings allocation by ticker (TWD)  ·  Total NT$ {total_twd:,.0f}",
                    hole=0.45,
                )
                fig_pie.update_traces(
                    textposition="outside",
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>NT$ %{value:,.0f}<br>%{percent}<extra></extra>",
                )
                fig_pie.update_layout(height=420, showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_b:
                market_pie = (
                    pie_df.groupby("market", as_index=False)["market_value_twd"].sum()
                )
                fig_pie_mkt = px.pie(
                    market_pie,
                    names="market",
                    values="market_value_twd",
                    title="Holdings allocation by market (TWD)",
                    hole=0.45,
                    color="market",
                    color_discrete_map={"TW": "#1f77b4", "US": "#d62728"},
                )
                fig_pie_mkt.update_traces(
                    textposition="outside",
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>NT$ %{value:,.0f}<br>%{percent}<extra></extra>",
                )
                fig_pie_mkt.update_layout(height=420, showlegend=True)
                st.plotly_chart(fig_pie_mkt, use_container_width=True)

    with tab_history:
        if prices.empty:
            st.info("No price history in this date range.")
        else:
            chosen = st.multiselect(
                "Tickers",
                options=sorted(prices["ticker"].unique()),
                default=sorted(prices["ticker"].unique()),
            )
            sub = prices[prices["ticker"].isin(chosen)]
            sub = sub[sub["close"].notna() & (sub["close"] > 0)]
            if sub.empty:
                st.info("Pick at least one ticker.")
            else:
                fig = px.line(
                    sub,
                    x="date",
                    y="close",
                    color="ticker",
                    title="Daily close price (native currency)",
                )
                fig.update_layout(hovermode="x unified", height=500)
                fig.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig, use_container_width=True)

    with tab_portfolio:
        ffill_twd = _ffill_pivot(totals_twd)
        combined = combined_twd_series(totals_twd)
        if not combined.empty:
            st.subheader("Portfolio performance")
            combined_sorted = combined.sort_values("date").reset_index(drop=True)
            combined_sorted["date_dt"] = pd.to_datetime(combined_sorted["date"])
            latest_dt = combined_sorted["date_dt"].max()
            latest_val = float(combined_sorted.iloc[-1]["value_twd"])

            periods = [
                ("1D", pd.Timedelta(days=1)),
                ("1W", pd.Timedelta(days=7)),
                ("1M", pd.DateOffset(months=1)),
                ("3M", pd.DateOffset(months=3)),
                ("6M", pd.DateOffset(months=6)),
                ("9M", pd.DateOffset(months=9)),
                ("1Y", pd.DateOffset(years=1)),
            ]

            perf_cols = st.columns(len(periods))
            for col, (label, offset) in zip(perf_cols, periods):
                target = latest_dt - offset
                past = combined_sorted[combined_sorted["date_dt"] <= target]
                with col:
                    if past.empty:
                        st.metric(label=label, value="N/A")
                    else:
                        past_val = float(past.iloc[-1]["value_twd"])
                        delta = latest_val - past_val
                        delta_pct = (delta / past_val * 100.0) if past_val > 0 else 0.0
                        st.metric(
                            label=label,
                            value=f"{delta:+,.0f}",
                            delta=f"{delta_pct:+.2f}%",
                        )

            st.divider()

            fig0 = px.line(
                combined,
                x="date",
                y="value_twd",
                title="Daily combined portfolio value (TWD equivalent)",
            )
            fig0.update_layout(hovermode="x unified", height=380, yaxis_title="Value (TWD)")
            fig0.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig0, use_container_width=True)

            # Shared controls row: range selector (left) + view toggle (right).
            # The same range filters both Total and By Stock views so the user
            # only chooses the window once.
            col_range, col_view = st.columns([2, 1])
            with col_range:
                st.caption("Range")
                range_choice = st.radio(
                    "Range",
                    options=["1W", "1M", "3M", "6M", "1Y", "All"],
                    index=1,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="gain_range",
                )
            with col_view:
                st.caption("View")
                gain_view = st.radio(
                    "Gain view",
                    ["Total", "By Stock"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="gain_view",
                )

            def _cutoff(max_date, choice):
                if choice == "1W":
                    return max_date - pd.Timedelta(days=7)
                if choice == "1M":
                    return max_date - pd.DateOffset(months=1)
                if choice == "3M":
                    return max_date - pd.DateOffset(months=3)
                if choice == "6M":
                    return max_date - pd.DateOffset(months=6)
                if choice == "1Y":
                    return max_date - pd.DateOffset(years=1)
                return None  # "All"

            if gain_view == "Total":
                gain = combined.copy()
                gain["gain_twd"] = gain["value_twd"].diff()
                gain["gain_pct"] = gain["value_twd"].pct_change() * 100.0
                gain = gain.dropna(subset=["gain_twd"])
                if not gain.empty:
                    gain["date"] = pd.to_datetime(gain["date"])

                    cutoff = _cutoff(gain["date"].max(), range_choice)
                    gain_filtered = gain if cutoff is None else gain[gain["date"] >= cutoff]

                    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in gain_filtered["gain_twd"]]
                    fig_gain = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_gain.add_trace(
                        go.Bar(
                            x=gain_filtered["date"],
                            y=gain_filtered["gain_twd"],
                            marker_color=colors,
                            name="Gain (TWD)",
                            hovertemplate="%{x|%Y-%m-%d}<br>Gain: %{y:,.0f} TWD<extra></extra>",
                        ),
                        secondary_y=False,
                    )
                    fig_gain.add_trace(
                        go.Scatter(
                            x=gain_filtered["date"],
                            y=gain_filtered["gain_pct"],
                            mode="lines",
                            name="Gain %",
                            line=dict(color="#1f77b4", width=1.5),
                            hovertemplate="%{x|%Y-%m-%d}<br>Gain: %{y:+.2f}%<extra></extra>",
                        ),
                        secondary_y=True,
                    )
                    fig_gain.update_layout(
                        title=f"Daily portfolio gain — {range_choice} (TWD and %)",
                        hovermode="x unified",
                        height=380,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    # autorange=True ensures both y-axes auto-fit to the
                    # filtered data (no leftover range from previous window).
                    fig_gain.update_yaxes(title_text="Gain (TWD)", secondary_y=False, autorange=True)
                    fig_gain.update_yaxes(title_text="Gain (%)", secondary_y=True, autorange=True)
                    st.plotly_chart(fig_gain, use_container_width=True, config={"displaylogo": False})
            else:
                stock_gains = []
                for ticker in ffill_twd.columns:
                    ticker_data = ffill_twd[[ticker]].copy()
                    ticker_data.columns = ["value_twd"]
                    ticker_data["ticker"] = ticker
                    ticker_data["date"] = ticker_data.index
                    ticker_data["gain_twd"] = ticker_data["value_twd"].diff()
                    ticker_data["gain_pct"] = ticker_data["value_twd"].pct_change() * 100.0
                    ticker_data = ticker_data.dropna(subset=["gain_twd"])
                    stock_gains.append(ticker_data)

                if stock_gains:
                    stock_gain_df = pd.concat(stock_gains, ignore_index=True)
                    stock_gain_df["date"] = pd.to_datetime(stock_gain_df["date"])

                    # Reuse the shared range_choice from above
                    cutoff = _cutoff(stock_gain_df["date"].max(), range_choice)
                    filtered = (
                        stock_gain_df
                        if cutoff is None
                        else stock_gain_df[stock_gain_df["date"] >= cutoff]
                    ).copy()

                    def _fmt_gain(v: float) -> str:
                        color = "#16a34a" if v >= 0 else "#dc2626"
                        return f"<span style='color:{color}'>{v:+,.0f}</span>"

                    palette = px.colors.qualitative.Plotly
                    tickers_sorted = sorted(filtered["ticker"].unique())
                    color_map = {t: palette[i % len(palette)] for i, t in enumerate(tickers_sorted)}

                    fig_stock_gain = go.Figure()
                    for ticker in tickers_sorted:
                        tdf = filtered[filtered["ticker"] == ticker].sort_values("date")
                        # Build customdata as a plain list of single-element
                        # lists. .values can be a PyArrow-backed pandas array
                        # on newer stacks (Streamlit Cloud / Python 3.14),
                        # which doesn't support .reshape(); .tolist() avoids
                        # the dtype-specific path entirely.
                        formatted = tdf["gain_twd"].apply(_fmt_gain).tolist()
                        fig_stock_gain.add_trace(
                            go.Bar(
                                x=tdf["date"],
                                y=tdf["gain_twd"],
                                name=ticker,
                                marker_color=color_map[ticker],
                                customdata=[[v] for v in formatted],
                                hovertemplate=f"{ticker}: %{{customdata[0]}}<extra></extra>",
                            )
                        )

                    daily_total = filtered.groupby("date")["gain_twd"].sum().reset_index()
                    daily_total["formatted"] = daily_total["gain_twd"].apply(_fmt_gain)
                    fig_stock_gain.add_trace(
                        go.Scatter(
                            x=daily_total["date"],
                            y=[0] * len(daily_total),
                            mode="markers",
                            marker=dict(opacity=0, size=0.1),
                            showlegend=False,
                            customdata=[[v] for v in daily_total["formatted"].tolist()],
                            hovertemplate="<b>Total: %{customdata[0]}</b><extra></extra>",
                            hoverinfo="text",
                        )
                    )

                    fig_stock_gain.update_layout(
                        title=f"Daily gain by stock — {range_choice} (TWD)",
                        hovermode="x unified",
                        height=760,
                        yaxis_title="Gain (TWD)",
                        barmode="group",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    fig_stock_gain.update_yaxes(autorange=True)
                    st.plotly_chart(
                        fig_stock_gain,
                        use_container_width=True,
                        config={"displaylogo": False},
                    )

        market_map = dict(zip(holdings["ticker"], holdings["market"]))
        rows = []
        for mkt in sorted(set(market_map.values())):
            cols = [t for t, m in market_map.items() if m == mkt and t in ffill_twd.columns]
            if not cols:
                continue
            sub = ffill_twd[cols].sum(axis=1)
            for d, v in sub.items():
                if v > 0:
                    rows.append({"date": d, "market": mkt, "value_twd": v})
        per_day_market = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()
        if not per_day_market.empty:
            fig1 = px.line(
                per_day_market,
                x="date",
                y="value_twd",
                color="market",
                title="Daily portfolio total — TW vs US (both in TWD)",
            )
            fig1.update_layout(hovermode="x unified", height=380, yaxis_title="Value (TWD)")
            fig1.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig1, use_container_width=True)

        stacked = stacked_twd_long(totals_twd)
        if not stacked.empty:
            fig2 = px.area(
                stacked,
                x="date",
                y="value_twd",
                color="ticker",
                line_group="ticker",
                title="Daily value contribution by ticker (stacked, TWD)",
            )
            fig2.update_layout(hovermode="x unified", height=500)
            fig2.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
