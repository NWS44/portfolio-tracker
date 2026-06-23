"""Streamlit dashboard for the stock-price portfolio tracker."""
from __future__ import annotations

import copy
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_local_storage import LocalStorage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from compute_totals import compute_daily_totals  # noqa: E402
from db import (  # noqa: E402
    get_meta,
    init_db,
    load_fx_rates,
    load_prices,
)
from fetch_prices import (  # noqa: E402
    fetch_for_tickers,
    fetch_fx_rates,
    parse_holdings_yaml,
)

LOCAL_HOLDINGS_PATH = ROOT / "config" / "holdings.yaml"

st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="🕹️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Retro arcade theme: pixel fonts (Press Start 2P for headings/buttons, VT323
# for body text and numbers), neon green/pink palette, chunky "8-bit" borders
# with hard drop-shadows, and a faint CRT scanline overlay.
# ---------------------------------------------------------------------------
NEON_GREEN = "#00ff9d"
NEON_RED = "#ff2e63"

GAME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

/* Explicit background so switching themes swaps the page color too. */
html, body, .stApp, [data-testid="stAppViewContainer"] { background: #0d0221; }
[data-testid="stHeader"] { background: #0d0221; }

/* Body text: VT323 is a readable pixel font, so bump the size up. */
html, body, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] p,
.stTextArea textarea, .stDataFrame, .stMetric {
    font-family: 'VT323', monospace !important;
    font-size: 1.15rem;
}

/* Headings, buttons, tabs: chunky arcade font. */
h1, h2, h3 {
    font-family: 'Press Start 2P', monospace !important;
    color: #00ff9d !important;
    text-shadow: 0 0 8px rgba(0, 255, 157, 0.7), 3px 3px 0 #ff2e63;
}
h1 { font-size: 1.7rem !important; line-height: 1.5 !important; }
h2 { font-size: 1.1rem !important; }
h3 { font-size: 0.9rem !important; }

/* Metric cards: 8-bit panels with hard pixel shadows. */
[data-testid="stMetric"] {
    background: #16163a;
    border: 3px solid #00ff9d;
    border-radius: 0;
    box-shadow: 5px 5px 0 #ff2e63;
    padding: 14px 16px;
    margin-bottom: 10px;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.62rem !important;
    color: #9d9dff !important;
}
[data-testid="stMetricValue"] {
    font-family: 'VT323', monospace !important;
    font-size: 2.6rem !important;
    color: #fffb96 !important;
    text-shadow: 0 0 10px rgba(255, 251, 150, 0.5);
}
[data-testid="stMetricDelta"] {
    font-family: 'VT323', monospace !important;
    font-size: 1.3rem !important;
}

/* Buttons: arcade cabinet buttons that "press down" on click. */
.stButton > button, .stDownloadButton > button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.62rem !important;
    word-break: keep-all;
    overflow-wrap: normal;
    line-height: 1.7;
    border: 3px solid #00ff9d !important;
    border-radius: 0 !important;
    background: #16163a !important;
    color: #00ff9d !important;
    box-shadow: 4px 4px 0 #ff2e63;
    transition: none;
}
.stButton > button p, .stDownloadButton > button p {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.62rem !important;
    word-break: keep-all !important;
    overflow-wrap: normal !important;
}
.stButton > button:hover {
    background: #00ff9d !important;
    color: #0d0221 !important;
}
.stButton > button:active {
    transform: translate(4px, 4px);
    box-shadow: none;
}
.stButton > button[kind="primary"] {
    border-color: #ff2e63 !important;
    color: #ff2e63 !important;
    box-shadow: 4px 4px 0 #00ff9d;
}
.stButton > button[kind="primary"]:hover {
    background: #ff2e63 !important;
    color: #0d0221 !important;
}

/* Tabs: stage-select bar. */
.stTabs [data-baseweb="tab"] {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 0.6rem !important;
    color: #9d9dff;
}
.stTabs [aria-selected="true"] {
    color: #00ff9d !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: #00ff9d; }

/* Sidebar: darker panel with a neon edge. */
[data-testid="stSidebar"] {
    background: #0a0118;
    border-right: 3px solid #00ff9d;
}

/* Toggle/radio labels readable in pixel style. */
.stRadio label, .stToggle label, .stMultiSelect label {
    font-family: 'VT323', monospace !important;
}

/* Faint CRT scanlines over the whole app. */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        rgba(0, 0, 0, 0.12) 0px,
        rgba(0, 0, 0, 0.12) 1px,
        transparent 1px,
        transparent 3px
    );
    z-index: 9999;
}
</style>
"""

NEON_COLORWAY = [
    "#00ff9d", "#ff2e63", "#fffb96", "#08f7fe", "#f5a623",
    "#bd93f9", "#ff79c6", "#50fa7b", "#ffb86c", "#8be9fd",
]

# ---------------------------------------------------------------------------
# Stock Exchange theme: professional trading-terminal look. Near-black panels,
# amber accents, Inter for labels, IBM Plex Mono for numbers, TradingView-style
# teal/red for gains/losses. No shadows, thin borders, uppercase labels.
# ---------------------------------------------------------------------------
AMBER = "#ffb000"
EXCH_GREEN = "#26a69a"
EXCH_RED = "#ef5350"
EXCH_COLORWAY = [
    "#ffb000", "#4f8ff7", "#26a69a", "#ef5350", "#ab7df6",
    "#00bcd4", "#ff8f40", "#9ccc65", "#ec407a", "#78909c",
]

EXCHANGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] { background: #0b0e14; }
[data-testid="stHeader"] { background: #0b0e14; }
[data-testid="stSidebar"] {
    background: #0e1320;
    border-right: 1px solid #232b38;
}

html, body, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] p {
    font-family: 'Inter', sans-serif !important;
}
.stDataFrame, .stTextArea textarea, code, pre {
    font-family: 'IBM Plex Mono', monospace !important;
}

h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: #e8eaed !important;
    letter-spacing: 0.01em;
    text-shadow: none;
}
h1 {
    border-left: 8px solid #ffb000;
    padding-left: 14px !important;
    font-size: 2rem !important;
}
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1.05rem !important; }

/* Metric cards: flat terminal panels with an amber index bar. */
[data-testid="stMetric"] {
    background: #11161f;
    border: 1px solid #232b38;
    border-left: 4px solid #ffb000;
    border-radius: 4px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b95a5 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 600 !important;
    color: #e8eaed !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.95rem !important;
}

.stButton > button, .stDownloadButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    background: #11161f !important;
    color: #ffb000 !important;
    border: 1px solid #ffb000 !important;
    border-radius: 3px !important;
    box-shadow: none;
}
.stButton > button:hover {
    background: #ffb000 !important;
    color: #0b0e14 !important;
}
.stButton > button[kind="primary"] {
    background: #ffb000 !important;
    color: #0b0e14 !important;
}
.stButton > button[kind="primary"]:hover { background: #ffc94d !important; }

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8b95a5;
}
.stTabs [aria-selected="true"] { color: #ffb000 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #ffb000; }

/* Inputs pick up the panel color instead of the arcade purple. */
.stTextArea textarea, [data-baseweb="select"] > div, [data-testid="stDateInput"] input {
    background-color: #11161f !important;
    border-color: #232b38 !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Solarized Light theme: easy-on-the-eyes paper look. Cream background, olive
# text, blue/orange accents. Lora serif for headings, Source Sans for body,
# JetBrains Mono for numbers. Light theme — needs explicit dark text overrides.
# ---------------------------------------------------------------------------
SOL_BG = "#fdf6e3"
SOL_PANEL = "#eee8d5"
SOL_INK = "#586e75"
SOL_HEAD = "#073642"
SOL_GREEN = "#859900"
SOL_RED = "#dc322f"
SOL_BLUE = "#268bd2"
SOL_COLORWAY = [
    "#268bd2", "#cb4b16", "#859900", "#d33682", "#6c71c4",
    "#2aa198", "#b58900", "#dc322f", "#586e75", "#93a1a1",
]

SOLARIZED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] { background: #fdf6e3; color: #586e75; }
[data-testid="stHeader"] { background: #fdf6e3; }
[data-testid="stSidebar"] {
    background: #eee8d5;
    border-right: 1px solid #93a1a1;
}

html, body, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] p,
.stRadio label, .stToggle label, .stMultiSelect label {
    font-family: 'Source Sans 3', sans-serif !important;
    color: #586e75 !important;
}
.stDataFrame, .stTextArea textarea, code, pre, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
}

h1, h2, h3 {
    font-family: 'Lora', serif !important;
    color: #073642 !important;
    text-shadow: none;
    letter-spacing: 0;
}
h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }

[data-testid="stMetric"] {
    background: #eee8d5;
    border: 1px solid #93a1a1;
    border-top: 3px solid #268bd2;
    border-radius: 2px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: none;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #93a1a1 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    color: #073642 !important;
}

.stButton > button, .stDownloadButton > button {
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    background: #fdf6e3 !important;
    color: #268bd2 !important;
    border: 1px solid #268bd2 !important;
    border-radius: 2px !important;
    box-shadow: none;
}
.stButton > button:hover {
    background: #268bd2 !important;
    color: #fdf6e3 !important;
}
.stButton > button[kind="primary"] {
    background: #cb4b16 !important;
    color: #fdf6e3 !important;
    border-color: #cb4b16 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Lora', serif !important;
    font-size: 0.95rem !important;
    color: #93a1a1;
}
.stTabs [aria-selected="true"] { color: #073642 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #268bd2; }

.stTextArea textarea, [data-baseweb="select"] > div, [data-testid="stDateInput"] input {
    background-color: #fdf6e3 !important;
    border-color: #93a1a1 !important;
    color: #586e75 !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Cyberpunk theme: deep violet-black, magenta + cyan neon, Orbitron headings,
# heavy glow. Sleeker / sharper than the pixel-art Retro Arcade.
# ---------------------------------------------------------------------------
CP_BG = "#0a0014"
CP_PANEL = "#1a0033"
CP_MAGENTA = "#ff00ff"
CP_CYAN = "#00ffff"
CP_UP = "#39ff14"
CP_DOWN = "#ff0055"
CP_COLORWAY = [
    "#ff00ff", "#00ffff", "#39ff14", "#ff0055", "#fffb00",
    "#ff8c00", "#bd00ff", "#00ff7f", "#ff1493", "#1e90ff",
]

CYBERPUNK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top, #1a0033 0%, #0a0014 70%);
}
[data-testid="stHeader"] { background: #0a0014; }
[data-testid="stSidebar"] {
    background: #100020;
    border-right: 1px solid #ff00ff;
    box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
}

html, body, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] p,
.stRadio label, .stToggle label {
    font-family: 'Rajdhani', sans-serif !important;
    color: #e0d4ff !important;
    font-size: 1.02rem;
}
.stDataFrame, .stTextArea textarea, code, pre, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: #00ffff !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-shadow: 0 0 6px rgba(0, 255, 255, 0.8), 0 0 14px rgba(255, 0, 255, 0.5);
}
h1 { font-size: 1.9rem !important; font-weight: 900 !important; }
h2 { font-size: 1.2rem !important; font-weight: 700 !important; }
h3 { font-size: 1rem !important; font-weight: 700 !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(26,0,51,0.9), rgba(10,0,20,0.9));
    border: 1px solid #ff00ff;
    border-radius: 0;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 0 10px rgba(255, 0, 255, 0.4), inset 0 0 20px rgba(0, 255, 255, 0.05);
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
}
[data-testid="stMetricLabel"] p {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #ff00ff !important;
    text-shadow: 0 0 4px rgba(255, 0, 255, 0.6);
}
[data-testid="stMetricValue"] {
    font-size: 2.1rem !important;
    color: #00ffff !important;
    text-shadow: 0 0 8px rgba(0, 255, 255, 0.6);
}

.stButton > button, .stDownloadButton > button {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: transparent !important;
    color: #00ffff !important;
    border: 1px solid #00ffff !important;
    border-radius: 0 !important;
    box-shadow: 0 0 8px rgba(0, 255, 255, 0.5), inset 0 0 8px rgba(0, 255, 255, 0.1);
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #00ffff !important;
    color: #0a0014 !important;
    box-shadow: 0 0 16px #00ffff;
}
.stButton > button[kind="primary"] {
    color: #ff00ff !important;
    border-color: #ff00ff !important;
    box-shadow: 0 0 8px rgba(255, 0, 255, 0.6);
}
.stButton > button[kind="primary"]:hover {
    background: #ff00ff !important;
    color: #0a0014 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b7fb5;
}
.stTabs [aria-selected="true"] {
    color: #ff00ff !important;
    text-shadow: 0 0 6px rgba(255, 0, 255, 0.7);
}
.stTabs [data-baseweb="tab-highlight"] { background-color: #ff00ff; }

.stTextArea textarea, [data-baseweb="select"] > div, [data-testid="stDateInput"] input {
    background-color: #100020 !important;
    border-color: #ff00ff !important;
    color: #e0d4ff !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Broadsheet theme: vintage financial newspaper. Cream paper, deep red accent,
# Playfair Display serif headings, double-rule borders. Classic WSJ broadsheet.
# ---------------------------------------------------------------------------
BS_BG = "#f5f1e8"
BS_PANEL = "#ebe5d3"
BS_INK = "#1a1a1a"
BS_ACCENT = "#8b0000"
BS_GREEN = "#1f6b1f"
BS_RED = "#8b0000"
BS_COLORWAY = [
    "#8b0000", "#1f6b1f", "#1c4587", "#a85d2c", "#5d4e75",
    "#7a5230", "#2f4f4f", "#8b3a62", "#556b2f", "#4b3621",
]

BROADSHEET_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;900&family=PT+Serif:wght@400;700&family=Old+Standard+TT:wght@400;700&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] { background: #f5f1e8; color: #1a1a1a; }
[data-testid="stHeader"] { background: #f5f1e8; }
[data-testid="stSidebar"] {
    background: #ebe5d3;
    border-right: 2px double #1a1a1a;
}

html, body, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] p,
.stRadio label, .stToggle label, .stMultiSelect label {
    font-family: 'PT Serif', serif !important;
    color: #1a1a1a !important;
}
.stDataFrame, .stTextArea textarea, code, pre {
    font-family: 'PT Serif', serif !important;
    color: #1a1a1a !important;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #1a1a1a !important;
    text-shadow: none;
    letter-spacing: -0.01em;
}
h1 {
    font-size: 2.6rem !important;
    font-weight: 900 !important;
    border-bottom: 3px double #1a1a1a;
    padding-bottom: 6px;
    font-style: italic;
}
h2 { font-size: 1.4rem !important; font-weight: 700 !important; }
h3 { font-size: 1.1rem !important; font-weight: 700 !important; font-style: italic; }

[data-testid="stMetric"] {
    background: #ebe5d3;
    border: 1px solid #1a1a1a;
    border-top: 3px double #1a1a1a;
    border-bottom: 3px double #1a1a1a;
    border-radius: 0;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: none;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Old Standard TT', serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4a4a4a !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'PT Serif', serif !important;
    font-size: 0.95rem !important;
}

.stButton > button, .stDownloadButton > button {
    font-family: 'Old Standard TT', serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    background: #f5f1e8 !important;
    color: #1a1a1a !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 0 !important;
    box-shadow: 2px 2px 0 #1a1a1a;
}
.stButton > button:hover {
    background: #1a1a1a !important;
    color: #f5f1e8 !important;
}
.stButton > button[kind="primary"] {
    background: #8b0000 !important;
    color: #f5f1e8 !important;
    border-color: #8b0000 !important;
    box-shadow: 2px 2px 0 #1a1a1a;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Old Standard TT', serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a4a4a;
}
.stTabs [aria-selected="true"] { color: #8b0000 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #8b0000; }

.stTextArea textarea, [data-baseweb="select"] > div, [data-testid="stDateInput"] input {
    background-color: #f5f1e8 !important;
    border-color: #1a1a1a !important;
    color: #1a1a1a !important;
}
</style>
"""


def _register_plotly_template(
    name: str,
    *,
    plot_bg: str,
    grid: str,
    font_family: str,
    font_size: int,
    font_color: str,
    colorway: list[str],
) -> None:
    t = copy.deepcopy(pio.templates["plotly_dark"])
    t.layout.paper_bgcolor = "rgba(0,0,0,0)"
    t.layout.plot_bgcolor = plot_bg
    t.layout.font = dict(family=font_family, size=font_size, color=font_color)
    t.layout.colorway = colorway
    t.layout.xaxis.gridcolor = grid
    t.layout.yaxis.gridcolor = grid
    pio.templates[name] = t


_register_plotly_template(
    "arcade",
    plot_bg="rgba(22,22,58,0.55)",
    grid="#2a2a5e",
    font_family="VT323, monospace",
    font_size=16,
    font_color="#e0e0ff",
    colorway=NEON_COLORWAY,
)
_register_plotly_template(
    "exchange",
    plot_bg="rgba(17,22,31,0.6)",
    grid="#1c2430",
    font_family="IBM Plex Mono, monospace",
    font_size=13,
    font_color="#c5ccd6",
    colorway=EXCH_COLORWAY,
)


def _register_light_plotly_template(
    name: str,
    *,
    paper: str,
    plot_bg: str,
    grid: str,
    font_family: str,
    font_size: int,
    font_color: str,
    colorway: list[str],
) -> None:
    t = copy.deepcopy(pio.templates["plotly_white"])
    t.layout.paper_bgcolor = paper
    t.layout.plot_bgcolor = plot_bg
    t.layout.font = dict(family=font_family, size=font_size, color=font_color)
    t.layout.colorway = colorway
    t.layout.xaxis.gridcolor = grid
    t.layout.yaxis.gridcolor = grid
    pio.templates[name] = t


_register_light_plotly_template(
    "solarized",
    paper=SOL_BG,
    plot_bg=SOL_PANEL,
    grid="#d8d2bf",
    font_family="JetBrains Mono, monospace",
    font_size=13,
    font_color=SOL_INK,
    colorway=SOL_COLORWAY,
)
_register_plotly_template(
    "cyberpunk",
    plot_bg="rgba(26,0,51,0.5)",
    grid="#3a0a5e",
    font_family="Rajdhani, sans-serif",
    font_size=14,
    font_color="#e0d4ff",
    colorway=CP_COLORWAY,
)
_register_light_plotly_template(
    "broadsheet",
    paper=BS_BG,
    plot_bg=BS_PANEL,
    grid="#bfb89e",
    font_family="PT Serif, serif",
    font_size=13,
    font_color=BS_INK,
    colorway=BS_COLORWAY,
)

# Each theme bundles its CSS, plotly template name, gain/loss/accent colors
# (used by inline-styled values and chart traces), and the title string.
# Order matters: the first entry is the default selection.
THEMES = {
    "📊 Stock Exchange": {
        "css": EXCHANGE_CSS,
        "plotly": "exchange",
        "up": EXCH_GREEN,
        "down": EXCH_RED,
        "accent": AMBER,
        "colorway": EXCH_COLORWAY,
        "title": "📊 Portfolio Tracker",
    },
    "🕹️ Retro Arcade": {
        "css": GAME_CSS,
        "plotly": "arcade",
        "up": NEON_GREEN,
        "down": NEON_RED,
        "accent": "#08f7fe",
        "colorway": NEON_COLORWAY,
        "title": "🕹️ PORTFOLIO TRACKER",
    },
    "☀️ Solarized Light": {
        "css": SOLARIZED_CSS,
        "plotly": "solarized",
        "up": SOL_GREEN,
        "down": SOL_RED,
        "accent": SOL_BLUE,
        "colorway": SOL_COLORWAY,
        "title": "☀️ Portfolio Tracker",
    },
    "🌌 Cyberpunk": {
        "css": CYBERPUNK_CSS,
        "plotly": "cyberpunk",
        "up": CP_UP,
        "down": CP_DOWN,
        "accent": CP_CYAN,
        "colorway": CP_COLORWAY,
        "title": "🌌 PORTFOLIO TRACKER",
    },
    "📰 Broadsheet": {
        "css": BROADSHEET_CSS,
        "plotly": "broadsheet",
        "up": BS_GREEN,
        "down": BS_RED,
        "accent": BS_ACCENT,
        "colorway": BS_COLORWAY,
        "title": "📰 The Portfolio Tracker",
    },
}


HOLDINGS_COLUMNS = ["ticker", "market", "shares", "cost_basis", "currency"]

# Browser localStorage key. Holdings are persisted here (per browser, JSON of
# the parsed rows) so a returning visitor keeps their portfolio across reloads.
# localStorage is per-browser, so this preserves the multi-user isolation:
# each visitor only ever restores their OWN saved holdings.
LS_HOLDINGS_KEY = "portfolio_holdings_json"
LS_THEME_KEY = "portfolio_theme"
LS_HIDE_SUMMARY_KEY = "portfolio_hide_summary"


def save_holdings_to_browser(ls: LocalStorage, rows: list[dict]) -> None:
    """Persist holdings to this browser's localStorage."""
    ls.setItem(LS_HOLDINGS_KEY, json.dumps(rows), key="ls_save_holdings")


def load_holdings_from_browser(ls: LocalStorage) -> list[dict] | None:
    """Read holdings back from localStorage; None if nothing saved/parseable."""
    stored = ls.getItem(LS_HOLDINGS_KEY)
    if not stored:
        return None
    try:
        rows = json.loads(stored)
        return rows or None
    except (ValueError, TypeError):
        return None


def clear_holdings_in_browser(ls: LocalStorage) -> None:
    """Remove the persisted holdings from this browser's localStorage."""
    if ls.getItem(LS_HOLDINGS_KEY) is not None:  # deleteItem KeyErrors if absent
        ls.deleteItem(LS_HOLDINGS_KEY, key="ls_clear_holdings")


def load_theme_from_browser(ls: LocalStorage) -> str | None:
    """Read the saved theme name from this browser's localStorage."""
    stored = ls.getItem(LS_THEME_KEY)
    return stored if isinstance(stored, str) and stored in THEMES else None


def save_theme_to_browser(ls: LocalStorage, theme_name: str) -> None:
    """Persist the selected theme to this browser's localStorage."""
    ls.setItem(LS_THEME_KEY, theme_name, key="ls_save_theme")


def load_hide_summary_from_browser(ls: LocalStorage) -> bool | None:
    """Read the saved Hide-totals flag from this browser's localStorage."""
    stored = ls.getItem(LS_HIDE_SUMMARY_KEY)
    if stored is None:
        return None
    return str(stored).lower() == "true"


def save_hide_summary_to_browser(ls: LocalStorage, hide: bool) -> None:
    """Persist the Hide-totals toggle to this browser's localStorage."""
    ls.setItem(LS_HIDE_SUMMARY_KEY, "true" if hide else "false", key="ls_save_hide_summary")


# IMPORTANT (multi-user): holdings are PER SESSION, never read from the shared
# DB. Streamlit's `st.cache_data` and the SQLite file are both process-global —
# shared across every visitor — so caching holdings there would let one user
# see another's portfolio. We keep holdings in `st.session_state` (per browser
# session) and derive a DataFrame from it on each run.
def session_holdings() -> pd.DataFrame:
    rows = st.session_state.get("holdings_rows") or []
    if not rows:
        return pd.DataFrame(columns=HOLDINGS_COLUMNS)
    df = pd.DataFrame(rows)
    for col in HOLDINGS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return (
        df[HOLDINGS_COLUMNS]
        .sort_values(["market", "ticker"])
        .reset_index(drop=True)
    )


# Prices and FX are PUBLIC market data keyed by ticker/date — safe and useful
# to share across all sessions, so these stay cached (6h; they only change
# after market close).
@st.cache_data(ttl=21600)  # 6 hours
def get_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return load_prices(tickers=list(tickers), start=start, end=end)


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


def _ensure_holdings_loaded(ls: LocalStorage) -> bool:
    """Resolve holdings source: paste / upload → this session → this browser's
    localStorage → local file.
    Returns True if holdings are present in `st.session_state` for this session.
    Holdings are kept per session/per browser only (never in the shared DB) so
    concurrent users on the same Streamlit Cloud container don't see each
    other's data; localStorage is per-browser so a returning visitor keeps
    their own portfolio across reloads.
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

        # Fall back to this session's earlier upload, then this browser's
        # localStorage (a returning visitor's saved portfolio), then (dev only)
        # the local YAML. We deliberately do NOT read holdings from the shared
        # DB: on Streamlit Cloud every visitor shares one container, so a DB
        # fallback would show one user another user's portfolio.
        if rows is None and "holdings_rows" in st.session_state:
            rows = st.session_state["holdings_rows"]
            source = "session (loaded earlier)"
        if rows is None:
            restored = load_holdings_from_browser(ls)
            if restored:
                rows = restored
                st.session_state["holdings_rows"] = rows
                source = "saved in this browser"
        if rows is None and LOCAL_HOLDINGS_PATH.exists():
            try:
                with open(LOCAL_HOLDINGS_PATH, "r", encoding="utf-8") as f:
                    rows = parse_holdings_yaml(f.read())
                st.session_state["holdings_rows"] = rows
                source = "local config/holdings.yaml"
            except Exception as e:
                st.error(f"Could not read local holdings.yaml: {e}")
                return False

        if not rows:
            st.info("👆 Paste your YAML in the **Paste** tab and click **Apply**.")
            return False

        # When holdings change (signature includes ticker and shares): trigger a
        # fresh price fetch for any new tickers, and persist to this browser so
        # the portfolio survives a reload / return visit.
        rows_sig = repr(sorted([(r.get("ticker"), r.get("shares")) for r in rows]))
        if st.session_state.get("holdings_sig") != rows_sig:
            st.session_state["holdings_sig"] = rows_sig
            st.session_state["holdings_changed"] = True
            save_holdings_to_browser(ls, rows)

        st.caption(f"Source: {source} · {len(rows)} tickers")
        return True


def _ensure_prices_loaded(tickers: list[str]) -> None:
    """Make sure the shared price cache covers this session's tickers.

    Prices and FX are public market data shared across all sessions, so we only
    fetch tickers that are missing from the `prices` table. Per-session daily
    totals are computed in-memory in main() from these prices × the session's
    holdings — nothing user-specific is written to the shared DB here.
    """
    # Clearing the flag keeps it from forcing repeat fetches once handled.
    st.session_state.pop("holdings_changed", False)

    prices_df = load_prices(tickers=tickers)
    have_tickers = set(prices_df["ticker"].unique()) if not prices_df.empty else set()
    missing = [t for t in tickers if t not in have_tickers]

    if missing:
        fetch_msg = st.empty()
        with st.spinner(f"Fetching prices for {len(missing)} ticker(s) from Yahoo Finance..."):
            try:
                fetch_for_tickers(missing)
                fetch_fx_rates()
            except Exception as e:
                fetch_msg.error(f"Yahoo Finance fetch failed: {e}")

        # New rows landed in the shared DB; drop the cached reads so this run
        # (and other sessions) see them.
        get_prices.clear()
        get_fx_rates.clear()

        # Re-check what we actually got
        prices_after = load_prices(tickers=tickers)
        got_tickers = set(prices_after["ticker"].unique()) if not prices_after.empty else set()
        still_missing = [t for t in tickers if t not in got_tickers]
        if still_missing:
            fetch_msg.error(
                f"Yahoo Finance returned no data for: {', '.join(still_missing)}. "
                f"Check the ticker symbols (US: plain, TW: append .TW or .TWO)."
            )


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
    """Re-fetch the latest prices/FX for THIS session's holdings into the shared
    price cache, then drop the cached reads so the new data shows immediately.
    Totals are recomputed in-memory on the next render — no user data is written
    to the shared DB.
    """
    holdings = session_holdings()
    if holdings.empty:
        return "No holdings to fetch."
    tickers = holdings["ticker"].tolist()
    fetch_for_tickers(tickers)
    fetch_fx_rates()
    get_prices.clear()
    get_fx_rates.clear()
    return f"Refreshed prices for {len(tickers)} holdings."


def _wipe_all_holdings_data() -> None:
    """Clear this session's holdings. Only session state is touched — the shared
    `prices`/`fx_rates` cache is public market data and is left intact, and no
    holdings/totals are stored in the shared DB anymore.
    """
    for key in (
        "holdings_rows",
        "holdings_sig",
        "holdings_text_input",
        "holdings_changed",
    ):
        st.session_state.pop(key, None)


def main() -> None:
    init_db()

    # Per-browser persistence. Instantiating LocalStorage mounts a hidden
    # component that loads all stored items (blocking briefly on first run),
    # so getItem/setItem work for the rest of this run.
    ls = LocalStorage()

    # A pending clear is handled here — before holdings are resolved — so the
    # localStorage delete renders in a normally-completing run (deleting inside
    # the button handler then st.rerun() would abort before the delete is sent).
    if st.session_state.pop("pending_clear", False):
        _wipe_all_holdings_data()
        clear_holdings_in_browser(ls)

    # Resolve the theme name BEFORE rendering anything that depends on
    # `theme` (title, columns). The selectbox itself is rendered below
    # inside the main top bar — CSS injected via st.markdown still applies
    # globally regardless of where the <style> tag sits in the DOM.
    # First visit in this session: hydrate the widget's session_state from
    # this browser's localStorage so the saved theme survives reloads.
    if "ui_theme" not in st.session_state:
        saved_theme = load_theme_from_browser(ls)
        if saved_theme:
            st.session_state["ui_theme"] = saved_theme

    theme_name = st.session_state.get("ui_theme", next(iter(THEMES)))
    if theme_name not in THEMES:
        theme_name = next(iter(THEMES))
    theme = THEMES[theme_name]

    st.markdown(theme["css"], unsafe_allow_html=True)
    pio.templates.default = theme["plotly"]

    title_col, theme_col, refresh_col, hide_col = st.columns(
        [4, 1.8, 1.4, 1.4], vertical_alignment="center"
    )
    with title_col:
        st.title(theme["title"])
        st.caption("US + TW daily prices · holdings × close → daily portfolio value (TWD combined)")
    with theme_col:
        def _persist_theme() -> None:
            save_theme_to_browser(ls, st.session_state["ui_theme"])

        st.selectbox(
            "🎨 Theme",
            list(THEMES),
            key="ui_theme",
            label_visibility="collapsed",
            on_change=_persist_theme,
        )
    with refresh_col:
        if st.button("🔄 Refresh prices", use_container_width=True):
            with st.spinner("Fetching from Yahoo Finance..."):
                msg = refresh_all()
            st.success(msg)
    with hide_col:
        if "hide_summary" not in st.session_state:
            saved_hide = load_hide_summary_from_browser(ls)
            if saved_hide is not None:
                st.session_state["hide_summary"] = saved_hide

        def _persist_hide_summary() -> None:
            save_hide_summary_to_browser(ls, st.session_state["hide_summary"])

        hide_summary = st.toggle(
            "Hide totals",
            value=False,
            key="hide_summary",
            on_change=_persist_hide_summary,
        )

    # Reserve a slot at the very top of the sidebar for the Controls block.
    # We populate it AFTER _ensure_holdings_loaded() runs so we know whether
    # holdings exist (after a fresh Apply click, the session state is only
    # updated mid-run inside that helper).
    with st.sidebar:
        controls_slot = st.empty()

    # Holdings input renders below the reserved slot.
    if not _ensure_holdings_loaded(ls):
        return

    with st.sidebar:
        st.caption("Holdings are saved in your browser only (this device).")

    today = date.today()
    default_start = today - timedelta(days=365)
    start_d, end_d = default_start, today

    # Now fill the reserved slot at the top — holdings exist at this point.
    with controls_slot.container():
        st.header("Controls")
        if st.button("🗑️ Clear holdings", use_container_width=True, type="secondary"):
            # Defer to the top-of-run handler so the localStorage delete is sent
            # before this session re-resolves (and re-saves) holdings.
            st.session_state["pending_clear"] = True
            st.rerun()

        date_range = st.date_input(
            "Date range",
            value=(default_start, today),
            max_value=today,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range

        st.divider()

    holdings = session_holdings()
    if holdings.empty:
        st.warning("No holdings loaded. Upload your `holdings.yaml` from the sidebar.")
        return

    # Fetch any tickers missing from the shared price cache (e.g. first run on
    # Streamlit Cloud, or a newly added ticker).
    _ensure_prices_loaded(holdings["ticker"].tolist())

    start_s = start_d.isoformat()
    end_s = end_d.isoformat()
    prices = get_prices(tuple(holdings["ticker"].tolist()), start_s, end_s)
    fx = get_fx_rates("USDTWD")

    # Compute this session's daily totals in-memory: shared prices × this
    # session's holdings. Never read/written from the shared DB, so each user
    # sees only their own portfolio.
    totals = compute_daily_totals(holdings=holdings, prices=prices)

    if totals.empty:
        st.info("No price data yet. Click **🔄 Refresh prices** beside the title to fetch.")
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

    if not hide_summary:
        st.subheader("Summary")
        last_fetch = get_meta("last_price_fetch")
        if last_fetch:
            st.caption(f"Prices last updated: **{last_fetch}** (Taipei, 24hr)")
        if fx_today:
            if fx_prev:
                fx_delta = fx_today - fx_prev
                fx_pct = fx_delta / fx_prev * 100.0
                color = theme["up"] if fx_delta >= 0 else theme["down"]
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
            return f"color:{theme['up']}" if v >= 0 else f"color:{theme['down']}"

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

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.Column(pinned=True),
            },
        )

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
                    color_discrete_map={"TW": theme["accent"], "US": theme["down"]},
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

                    colors = [theme["up"] if v >= 0 else theme["down"] for v in gain_filtered["gain_twd"]]
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
                            line=dict(color=theme["accent"], width=1.5),
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
                        color = theme["up"] if v >= 0 else theme["down"]
                        return f"<span style='color:{color}'>{v:+,.0f}</span>"

                    palette = theme["colorway"]
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
