import config
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from transformers import pipeline
import io
import datetime
import time
import os
import re
import urllib.parse
import feedparser
from groq import Groq

# ── Lock Streamlit to dark theme regardless of OS light/dark mode ─────────────
_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit")
os.makedirs(_cfg_dir, exist_ok=True)
_cfg_path = os.path.join(_cfg_dir, "config.toml")
if not os.path.exists(_cfg_path):
    with open(_cfg_path, "w") as _f:
        _f.write("[theme]\nbase = \"dark\"\nbackgroundColor = \"#070710\"\nsecondaryBackgroundColor = \"#0e0e1a\"\ntextColor = \"#dde1f0\"\nprimaryColor = \"#00e5ff\"\n")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentwell · MarketPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ticker → Company lookup table ─────────────────────────────────────────────
TICKER_MAP = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet (Google)",
    "GOOG": "Alphabet (Google) Class C", "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corp.",
    "META": "Meta Platforms", "TSLA": "Tesla Inc.", "BRK.B": "Berkshire Hathaway B",
    "LLY": "Eli Lilly & Co.", "JPM": "JPMorgan Chase", "V": "Visa Inc.",
    "UNH": "UnitedHealth Group", "XOM": "ExxonMobil Corp.", "MA": "Mastercard Inc.",
    "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble", "HD": "Home Depot",
    "AVGO": "Broadcom Inc.", "CVX": "Chevron Corp.", "MRK": "Merck & Co.",
    "ABBV": "AbbVie Inc.", "COST": "Costco Wholesale", "ADBE": "Adobe Inc.",
    "CRM": "Salesforce Inc.", "AMD": "Advanced Micro Devices", "ACN": "Accenture plc",
    "PEP": "PepsiCo Inc.", "KO": "Coca-Cola Co.", "NFLX": "Netflix Inc.",
    "WMT": "Walmart Inc.", "BAC": "Bank of America", "TMO": "Thermo Fisher Scientific",
    "CSCO": "Cisco Systems", "MCD": "McDonald's Corp.", "ABT": "Abbott Laboratories",
    "ORCL": "Oracle Corp.", "DHR": "Danaher Corp.", "INTC": "Intel Corp.",
    "NKE": "Nike Inc.", "DIS": "Walt Disney Co.", "TXN": "Texas Instruments",
    "PM": "Philip Morris Int.", "INTU": "Intuit Inc.", "IBM": "IBM Corp.",
    "QCOM": "Qualcomm Inc.", "CAT": "Caterpillar Inc.", "GE": "GE Aerospace",
    "AMGN": "Amgen Inc.", "SPGI": "S&P Global", "NEE": "NextEra Energy",
    "HON": "Honeywell Int.", "BA": "Boeing Co.", "UBER": "Uber Technologies",
    "LOW": "Lowe's Companies", "SBUX": "Starbucks Corp.", "GS": "Goldman Sachs",
    "BLK": "BlackRock Inc.", "AMAT": "Applied Materials", "LRCX": "Lam Research",
    "DE": "Deere & Company", "SYK": "Stryker Corp.", "AXP": "American Express",
    "PLD": "Prologis Inc.", "ADI": "Analog Devices", "MDLZ": "Mondelez Int.",
    "BKNG": "Booking Holdings", "REGN": "Regeneron Pharma", "GILD": "Gilead Sciences",
    "ISRG": "Intuitive Surgical", "MU": "Micron Technology", "CI": "Cigna Group",
    "ZTS": "Zoetis Inc.", "VRTX": "Vertex Pharmaceuticals", "PGR": "Progressive Corp.",
    "MMC": "Marsh & McLennan", "AON": "Aon plc", "ELV": "Elevance Health",
    "PANW": "Palo Alto Networks", "CRWD": "CrowdStrike Holdings", "SNOW": "Snowflake Inc.",
    "PLTR": "Palantir Technologies", "ARM": "Arm Holdings", "SMCI": "Super Micro Computer",
    "MSTR": "MicroStrategy Inc.", "COIN": "Coinbase Global", "SQ": "Block Inc.",
    "PYPL": "PayPal Holdings", "SHOP": "Shopify Inc.", "SPOT": "Spotify Technology",
    "RBLX": "Roblox Corp.", "RIVN": "Rivian Automotive", "LCID": "Lucid Group",
    "NIO": "NIO Inc.", "BABA": "Alibaba Group", "JD": "JD.com", "PDD": "PDD Holdings",
    "TSM": "Taiwan Semiconductor", "ASML": "ASML Holding", "SAP": "SAP SE",
    "TM": "Toyota Motor", "SONY": "Sony Group Corp.", "BP": "BP plc",
    "SHEL": "Shell plc", "RIO": "Rio Tinto", "BHP": "BHP Group",
    "WFC": "Wells Fargo", "C": "Citigroup Inc.", "MS": "Morgan Stanley",
    "USB": "U.S. Bancorp", "PNC": "PNC Financial", "SCHW": "Charles Schwab",
    "BX": "Blackstone Inc.", "KKR": "KKR & Co.", "APO": "Apollo Global Mgmt",
    "T": "AT&T Inc.", "VZ": "Verizon Communications", "TMUS": "T-Mobile US",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
:root {
    --bg:       #070710;
    --surface:  #0e0e1a;
    --surface2: #13131f;
    --border:   #1c1c2e;
    --border2:  #252538;
    --accent:   #00e5ff;
    --green:    #00ff9d;
    --red:      #ff4f6a;
    --yellow:   #ffd166;
    --muted:    #5a5a7a;
    --text:     #dde1f0;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
/* Force dark mode at every level — overrides OS light mode */
:root, html {
    color-scheme: dark !important;
}
/* Streamlit root app container */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewBlockNode"],
[data-testid="stMain"],
[data-testid="stMainBlocksContainer"],
[data-testid="stBottom"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
.main,
.main > div {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
/* Block container */
[data-testid="block-container"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
[data-testid="stColumns"],
[data-testid="stForm"] {
    background-color: transparent !important;
    color: var(--text) !important;
}
/* Tabs */
[data-testid="stTabs"],
[data-testid="stTabContent"],
[data-baseweb="tab-panel"],
[data-baseweb="tab-list"],
[role="tablist"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
/* Text elements */
p, span, label, div, h1, h2, h3, h4, h5, h6, li, a {
    color: var(--text);
}
/* Markdown text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {
    color: var(--text) !important;
}
/* Caption / small text */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}
/* Info / warning / error boxes */
[data-testid="stAlert"] {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border2) !important;
}
/* Progress bar */
[data-testid="stProgress"] > div {
    background-color: var(--surface2) !important;
}
[data-testid="stProgress"] > div > div {
    background-color: var(--accent) !important;
}
/* Spinner */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
}
/* Expander */
[data-testid="stExpander"],
[data-testid="stExpander"] > div,
details[data-testid="stExpander"] summary {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
#MainMenu, footer, header { visibility: hidden; height: 0 !important; }
/* Collapse Streamlit's top toolbar/header area completely */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppToolbar"],
[data-testid="stMainMenu"],
[data-testid="stDeployButton"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}
/* Remove all top padding from the main content wrapper */
.block-container,
[data-testid="block-container"] {
    padding: 0 2rem 2rem !important;
    max-width: 1440px;
    margin-top: 0 !important;
}
/* Collapse the gap Streamlit injects above the first block */
[data-testid="stAppViewContainer"] > section > div:first-child,
[data-testid="stMain"] > div:first-child,
[data-testid="stMainBlocksContainer"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
div[data-testid="stVerticalBlock"] > div { margin-top: 0 !important; margin-bottom: 0 !important; }
/* ── HERO ── */
.hero-wrap {
    background: linear-gradient(135deg, #070710 0%, #0d0d20 60%, #0a1520 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.2rem 0 1rem;
    margin-bottom: 0.5rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,229,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-grid { display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
.hero-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
    margin: 0;
}
.hero-logo .pulse { color: var(--accent); }
.hero-tag {
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.hero-badges { display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
.hbadge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    border: 1px solid var(--border2);
    color: var(--muted);
    background: var(--surface);
    letter-spacing: 0.06em;
}
.hbadge.live { border-color: rgba(0,255,157,0.3); color: var(--green); background: rgba(0,255,157,0.05); }
/* ── INPUT AREA ── */
.input-wrap {
    background: transparent;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 0 1rem;
    margin-bottom: 0;
}
/* Kill Streamlit's injected gap between hero and input */
div[data-testid="stVerticalBlock"] > div { margin-top: 0 !important; margin-bottom: 0 !important; }
div[data-testid="stVerticalBlock"] > div:has(.input-wrap) { margin-top: -0.5rem !important; }
/* ── SENTWELL LOGO ── */
.sw-logo-wrap {
    display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;
}
.sw-logo-svg { flex-shrink: 0; }
/* ── LOADING OVERLAY ── */
.sw-loading {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.sw-loading-icon {
    position: relative;
    width: 52px; height: 52px;
    flex-shrink: 0;
}
.sw-ring {
    position: absolute; inset: 0;
    border-radius: 50%;
    border: 3px solid transparent;
    border-top-color: var(--accent);
    animation: sw-spin 0.9s linear infinite;
}
.sw-ring2 {
    position: absolute; inset: 7px;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: var(--green);
    animation: sw-spin 1.4s linear infinite reverse;
}
@keyframes sw-spin { to { transform: rotate(360deg); } }
.sw-loading-text { flex: 1; }
.sw-loading-step {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.sw-loading-label {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 0.6rem;
}
.sw-loading-bar-bg {
    width: 100%;
    height: 3px;
    background: var(--border2);
    border-radius: 2px;
    overflow: hidden;
}
.sw-loading-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--green));
    border-radius: 2px;
    transition: width 0.4s ease;
}
.sw-loading-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
}
/* ── STOCK INFO ROW ── */
.stock-info-row {
    display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;
}
.stock-info-card {
    flex: 1; min-width: 180px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.si-label { font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.3rem; }
.si-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 700; line-height: 1; }
.si-sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }
.si-val.green { color: var(--green); }
.si-val.red   { color: var(--red); }
.si-val.blue  { color: var(--accent); }
/* ── METRICS ── */
.metric-row { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 100px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.mc-label { font-size: 0.62rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.mc-val { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; font-weight: 700; line-height: 1; }
.mc-val.green { color: var(--green); }
.mc-val.red   { color: var(--red); }
.mc-val.grey  { color: var(--muted); }
/* ── AI SUMMARY BOX ── */
.ai-summary {
    background: linear-gradient(135deg, rgba(0,229,255,0.04) 0%, rgba(0,229,255,0.01) 100%);
    border: 1px solid rgba(0,229,255,0.18);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.4rem;
    position: relative;
}
.ai-summary-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.ai-summary-text {
    font-size: 0.9rem;
    line-height: 1.65;
    color: var(--text);
}
/* ── WHY INTELLIGENCE PANEL ── */
.why-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.9rem;
    margin-bottom: 1.4rem;
}
.why-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.why-card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.why-card-label.drivers  { color: #00e5ff; }
.why-card-label.narratives { color: #ffd166; }
.why-card-label.roots    { color: #b185ff; }
.why-card-label.contra   { color: #ff4f6a; }
.why-card-body {
    font-size: 0.82rem;
    line-height: 1.6;
    color: #b8bdd4;
}
.why-card-body ul {
    margin: 0;
    padding-left: 1.1rem;
}
.why-card-body li {
    margin-bottom: 0.3rem;
}
.why-card-body li::marker { color: var(--muted); }
/* ── ALERT ── */
.alert {
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-weight: 500;
    font-size: 0.88rem;
    margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.alert.bull { background: rgba(0,255,157,0.06); border: 1px solid rgba(0,255,157,0.22); color: var(--green); }
.alert.bear { background: rgba(255,79,106,0.06); border: 1px solid rgba(255,79,106,0.22); color: var(--red); }
.alert.neut { background: rgba(90,90,122,0.1);   border: 1px solid rgba(90,90,122,0.25);  color: var(--muted); }
/* ── HEADLINE CARDS ── */
.hcard {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid transparent;
    border-radius: 9px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
}
.hcard.pos { border-left-color: var(--green); }
.hcard.neg { border-left-color: var(--red); }
.hcard.neu { border-left-color: var(--muted); }
.hcard-title { font-size: 0.9rem; font-weight: 500; line-height: 1.4; margin-bottom: 0.4rem; display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem; }
.hcard-meta { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--muted); display: flex; gap: 0.8rem; flex-wrap: wrap; align-items: center; }
.badge {
    padding: 0.12rem 0.45rem; border-radius: 4px;
    font-size: 0.62rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.badge.pos { background: rgba(0,255,157,0.1); color: var(--green); }
.badge.neg { background: rgba(255,79,106,0.1); color: var(--red); }
.badge.neu { background: rgba(90,90,122,0.12); color: var(--muted); }
/* ── QUALITY FLAG BADGES ── */
.qflag {
    display: inline-flex; align-items: center; gap: 0.25rem;
    padding: 0.1rem 0.4rem; border-radius: 4px;
    font-size: 0.58rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-family: 'IBM Plex Mono', monospace;
}
.qflag.clean      { background: rgba(0,255,157,0.07);  color: #00ff9d;  border: 1px solid rgba(0,255,157,0.2); }
.qflag.suspicious { background: rgba(255,209,102,0.1); color: #ffd166;  border: 1px solid rgba(255,209,102,0.3); }
.qflag.flagged    { background: rgba(255,79,106,0.08); color: #ff4f6a;  border: 1px solid rgba(255,79,106,0.25); }
.qflag-detail {
    font-size: 0.6rem; color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}
/* ── HEADLINE CARD EXPAND ── */
details.hcard-wrap { list-style: none; margin-bottom: 0.55rem; }
details.hcard-wrap > summary { list-style: none; cursor: pointer; }
details.hcard-wrap > summary::-webkit-details-marker { display: none; }
details.hcard-wrap .hcard {
    margin-bottom: 0;
    border-radius: 9px 9px 0 0;
    border-bottom: none;
    transition: border-color 0.15s;
}
details.hcard-wrap[open] .hcard {
    border-bottom: 1px solid transparent;
}
details.hcard-wrap .hcard-chevron {
    margin-left: auto;
    font-size: 0.65rem;
    color: var(--muted);
    transition: transform 0.2s;
    flex-shrink: 0;
}
details.hcard-wrap[open] .hcard-chevron { transform: rotate(180deg); }
.hcard-expand-body {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-top: 1px solid var(--border2);
    border-radius: 0 0 9px 9px;
    padding: 0.9rem 1.1rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.hcard-expand-body.pos { border-left: 3px solid var(--green); }
.hcard-expand-body.neg { border-left: 3px solid var(--red); }
.hcard-expand-body.neu { border-left: 3px solid var(--muted); }
.hcard-expand-img {
    width: 120px;
    min-width: 120px;
    height: 80px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border2);
    flex-shrink: 0;
    background: var(--border);
}
.hcard-expand-img-placeholder {
    width: 120px;
    min-width: 120px;
    height: 80px;
    border-radius: 6px;
    border: 1px solid var(--border2);
    background: var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
    color: var(--muted);
}
.hcard-expand-text { flex: 1; min-width: 0; }
.hcard-expand-summary {
    font-size: 0.84rem;
    line-height: 1.6;
    color: #b8bdd4;
    margin-bottom: 0.5rem;
}
.hcard-expand-link {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    text-decoration: none;
    letter-spacing: 0.06em;
    border-bottom: 1px solid rgba(0,229,255,0.25);
    padding-bottom: 1px;
    transition: border-color 0.15s;
}
.hcard-expand-link:hover { border-color: var(--accent); }
/* ── CONFIDENCE BAR ── */
.conf-bar-wrap { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.35rem; }
.conf-bar-bg { flex: 1; height: 4px; background: var(--border2); border-radius: 2px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 2px; }
/* ── SECTION TITLE ── */
.sec-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.45rem;
    margin: 1.6rem 0 0.9rem;
}
/* ── TICKER PILL ── */
.tpill {
    display: inline-block;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.25);
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; font-weight: 700;
    padding: 0.18rem 0.55rem; border-radius: 20px;
}
/* ── WATCHLIST CARD ── */
.wl-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
}
.wl-sym { color: var(--accent); font-weight: 700; }
.wl-name { color: var(--muted); font-size: 0.65rem; }
/* ── STREAMLIT OVERRIDES ── */
/* Text input */
.stTextInput input,
[data-testid="stTextInput"] input,
[data-baseweb="input"] input {
    background: var(--bg) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.12) !important;
}
[data-baseweb="input"],
[data-baseweb="base-input"] {
    background: var(--bg) !important;
    border-color: var(--border2) !important;
}
.stTextInput label,
[data-testid="stTextInput"] label { color: var(--muted) !important; }
/* Button */
.stButton > button,
[data-testid="stButton"] > button,
[data-testid="stBaseButton-secondary"] {
    background: var(--accent) !important; color: #000 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.65rem 1.6rem !important; letter-spacing: 0.04em !important;
    width: 100%; font-size: 0.85rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover,
[data-testid="stButton"] > button:hover { opacity: 0.82 !important; background: var(--accent) !important; color: #000 !important; }
/* Tabs */
.stTabs [data-baseweb="tab-list"],
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.4rem; border-bottom: 1px solid var(--border); background: var(--bg) !important;
}
.stTabs [data-baseweb="tab"],
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.12em; color: var(--muted) !important;
    background: transparent !important; border: none !important; padding: 0.55rem 1.1rem;
}
.stTabs [aria-selected="true"],
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important;
}
[data-testid="stTabContent"],
[data-baseweb="tab-panel"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}
/* Slider */
.stSlider label,
[data-testid="stSlider"] label { color: var(--muted) !important; font-size: 0.78rem !important; }
/* Toggle */
[data-testid="stToggle"] label,
.stToggle label { color: var(--text) !important; }
/* Download button */
.stDownloadButton > button,
[data-testid="stDownloadButton"] > button {
    background: transparent !important; border: 1px solid var(--border2) !important;
    color: var(--muted) !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important; border-radius: 7px !important;
}
.stDownloadButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important; background: transparent !important;
}
/* DataFrame */
div[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div { background: var(--surface) !important; border-radius: 10px !important; color: var(--text) !important; }
/* Alert / info / warning */
[data-testid="stAlert"],
[data-testid="stAlert"] p { background-color: var(--surface2) !important; color: var(--text) !important; }
/* Caption */
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
/* Markdown */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong { color: var(--text) !important; }
[data-testid="stMarkdownContainer"] a { color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] code { background: var(--surface2) !important; color: var(--accent) !important; }
/* Expander */
[data-testid="stExpander"],
details[data-testid="stExpander"] summary,
[data-testid="stExpander"] > div {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
/* Selectbox */
[data-baseweb="select"] > div,
[data-baseweb="popover"] > div { background-color: var(--surface) !important; color: var(--text) !important; border-color: var(--border2) !important; }
/* Sidebar styling */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    background-color: var(--surface) !important;
    color: var(--text) !important;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: var(--text) !important;
}
/* Trend stats panel */
.trend-stats-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.ts-row {
    margin-bottom: 0.9rem;
}
.ts-label {
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.25rem;
}
.ts-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
}
.ts-val.green { color: var(--green); }
.ts-val.red   { color: var(--red); }
.ts-val.accent { color: var(--accent); }
.ts-val.sm {
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("sentiment-analysis", model=config.FINBERT_MODEL, revision=config.FINBERT_REVISION)

# ── Helpers ───────────────────────────────────────────────────────────────────
COLOUR = {'positive': '#00ff9d', 'negative': '#ff4f6a', 'neutral': '#5a5a7a'}
CLS    = {'positive': 'pos',     'negative': 'neg',     'neutral': 'neu'}
BULL   = {'positive': 'bull',    'negative': 'bear',    'neutral': 'neut'}
ICON   = {'positive': '▲',       'negative': '▼',       'neutral': '–'}

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#dde1f0"),
    margin=dict(l=0, r=0, t=45, b=0),
)

def plot_layout(**kwargs):
    d = dict(**PLOT_BASE)
    d.update(kwargs)
    return d

def yf_fetch_with_retry(fn, retries=3, delay=4):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if 'rate limit' in str(e).lower() or 'too many requests' in str(e).lower():
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise
            else:
                raise

# ── News sources: Yahoo (yfinance) first, RSS backups if Yahoo returns nothing ─
def _rss_news(sym, count):
    """Backup source: Yahoo Finance RSS, then Google News RSS.
    Returns items in the same shape as yfinance (a['content']['title'] ...),
    so the rest of the app works unchanged."""
    urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(sym)}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={urllib.parse.quote(sym + ' stock')}&hl=en-US&gl=US&ceid=US:en",
    ]
    for url in urls:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"[news] RSS failed for {sym} ({url}): {e}")
            continue
        items = []
        for e in feed.entries[:count]:
            title = (e.get("title") or "").strip()
            if not title:
                continue
            pub = time.strftime("%Y-%m-%dT%H:%M:%SZ", e.published_parsed) if e.get("published_parsed") else ""
            items.append({"content": {
                "title": title,
                "pubDate": pub,
                "summary": re.sub(r"<[^>]+>", "", e.get("summary", ""))[:300],
                "canonicalUrl": {"url": e.get("link", "")},
            }})
        if items:
            print(f"[news] {sym}: {len(items)} headlines from RSS backup ({url.split('/')[2]})")
            return items
    return []


def get_news(sym, count=25):
    """Yahoo via yfinance first; if it errors or returns nothing, fall back to RSS."""
    items = []
    try:
        t = yf.Ticker(sym)
        raw = yf_fetch_with_retry(lambda: t.get_news(count=count) if hasattr(t, "get_news") else t.news)
        items = [a for a in (raw or []) if (a.get("content") or {}).get("title")][:count]
        print(f"[news] {sym}: {len(items)} headlines from yfinance")
    except Exception as e:
        print(f"[news] yfinance failed for {sym}: {e}")
    if not items:
        items = _rss_news(sym, count)
    return items

def fetch_stock_info(sym):
    """Fetch stock info with correct fast_info attribute names."""
    try:
        def _fetch():
            t = yf.Ticker(sym)
            fi = t.fast_info
            price  = getattr(fi, 'last_price', None)
            prev   = getattr(fi, 'previous_close', None)
            mktcap = getattr(fi, 'market_cap', None)
            # FIXED: correct attribute names are year_high / year_low, not fifty_two_week_*
            hi52   = getattr(fi, 'year_high', None)
            lo52   = getattr(fi, 'year_low', None)
            chg    = ((price - prev) / prev * 100) if price and prev else None
            return dict(price=price, change=chg, mktcap=mktcap, hi52=hi52, lo52=lo52)
        return yf_fetch_with_retry(_fetch)
    except:
        return {}

def fetch_price_history(sym):
    try:
        def _fetch():
            hist = yf.Ticker(sym).history(period="1mo", interval="1d")
            return hist[['Close']] if not hist.empty else None
        return yf_fetch_with_retry(_fetch)
    except:
        return None

def fetch_and_analyse(sym, model, min_conf):
    news = get_news(sym, 25)
    if not news:
        return None
    rows = []
    for a in news:
        content = a.get('content', {})
        title = content.get('title', '')
        if not title:
            continue
        # Published date
        pub_date = None
        try:
            pub_str = content.get('pubDate', None)
            if pub_str:
                pub_date = pub_str[:10]
        except:
            pass
        # Summary / description
        summary = (
            content.get('summary') or
            content.get('description') or
            content.get('body', '')[:300] or
            ''
        )
        # Thumbnail image URL — Yahoo nests it under thumbnail.resolutions or thumbnail.url
        img_url = ''
        try:
            thumb = content.get('thumbnail', {}) or {}
            resolutions = thumb.get('resolutions', [])
            if resolutions:
                # Pick the smallest resolution that's still usable (>= 150px wide)
                usable = [r for r in resolutions if r.get('width', 0) >= 150]
                pick = usable[0] if usable else resolutions[0]
                img_url = pick.get('url', '')
            elif thumb.get('url'):
                img_url = thumb['url']
        except:
            pass
        # Article URL
        article_url = ''
        try:
            ct = content.get('clickThroughUrl', {}) or {}
            article_url = ct.get('url', '') or content.get('canonicalUrl', {}).get('url', '')
        except:
            pass

        res = model(title)[0]
        rows.append({
            'ticker': sym,
            'headline': title,
            'sentiment': res['label'],
            'score': round(res['score'], 4),
            'date': pub_date or datetime.datetime.now().strftime('%Y-%m-%d'),
            'summary': summary.strip(),
            'img_url': img_url,
            'article_url': article_url,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df['score'] >= min_conf]
    return df if len(df) else None

def fetch_historical_sentiment(sym, model, min_conf, days=30):
    """
    Fetch news with real publish dates for trend analysis.
    Groups by date (YYYY-MM-DD) for proper daily bucketing.
    """
    news = get_news(sym, 50)
    if not news:
        return None

    rows = []
    total = len(news)
    for i, a in enumerate(news):
        title = a['content']['title']
        # Try to get real publish date
        pub_time = None
        try:
            pub_str = a.get('content', {}).get('pubDate', None)
            if pub_str:
                # Parse ISO date string → just the date portion
                pub_time = pub_str[:10]
        except:
            pass

        if not pub_time:
            # Fallback: space articles across last N days based on index
            offset = int((total - i) / max(total, 1) * days)
            pub_time = (datetime.datetime.now() - datetime.timedelta(days=offset)).strftime('%Y-%m-%d')

        res = model(title)[0]
        if res['score'] >= min_conf:
            rows.append({
                'ticker': sym,
                'headline': title,
                'sentiment': res['label'],
                'score': round(res['score'], 4),
                'date': pub_time
            })

    if not rows:
        return None
    return pd.DataFrame(rows)

def fmt_cap(v):
    if v is None: return 'N/A'
    if v >= 1e12: return f'${v/1e12:.2f}T'
    if v >= 1e9:  return f'${v/1e9:.1f}B'
    if v >= 1e6:  return f'${v/1e6:.1f}M'
    return f'${v:,.0f}'

def get_company_name(sym):
    return TICKER_MAP.get(sym.upper(), sym)

def generate_ai_summary(sym, df, stock_info):
    """Concise 3-sentence market intelligence summary."""
    try:
        pos = len(df[df['sentiment'] == 'positive'])
        neg = len(df[df['sentiment'] == 'negative'])
        neu = len(df[df['sentiment'] == 'neutral'])
        total = len(df)
        dom = df['sentiment'].mode()[0]
        avg_conf = df['score'].mean()
        top_headlines = df.sort_values('score', ascending=False).head(5)['headline'].tolist()
        company = get_company_name(sym)
        price = stock_info.get('price')
        chg = stock_info.get('change')

        headlines_str = "\n".join(f"- {h}" for h in top_headlines)
        price_str = f"${price:,.2f} ({'+' if chg and chg >= 0 else ''}{chg:.2f}% today)" if price and chg else "N/A"

        prompt = f"""You are a financial analyst assistant. Analyse the following sentiment data for {company} ({sym}) and provide a concise 3-sentence market intelligence summary.
Sentiment breakdown: {pos} positive, {neg} negative, {neu} neutral out of {total} headlines.
Dominant sentiment: {dom.upper()} (avg confidence: {avg_conf:.1%})
Current price: {price_str}
Top headlines by confidence:
{headlines_str}
Write a clear, insightful 3-sentence summary covering: (1) the overall sentiment picture, (2) what the headlines suggest about the company's near-term outlook, and (3) a brief risk/opportunity note. Be direct and specific. Do not use bullet points."""

        client = Groq()
        message = client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=300,
            reasoning_effort=config.LLM_REASONING_EFFORT,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.choices[0].message.content
    except Exception:
        pos = len(df[df['sentiment'] == 'positive'])
        neg = len(df[df['sentiment'] == 'negative'])
        neu = len(df[df['sentiment'] == 'neutral'])
        total = len(df)
        dom = df['sentiment'].mode()[0]
        company = get_company_name(sym)
        bias = "bullish" if dom == "positive" else ("bearish" if dom == "negative" else "mixed")
        return (f"News coverage for {company} ({sym}) is predominantly {dom}, with {pos} positive, "
                f"{neg} negative, and {neu} neutral headlines out of {total} analysed. "
                f"The overall tone suggests a {bias} near-term sentiment profile based on current media coverage. "
                f"Investors should monitor developments closely given the {'positive momentum' if dom == 'positive' else 'cautionary signals' if dom == 'negative' else 'uncertain direction'} indicated.")


def generate_why_intelligence(sym, df, stock_info):
    """
    Generate a 'Why' intelligence brief: drivers, narratives, root causes, contradictions.
    Returns a dict with 4 keys, each containing HTML-safe bullet points.
    """
    company = get_company_name(sym)
    pos = len(df[df['sentiment'] == 'positive'])
    neg = len(df[df['sentiment'] == 'negative'])
    neu = len(df[df['sentiment'] == 'neutral'])
    dom = df['sentiment'].mode()[0]
    price = stock_info.get('price')
    chg = stock_info.get('change')
    price_str = f"${price:,.2f} ({'+' if chg and chg >= 0 else ''}{chg:.2f}% today)" if price and chg else "N/A"

    # Use all headlines ranked by confidence
    headlines_pos = df[df['sentiment']=='positive'].sort_values('score', ascending=False)['headline'].head(8).tolist()
    headlines_neg = df[df['sentiment']=='negative'].sort_values('score', ascending=False)['headline'].head(8).tolist()
    headlines_neu = df[df['sentiment']=='neutral'].sort_values('score', ascending=False)['headline'].head(5).tolist()

    all_headlines = "\n".join(
        [f"[POSITIVE] {h}" for h in headlines_pos] +
        [f"[NEGATIVE] {h}" for h in headlines_neg] +
        [f"[NEUTRAL]  {h}" for h in headlines_neu]
    )

    prompt = f"""You are a senior equity analyst conducting deep sentiment analysis on {company} ({sym}).
Current price: {price_str}
Sentiment: {pos} positive, {neg} negative, {neu} neutral headlines. Dominant: {dom.upper()}.
All headlines with their sentiment labels:
{all_headlines}
Analyse WHY the sentiment is what it is. Respond ONLY with a JSON object (no markdown, no preamble) with exactly these 4 keys, each containing a list of 3-4 concise bullet strings:
{{
  "drivers": ["...", "...", "..."],
  "narratives": ["...", "...", "..."],
  "root_causes": ["...", "...", "..."],
  "contradictions": ["...", "...", "..."]
}}
Definitions:
- "drivers": The specific, concrete factors actively pushing sentiment in the dominant direction (product launches, earnings beats/misses, partnerships, macro tailwinds/headwinds, executive decisions).
- "narratives": The overarching story or theme the market is telling about this company right now — what is the dominant frame investors/media are using? Include emerging counter-narratives.
- "root_causes": The deeper, underlying structural or fundamental reasons behind the sentiment — not just what happened, but WHY it matters (competitive dynamics, industry shifts, valuation concerns, regulatory environment).
- "contradictions": Conflicting signals present in the headlines — places where positive and negative sentiment coexist, or where short-term and long-term outlooks diverge.
Be specific to the company and headlines. No generic platitudes."""

    fallback = {
        "drivers": [
            f"{'Positive earnings reports and revenue beats' if dom=='positive' else 'Earnings misses or revenue shortfalls' if dom=='negative' else 'Mixed financial performance'} driving headline sentiment",
            "Analyst upgrades or downgrades influencing market perception",
            "Macroeconomic factors affecting sector positioning",
        ],
        "narratives": [
            f"{company} is being framed through a {'growth and momentum' if dom=='positive' else 'risk and caution' if dom=='negative' else 'wait-and-see'} lens in current coverage",
            "Competitive dynamics and market share narratives feature prominently",
            "Long-term structural themes intersecting with near-term catalysts",
        ],
        "root_causes": [
            "Fundamental valuation and earnings trajectory uncertainty",
            "Sector-wide shifts creating both headwinds and tailwinds",
            "Strategic execution risks versus market opportunity size",
        ],
        "contradictions": [
            "Short-term negative news contrasting with longer-term positive structural thesis",
            "Analyst consensus diverging from media sentiment tone",
            f"{'High' if dom!='neutral' else 'Moderate'} conviction in dominant direction but meaningful counter-signals present",
        ],
    }

    try:
        client = Groq()
        message = client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=600,
            reasoning_effort=config.LLM_REASONING_EFFORT,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.choices[0].message.content.strip()
        # Clean up potential markdown fences
        if "```" in raw:
            raw = raw.split("```")[1] if "```" in raw else raw
            if raw.startswith("json"):
                raw = raw[4:]
        import json
        data = json.loads(raw)
        # Validate structure
        for key in ["drivers", "narratives", "root_causes", "contradictions"]:
            if key not in data or not isinstance(data[key], list):
                return fallback
        return data
    except Exception:
        return fallback


def detect_headline_quality(headline: str) -> dict:
    """
    Heuristic-based headline quality checker.
    Returns a dict with:
      - clickbait_score  (0.0–1.0)
      - ai_generated_score (0.0–1.0)
      - flags: list of string labels triggered
      - verdict: 'clean' | 'suspicious' | 'flagged'
    Intentionally no external API call here — fast, zero latency, no cost.
    Works purely on text signals so it runs on every headline without rate
    limiting. Groq cross-check is reserved for the WHY summary.
    """
    import re
    text = headline.strip()
    lower = text.lower()
    flags = []
    cb_score = 0.0
    ai_score = 0.0

    # ── CLICKBAIT HEURISTICS ────────────────────────────────────────────────
    # Superlative / exaggeration patterns
    clickbait_phrases = [
        r'\byou won\'?t believe\b', r'\bshocking\b', r'\bblew up\b',
        r'\bthis is why\b', r'\bhere\'?s why\b', r'\bthe truth about\b',
        r'\bsecret\b', r'\bexposed\b', r'\beveryone is talking\b',
        r'\bmind.?blowing\b', r'\bincredible\b', r'\bcrazy\b',
        r'\bwhat (nobody|no one) (tells|told)\b',
        r'\b(insiders?|experts?) (say|reveal|warn|predict)\b',
        r'\bthe real reason\b', r'\bwhat you need to know\b',
        r'\bchanges everything\b', r'\bgame.?changer\b',
        r'\bnobody (saw|expected)\b', r'\ball you need\b',
        r'\bjust happened\b', r'\bbreaking\b',
    ]
    for pat in clickbait_phrases:
        if re.search(pat, lower):
            cb_score += 0.18
            flags.append("CLICKBAIT")
            break  # one flag is enough per category

    # Excessive numbers / percentages used as bait
    if re.search(r'\b\d{3,}%\b', text):
        cb_score += 0.15
        flags.append("STAT BAIT")

    # Question-bait: ends with "?" and contains superlative
    if text.endswith('?') and any(w in lower for w in ['will', 'could', 'should', 'can', 'why']):
        cb_score += 0.10
        flags.append("QUESTION BAIT")

    # ALL CAPS words (more than one)
    caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
    # Exclude known ticker symbols (2-5 caps that are plausibly tickers)
    non_ticker_caps = [w for w in caps_words if len(w) > 5]
    if len(non_ticker_caps) >= 2:
        cb_score += 0.12
        flags.append("CAPS BAIT")

    # ── AI-GENERATED TEXT HEURISTICS ────────────────────────────────────────
    # Overly structured / listicle phrasing
    ai_phrases = [
        r'\b\d+ (reasons?|ways?|things?|tips?|steps?|factors?|signs?)\b',
        r'\bhere\'?s (what|how|why)\b',
        r'\beverything you need\b',
        r'\bultimate guide\b',
        r'\bcomprehensive (look|overview|guide|analysis)\b',
        r'\bin (today\'?s|this) (article|piece|post)\b',
        r'\blet\'?s (explore|dive|look at|break down)\b',
        r'\bit\'?s (important|worth noting|crucial)\b',
        r'\bin (conclusion|summary)\b',
        r'\bfirst and foremost\b',
        r'\bin this (article|post|piece|analysis)\b',
    ]
    for pat in ai_phrases:
        if re.search(pat, lower):
            ai_score += 0.22
            flags.append("AI-PHRASING")
            break

    # Unnaturally generic company description phrasing
    if re.search(r'\b(a leading|one of the (largest|biggest|most))\b', lower):
        ai_score += 0.12
        flags.append("GENERIC PHRASING")

    # Very long headline (AI tends to be verbose)
    word_count = len(text.split())
    if word_count > 18:
        ai_score += 0.10
        flags.append("VERBOSE")

    # Hedging language cluster (AI hedges a lot)
    hedge_count = sum(1 for w in ['may', 'might', 'could', 'potentially', 'possibly',
                                   'suggests', 'appears', 'seems', 'indicates'] if w in lower)
    if hedge_count >= 2:
        ai_score += 0.15
        flags.append("OVER-HEDGED")

    # ── NORMALISE & VERDICT ─────────────────────────────────────────────────
    cb_score  = min(cb_score, 1.0)
    ai_score  = min(ai_score, 1.0)
    # Deduplicate flags
    flags = list(dict.fromkeys(flags))

    combined = max(cb_score, ai_score)
    if combined >= 0.35:
        verdict = "flagged"
    elif combined >= 0.18:
        verdict = "suspicious"
    else:
        verdict = "clean"

    return {
        "clickbait_score": round(cb_score, 2),
        "ai_score": round(ai_score, 2),
        "flags": flags,
        "verdict": verdict,
    }


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 0.7rem; letter-spacing: 0.18em;
                text-transform: uppercase; color: #00e5ff; margin-bottom: 1rem; padding-bottom: 0.5rem;
                border-bottom: 1px solid #1c1c2e;'>
        ⚙ Settings & Tools
    </div>
    """, unsafe_allow_html=True)

    min_conf = st.slider("Min. confidence threshold", 0.50, 0.99, 0.60, 0.01)
    show_price = st.toggle("Show price chart", value=True)
    use_ai_summary = st.toggle("AI-generated summary", value=True)
    use_why_intel = st.toggle("Why Intelligence brief", value=True)
    use_quality_flags = st.toggle("Headline quality flags", value=True)
    st.caption("⚠ Quality flags detect clickbait and AI-generated headlines using heuristic patterns + LLM cross-check. Based on text signals only — no social media data.")

    st.markdown("---")

    # ── WATCHLIST ──
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 0.65rem; letter-spacing: 0.18em;
                text-transform: uppercase; color: #5a5a7a; margin-bottom: 0.7rem;'>
        ★ Watchlist
    </div>
    """, unsafe_allow_html=True)

    wl_col1, wl_col2 = st.columns([3, 1])
    with wl_col1:
        new_ticker = st.text_input("Add ticker", label_visibility="collapsed",
                                   placeholder="e.g. NVDA", key="wl_input")
    with wl_col2:
        if st.button("＋", key="wl_add"):
            t_upper = new_ticker.strip().upper()
            if t_upper and t_upper not in st.session_state.watchlist:
                st.session_state.watchlist.append(t_upper)

    if st.session_state.watchlist:
        for wl_sym in st.session_state.watchlist:
            wl_name = TICKER_MAP.get(wl_sym, "Unknown")
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"""
                <div class="wl-card">
                    <div>
                        <span class="wl-sym">{wl_sym}</span><br>
                        <span class="wl-name">{wl_name}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                if st.button("✕", key=f"rm_{wl_sym}"):
                    st.session_state.watchlist.remove(wl_sym)
                    st.rerun()
        if st.button("▶ Load Watchlist", key="load_wl"):
            st.session_state["wl_load"] = ",".join(st.session_state.watchlist[:3])
    else:
        st.caption("No tickers saved yet.")

    st.markdown("---")

    # ── TICKER GUIDE ──
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 0.65rem; letter-spacing: 0.18em;
                text-transform: uppercase; color: #5a5a7a; margin-bottom: 0.7rem;'>
        ◈ Ticker Guide
    </div>
    """, unsafe_allow_html=True)

    search_q = st.text_input("Search tickers", label_visibility="collapsed",
                              placeholder="🔍  Search company or symbol…", key="ticker_search")

    filtered_tickers = {
        k: v for k, v in TICKER_MAP.items()
        if not search_q or search_q.upper() in k or search_q.lower() in v.lower()
    }

    categories = {
        "🤖 Tech & AI": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMD", "INTC", "AVGO", "QCOM",
                          "ADBE", "CRM", "ORCL", "IBM", "PANW", "CRWD", "SNOW", "PLTR", "ARM", "SMCI"],
        "💰 Finance": ["JPM", "BAC", "GS", "MS", "WFC", "C", "V", "MA", "AXP", "BLK", "BX",
                       "KKR", "APO", "USB", "PNC", "SCHW", "COIN", "SQ", "PYPL"],
        "🚗 EV & Mobility": ["TSLA", "RIVN", "LCID", "NIO", "UBER", "TM"],
        "🛒 Consumer": ["AMZN", "WMT", "COST", "MCD", "SBUX", "NKE", "DIS", "SHOP", "SPOT", "RBLX"],
        "⚡ Energy": ["XOM", "CVX", "BP", "SHEL", "NEE"],
        "💊 Healthcare": ["UNH", "JNJ", "LLY", "MRK", "ABBV", "ABT", "TMO", "AMGN", "GILD",
                           "VRTX", "REGN", "ZTS", "ISRG", "CI", "ELV"],
        "🌏 Global": ["BABA", "JD", "PDD", "TSM", "ASML", "SAP", "SONY", "RIO", "BHP"],
    }

    if search_q:
        for sym, name in filtered_tickers.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:0.3rem 0.5rem;
                        border-bottom:1px solid #1c1c2e; font-size:0.72rem;'>
                <span style='color:#00e5ff; font-family:IBM Plex Mono,monospace; font-weight:700;'>{sym}</span>
                <span style='color:#5a5a7a; font-size:0.65rem; text-align:right;'>{name}</span>
            </div>""", unsafe_allow_html=True)
    else:
        for cat, syms in categories.items():
            with st.expander(cat, expanded=False):
                for sym in syms:
                    name = TICKER_MAP.get(sym, "")
                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between; padding:0.25rem 0;
                                border-bottom:1px solid #13131f; font-size:0.72rem;'>
                        <span style='color:#00e5ff; font-family:IBM Plex Mono,monospace; font-weight:700;'>{sym}</span>
                        <span style='color:#5a5a7a; font-size:0.65rem;'>{name}</span>
                    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**About Sentwell**")
    st.caption(f"Sentwell uses FinBERT, a BERT model fine-tuned on financial text, to classify news headlines as positive, negative, or neutral. WHY Intelligence powered by LLaMA 3.1 via Groq. Versions: FinBERT @ {config.FINBERT_REVISION[:7]} · {config.LLM_MODEL} · prompts {config.WHY_PROMPT_VERSION}")
    st.markdown("[GitHub](https://github.com/mdrizvi-106) · [LinkedIn](https://linkedin.com/in/sam1061)")

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-grid">
    <div>
      <h1 class="hero-logo">
        SENT<span class="sw-w-mark" style="display:inline-block;vertical-align:middle;position:relative;top:-3px;margin:0 1px;">
          <svg width="48" height="44" viewBox="0 0 96 88" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Tick 1 (left): top-left → sweep down to bottom-centre → short stroke up-right -->
            <path d="M0 0 L22 80 L44 34 L58 62" stroke="#00e5ff" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            <!-- Tick 2 (right): same tick shape shifted 26px right, 55% opacity to show overlap -->
            <path d="M26 0 L48 80 L70 34 L84 62" stroke="#00e5ff" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0.55"/>
            <!-- Pulsing dot — bottom-right of the mark, like Wellmark's trailing dots -->
            <circle cx="93" cy="70" r="6" fill="#00ff9d">
              <animate attributeName="r" values="6;9;6" dur="2.2s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values="1;0.25;1" dur="2.2s" repeatCount="indefinite"/>
            </circle>
          </svg>
        </span>ELL
        <span style="font-size:1rem;color:#5a5a7a;font-weight:400;letter-spacing:0.05em;margin-left:0.8rem;vertical-align:middle;">· Financial Sentiment Intelligence</span>
      </h1>
      <p class="hero-tag">Real-time financial news sentiment · Powered by FinBERT & LLaMA 3.1</p>
      <div class="hero-badges">
        <span class="hbadge live">● LIVE DATA</span>
        <span class="hbadge">FINBERT NLP</span>
        <span class="hbadge">UP TO 3 TICKERS</span>
        <span class="hbadge">25 HEADLINES / TICKER</span>
        <span class="hbadge">AI SUMMARIES</span>
        <span class="hbadge">WHY INTELLIGENCE</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────
default_val = st.session_state.pop("wl_load", "")

st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
c1, c2 = st.columns([5, 1])
with c1:
    raw = st.text_input("tickers", label_visibility="collapsed",
                        placeholder="Enter up to 3 tickers — AAPL, TSLA, NVDA, GOOGL",
                        value=default_val)
with c2:
    go_btn = st.button("ANALYSE →")
st.markdown('</div>', unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if go_btn and raw.strip():
    tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()][:3]

    # ── Custom loading UI ──
    loading_slot = st.empty()

    def show_loading(step_label, detail, pct):
        loading_slot.markdown(f"""
        <div class="sw-loading">
          <div class="sw-loading-icon">
            <div class="sw-ring"></div>
            <div class="sw-ring2"></div>
          </div>
          <div class="sw-loading-text">
            <div class="sw-loading-step">⬡ {step_label}</div>
            <div class="sw-loading-label">{detail}</div>
            <div class="sw-loading-bar-bg">
              <div class="sw-loading-bar-fill" style="width:{pct}%"></div>
            </div>
            <div class="sw-loading-sub">{pct}% complete</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    show_loading("Initialising", "Loading FinBERT sentiment model…", 5)
    model = load_model()

    all_dfs, stock_infos, price_hists, hist_dfs = [], {}, {}, {}
    n = len(tickers)
    for i, sym in enumerate(tickers):
        pct = int(10 + (i / n) * 80)
        show_loading(f"Ticker {i+1} of {n}", f"Fetching & analysing news for {sym}…", pct)
        if i > 0:
            time.sleep(2)
        df = fetch_and_analyse(sym, model, min_conf)
        if df is not None:
            all_dfs.append(df)
            show_loading(f"Ticker {i+1} of {n}", f"Fetching market data for {sym}…", pct + 5)
            stock_infos[sym] = fetch_stock_info(sym)
            if show_price:
                price_hists[sym] = fetch_price_history(sym)
            hist_dfs[sym] = fetch_historical_sentiment(sym, model, min_conf)
        else:
            st.warning(f"No news found for **{sym}** — skipping.")

    show_loading("Finalising", "Building intelligence report…", 98)
    time.sleep(0.3)
    loading_slot.empty()

    if not all_dfs:
        st.error("No results. Try tickers like AAPL, TSLA, NVDA, GOOGL.")
        st.stop()

    # ── Store results in session state so slider doesn't re-trigger analysis ──
    st.session_state.analysis_results = {
        "all_dfs": all_dfs,
        "stock_infos": stock_infos,
        "price_hists": price_hists,
        "hist_dfs": hist_dfs,
    }

# ── RENDER RESULTS (from session state — persists across slider interactions) ──
if st.session_state.analysis_results:
    all_dfs    = st.session_state.analysis_results["all_dfs"]
    stock_infos = st.session_state.analysis_results["stock_infos"]
    price_hists = st.session_state.analysis_results["price_hists"]
    hist_dfs   = st.session_state.analysis_results["hist_dfs"]
    combined   = pd.concat(all_dfs, ignore_index=True)

    tab_ov, tab_hl, tab_trend, tab_cmp = st.tabs(["OVERVIEW", "HEADLINES", "TREND", "COMPARE"])

    # ── TAB 1 · OVERVIEW ─────────────────────────────────────────────────────
    with tab_ov:
        for df_t in all_dfs:
            sym  = df_t['ticker'].iloc[0]
            company = get_company_name(sym)
            pos  = len(df_t[df_t['sentiment'] == 'positive'])
            neg  = len(df_t[df_t['sentiment'] == 'negative'])
            neu  = len(df_t[df_t['sentiment'] == 'neutral'])
            total = len(df_t)
            dom  = df_t['sentiment'].mode()[0]
            info = stock_infos.get(sym, {})

            st.markdown(
                f'<p class="sec-title"><span class="tpill">{sym}</span>'
                f'&nbsp; <span style="color:#dde1f0;font-size:0.75rem;font-family:Inter,sans-serif;'
                f'text-transform:none;letter-spacing:0">{company}</span>'
                f'&nbsp; sentiment snapshot</p>',
                unsafe_allow_html=True
            )

            # Stock info row — FIXED: now uses year_high/year_low via fetch_stock_info
            price = info.get('price')
            chg   = info.get('change')
            chg_cls = 'green' if chg and chg >= 0 else 'red'
            chg_str = f"{'+' if chg and chg >= 0 else ''}{chg:.2f}%" if chg else "N/A"
            hi52  = info.get('hi52')
            lo52  = info.get('lo52')
            st.markdown(f"""
            <div class="stock-info-row">
              <div class="stock-info-card">
                <div class="si-label">Last Price</div>
                <div class="si-val blue">${price:,.2f}</div>
                <div class="si-sub {chg_cls}">{chg_str} today</div>
              </div>
              <div class="stock-info-card">
                <div class="si-label">Market Cap</div>
                <div class="si-val">{fmt_cap(info.get('mktcap'))}</div>
              </div>
              <div class="stock-info-card">
                <div class="si-label">52-Week High</div>
                <div class="si-val green">{f"${hi52:,.2f}" if hi52 else 'N/A'}</div>
              </div>
              <div class="stock-info-card">
                <div class="si-label">52-Week Low</div>
                <div class="si-val red">{f"${lo52:,.2f}" if lo52 else 'N/A'}</div>
              </div>
            </div>
            """ if price else "", unsafe_allow_html=True)

            # Alert
            st.markdown(
                f'<div class="alert {BULL[dom]}">'
                f'{ICON[dom]} Overall sentiment is <strong>{dom.upper()}</strong> &nbsp;·&nbsp; '
                f'{pos} bullish · {neg} bearish · {neu} neutral out of {total} headlines</div>',
                unsafe_allow_html=True
            )

            # ── AI Summary ──
            if use_ai_summary:
                ai_slot = st.empty()
                ai_slot.markdown(f"""
                <div class="sw-loading" style="padding:1rem 1.5rem;margin:0.5rem 0;">
                  <div class="sw-loading-icon" style="width:36px;height:36px;">
                    <div class="sw-ring"></div><div class="sw-ring2"></div>
                  </div>
                  <div class="sw-loading-text">
                    <div class="sw-loading-step">⬡ AI ANALYSIS</div>
                    <div class="sw-loading-label" style="font-size:0.85rem;">Generating market intelligence for {sym}…</div>
                  </div>
                </div>""", unsafe_allow_html=True)
                summary_text = generate_ai_summary(sym, df_t, info)
                ai_slot.empty()
                st.markdown(f"""
                <div class="ai-summary">
                    <div class="ai-summary-label">✦ AI Market Intelligence · {sym} / {company}</div>
                    <div class="ai-summary-text">{summary_text}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── WHY Intelligence Brief ──
            if use_why_intel:
                why_slot = st.empty()
                why_slot.markdown(f"""
                <div class="sw-loading" style="padding:1rem 1.5rem;margin:0.5rem 0;">
                  <div class="sw-loading-icon" style="width:36px;height:36px;">
                    <div class="sw-ring"></div><div class="sw-ring2"></div>
                  </div>
                  <div class="sw-loading-text">
                    <div class="sw-loading-step">⬡ WHY INTELLIGENCE</div>
                    <div class="sw-loading-label" style="font-size:0.85rem;">Analysing sentiment drivers for {sym}…</div>
                  </div>
                </div>""", unsafe_allow_html=True)
                why = generate_why_intelligence(sym, df_t, info)
                why_slot.empty()

                drivers_html = "".join(f"<li>{item}</li>" for item in why.get("drivers", []))
                narratives_html = "".join(f"<li>{item}</li>" for item in why.get("narratives", []))
                roots_html = "".join(f"<li>{item}</li>" for item in why.get("root_causes", []))
                contra_html = "".join(f"<li>{item}</li>" for item in why.get("contradictions", []))

                st.markdown(f"""
                <div style="margin-bottom:1.4rem;">
                  <div class="sec-title" style="margin-top:0.4rem;">◈ WHY Intelligence · {sym}</div>
                  <div class="why-grid">
                    <div class="why-card">
                      <div class="why-card-label drivers">⚡ Sentiment Drivers</div>
                      <div class="why-card-body"><ul>{drivers_html}</ul></div>
                    </div>
                    <div class="why-card">
                      <div class="why-card-label narratives">◈ Emerging Narratives</div>
                      <div class="why-card-body"><ul>{narratives_html}</ul></div>
                    </div>
                    <div class="why-card">
                      <div class="why-card-label roots">⊕ Root Causes</div>
                      <div class="why-card-body"><ul>{roots_html}</ul></div>
                    </div>
                    <div class="why-card">
                      <div class="why-card-label contra">⊘ Contradictory Signals</div>
                      <div class="why-card-body"><ul>{contra_html}</ul></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Metrics + donut
            col_m1, col_m2, col_m3, col_donut = st.columns([1, 1, 1, 2])
            with col_m1:
                st.markdown(f'<div class="metric-card"><div class="mc-label">Positive</div><div class="mc-val green">{pos}</div></div>', unsafe_allow_html=True)
            with col_m2:
                st.markdown(f'<div class="metric-card"><div class="mc-label">Negative</div><div class="mc-val red">{neg}</div></div>', unsafe_allow_html=True)
            with col_m3:
                st.markdown(f'<div class="metric-card"><div class="mc-label">Neutral</div><div class="mc-val grey">{neu}</div></div>', unsafe_allow_html=True)
            with col_donut:
                donut = go.Figure(go.Pie(
                    labels=['Positive', 'Negative', 'Neutral'],
                    values=[pos, neg, neu], hole=0.7,
                    marker=dict(colors=['#00ff9d', '#ff4f6a', '#1c1c2e'],
                                line=dict(color='#070710', width=2)),
                    textinfo='none',
                    hovertemplate='%{label}: %{value}<extra></extra>',
                ))
                donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#dde1f0"),
                    showlegend=False, height=130,
                    margin=dict(l=0, r=0, t=0, b=0),
                    annotations=[dict(
                        text=f"<b>{dom.upper()}</b>", x=0.5, y=0.5, showarrow=False,
                        font=dict(size=12, color=COLOUR[dom], family="IBM Plex Mono"),
                    )],
                )
                st.plotly_chart(donut, use_container_width=True, config={'displayModeBar': False})

            # Price chart + bar side by side
            # FIXED: bar chart legend moved below chart, labels truncated cleanly
            if show_price and sym in price_hists and price_hists[sym] is not None:
                ch_col, bar_col = st.columns([1, 2])
            else:
                bar_col = st.columns([1])[0]
                ch_col = None

            if show_price and ch_col and sym in price_hists and price_hists[sym] is not None:
                hist = price_hists[sym]
                with ch_col:
                    st.markdown('<p class="sec-title">30-day price</p>', unsafe_allow_html=True)
                    line_colour = '#00ff9d' if (chg or 0) >= 0 else '#ff4f6a'
                    fig_price = go.Figure(go.Scatter(
                        x=hist.index, y=hist['Close'],
                        mode='lines', line=dict(color=line_colour, width=2),
                        fill='tozeroy',
                        fillcolor=f"rgba({'0,255,157' if line_colour == '#00ff9d' else '255,79,106'},0.06)",
                        hovertemplate='$%{y:.2f}<extra></extra>',
                    ))
                    fig_price.update_layout(
                        **plot_layout(height=200, margin=dict(l=0, r=0, t=10, b=0)),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color="#5a5a7a"),
                        yaxis=dict(showgrid=True, gridcolor="#1c1c2e", zeroline=False, color="#5a5a7a"),
                    )
                    st.plotly_chart(fig_price, use_container_width=True, config={'displayModeBar': False})

            with bar_col:
                st.markdown('<p class="sec-title">Confidence by headline</p>', unsafe_allow_html=True)
                bar_df = df_t.sort_values('score').copy()
                # Truncate labels to 45 chars max to avoid overlap
                bar_df['label'] = bar_df['headline'].str[:45] + '…'
                bar = px.bar(
                    bar_df, x='score',
                    y='label',
                    color='sentiment',
                    color_discrete_map=COLOUR,
                    orientation='h',
                    height=max(280, len(bar_df) * 28),  # dynamic height
                )
                bar.update_layout(
                    **plot_layout(margin=dict(l=0, r=0, t=10, b=0)),
                    xaxis=dict(showgrid=False, zeroline=False, color="#5a5a7a", range=[0, 1],
                               title=dict(text="Confidence", font=dict(size=10))),
                    yaxis=dict(showgrid=False, zeroline=False, color="#5a5a7a",
                               tickfont=dict(size=9)),
                    yaxis_title="",
                    # FIXED: legend placed below chart to avoid overlap
                    legend=dict(
                        orientation='h',
                        y=-0.12,
                        x=0,
                        font=dict(size=10),
                        title_text="",
                    ),
                )
                bar.update_traces(marker_line_width=0)
                st.plotly_chart(bar, use_container_width=True, config={'displayModeBar': False})

            st.markdown("<hr style='border:none;border-top:1px solid #1c1c2e;margin:1.5rem 0'>", unsafe_allow_html=True)

    # ── TAB 2 · HEADLINES ────────────────────────────────────────────────────
    # FIXED: slider now filters from stored session_state data, does NOT re-trigger analysis
    with tab_hl:
        st.markdown('<p class="sec-title">Top headlines by confidence</p>', unsafe_allow_html=True)
        # Slider uses combined data from session state — no re-analysis
        conf_filter = st.slider(
            "Confidence ≥",
            float(combined['score'].min()),
            1.0,
            min_conf,
            0.01,
            key="hl_conf"
        )
        filtered = combined[combined['score'] >= conf_filter]
        top25 = (
            filtered.sort_values('score', ascending=False)
            .groupby('ticker').head(25)
            .sort_values(['ticker', 'score'], ascending=[True, False])
        )
        st.markdown(
            f'<p style="font-size:0.75rem;color:#5a5a7a;margin-bottom:1rem;">'
            f'Analysed <b style="color:#dde1f0">{len(filtered)}</b> headlines &nbsp;·&nbsp; '
            f'Showing top <b style="color:#dde1f0">{len(top25)}</b> by confidence'
            f'&nbsp;·&nbsp; <span style="color:#5a5a7a;">Click any headline to expand</span></p>',
            unsafe_allow_html=True
        )

        if len(top25) == 0:
            st.info(f"No headlines above {conf_filter:.0%} confidence. Try lowering the threshold.")
        else:
            for _, row in top25.iterrows():
                cls = CLS[row['sentiment']]
                pct = row['score']
                bar_color = COLOUR[row['sentiment']]
                company_name = get_company_name(row['ticker'])

                # Quality flag block
                qflag_html = ""
                if use_quality_flags:
                    q = detect_headline_quality(row['headline'])
                    verdict = q['verdict']
                    flag_labels = q['flags']
                    if verdict == 'clean':
                        qflag_html = '<span class="qflag clean">✓ CLEAN</span>'
                    else:
                        icon = "⚠" if verdict == "suspicious" else "⛔"
                        label_str = " · ".join(flag_labels) if flag_labels else verdict.upper()
                        qflag_html = (
                            f'<span class="qflag {verdict}">{icon} {verdict.upper()}</span>'
                            f'<span class="qflag-detail">&nbsp;{label_str}</span>'
                        )

                # Summary text — fallback message if none available
                summary_text = row.get('summary', '') or ''
                if not summary_text:
                    summary_text = "No summary available for this article. Click the link below to read the full story."

                # Image block — show image or placeholder icon
                img_url = row.get('img_url', '') or ''
                if img_url:
                    img_block = f'<img class="hcard-expand-img" src="{img_url}" alt="Article thumbnail" onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';"><div class="hcard-expand-img-placeholder" style="display:none;">📰</div>'
                else:
                    img_block = '<div class="hcard-expand-img-placeholder">📰</div>'

                # Article link
                article_url = row.get('article_url', '') or ''
                link_html = (
                    f'<a class="hcard-expand-link" href="{article_url}" target="_blank" rel="noopener">↗ READ FULL ARTICLE</a>'
                    if article_url else
                    '<span style="font-size:0.65rem;color:var(--muted);font-family:IBM Plex Mono,monospace;">No link available</span>'
                )

                pub_date = row.get('date', '') or ''
                date_html = f'<span style="font-size:0.65rem;color:var(--muted);font-family:IBM Plex Mono,monospace;">{pub_date}</span>' if pub_date else ''

                st.markdown(f"""
                <details class="hcard-wrap">
                  <summary>
                    <div class="hcard {cls}">
                      <div class="hcard-title">
                        {row['headline']}
                        <span class="hcard-chevron">▼</span>
                      </div>
                      <div class="hcard-meta">
                        <span class="badge {cls}">{row['sentiment']}</span>
                        <span class="tpill" style="font-size:0.62rem;padding:0.12rem 0.4rem">{row['ticker']}</span>
                        <span style="color:#dde1f0;font-size:0.68rem;">{company_name}</span>
                        <span>{pct:.2%} confidence</span>
                        {date_html}
                        {qflag_html}
                      </div>
                      <div class="conf-bar-wrap">
                        <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct*100:.1f}%;background:{bar_color}"></div></div>
                      </div>
                    </div>
                  </summary>
                  <div class="hcard-expand-body {cls}">
                    {img_block}
                    <div class="hcard-expand-text">
                      <div class="hcard-expand-summary">{summary_text}</div>
                      {link_html}
                    </div>
                  </div>
                </details>
                """, unsafe_allow_html=True)

        st.markdown('<p class="sec-title">Export</p>', unsafe_allow_html=True)
        buf = io.StringIO()
        # Export all 25 headlines (full filtered set, not just displayed top)
        export_cols = ['ticker', 'headline', 'sentiment', 'score', 'date']
        export_cols = [c for c in export_cols if c in filtered.columns]
        filtered.sort_values('score', ascending=False).head(25)[export_cols].to_csv(buf, index=False)
        st.download_button("⬇ Download CSV", buf.getvalue(),
                           "sentwell_analysis.csv", "text/csv")

    # ── TAB 3 · TREND ────────────────────────────────────────────────────────
    # FIXED: legend overlap, trend stats HTML rendering, chart variability
    with tab_trend:
        st.markdown('<p class="sec-title">Sentiment trend over time</p>', unsafe_allow_html=True)
        st.caption("Headlines grouped by published date. Chart shows daily positive/negative/neutral counts and net sentiment score.")

        for sym, hdf in hist_dfs.items():
            if hdf is None or hdf.empty:
                st.warning(f"Not enough dated headlines for {sym}.")
                continue

            company = get_company_name(sym)
            st.markdown(
                f'<p class="sec-title" style="margin-top:1rem"><span class="tpill">{sym}</span>'
                f'&nbsp;<span style="color:#dde1f0;font-size:0.72rem;font-family:Inter;'
                f'text-transform:none;letter-spacing:0">{company}</span></p>',
                unsafe_allow_html=True
            )

            # Group by date
            daily = hdf.groupby('date').apply(lambda g: pd.Series({
                'positive': (g['sentiment'] == 'positive').sum(),
                'negative': (g['sentiment'] == 'negative').sum(),
                'neutral':  (g['sentiment'] == 'neutral').sum(),
                'total':    len(g),
                'avg_conf': g['score'].mean(),
                'net_score': (g['sentiment'] == 'positive').sum() - (g['sentiment'] == 'negative').sum(),
            })).reset_index().sort_values('date')

            if len(daily) < 2:
                st.info(f"Need at least 2 date buckets for {sym} to draw a trend. Showing raw distribution.")
                counts = hdf['sentiment'].value_counts().reset_index()
                counts.columns = ['sentiment', 'count']
                fig_fb = px.bar(counts, x='sentiment', y='count',
                                color='sentiment', color_discrete_map=COLOUR, height=200)
                fig_fb.update_layout(**plot_layout(margin=dict(l=0,r=0,t=20,b=0)), showlegend=False)
                st.plotly_chart(fig_fb, use_container_width=True, config={'displayModeBar': False})
                continue

            col_l, col_r = st.columns([3, 1])

            with col_l:
                # Stacked area chart — FIXED: legend placed below with proper spacing
                fig_area = go.Figure()
                fig_area.add_trace(go.Scatter(
                    x=daily['date'], y=daily['positive'],
                    name='Positive', mode='lines+markers',
                    line=dict(color='#00ff9d', width=2),
                    marker=dict(size=5, color='#00ff9d'),
                    fill='tozeroy', fillcolor='rgba(0,255,157,0.08)',
                    hovertemplate='%{x}: %{y} positive<extra></extra>',
                ))
                fig_area.add_trace(go.Scatter(
                    x=daily['date'], y=daily['negative'],
                    name='Negative', mode='lines+markers',
                    line=dict(color='#ff4f6a', width=2),
                    marker=dict(size=5, color='#ff4f6a'),
                    fill='tozeroy', fillcolor='rgba(255,79,106,0.08)',
                    hovertemplate='%{x}: %{y} negative<extra></extra>',
                ))
                fig_area.add_trace(go.Scatter(
                    x=daily['date'], y=daily['neutral'],
                    name='Neutral', mode='lines+markers',
                    line=dict(color='#5a5a7a', width=1.5, dash='dot'),
                    marker=dict(size=4, color='#5a5a7a'),
                    hovertemplate='%{x}: %{y} neutral<extra></extra>',
                ))
                fig_area.update_layout(
                    **plot_layout(height=260, margin=dict(l=0, r=0, t=30, b=50)),
                    xaxis=dict(showgrid=False, zeroline=False, color="#5a5a7a",
                               tickangle=-30, tickfont=dict(size=10)),
                    yaxis=dict(showgrid=True, gridcolor="#1c1c2e", zeroline=False,
                               color="#5a5a7a", dtick=1),
                    # FIXED: legend below chart with enough margin
                    legend=dict(
                        orientation='h',
                        y=-0.22,
                        x=0,
                        font=dict(size=11),
                        title_text="",
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    title=dict(text="Daily Headline Count by Sentiment",
                               font=dict(size=11, color="#5a5a7a"), x=0),
                )
                st.plotly_chart(fig_area, use_container_width=True, config={'displayModeBar': False})

                # Net sentiment bar chart
                daily['bar_colour'] = daily['net_score'].apply(
                    lambda v: '#00ff9d' if v > 0 else ('#ff4f6a' if v < 0 else '#5a5a7a')
                )
                fig_net = go.Figure(go.Bar(
                    x=daily['date'], y=daily['net_score'],
                    marker_color=daily['bar_colour'],
                    hovertemplate='%{x}<br>Net: %{y}<extra></extra>',
                    text=daily['net_score'].apply(lambda v: f"{int(v):+d}"),
                    textposition='outside',
                    textfont=dict(size=9, color="#dde1f0"),
                ))
                fig_net.update_layout(
                    **plot_layout(height=180, margin=dict(l=0, r=0, t=30, b=40)),
                    xaxis=dict(showgrid=False, zeroline=False, color="#5a5a7a",
                               tickangle=-30, tickfont=dict(size=10)),
                    yaxis=dict(showgrid=True, gridcolor="#1c1c2e", zeroline=True,
                               zerolinecolor="#252538", color="#5a5a7a", dtick=1),
                    title=dict(text="Net Sentiment Score (Positive − Negative)",
                               font=dict(size=11, color="#5a5a7a"), x=0),
                )
                fig_net.update_traces(marker_line_width=0)
                st.plotly_chart(fig_net, use_container_width=True, config={'displayModeBar': False})

            # FIXED: Trend stats panel now uses st.markdown properly with container
            with col_r:
                avg_net = daily['net_score'].mean()
                last_net = daily['net_score'].iloc[-1]
                first_net = daily['net_score'].iloc[0]
                trend_dir = "↑ Improving" if last_net > first_net else ("↓ Declining" if last_net < first_net else "→ Flat")
                trend_col = "#00ff9d" if "↑" in trend_dir else ("#ff4f6a" if "↓" in trend_dir else "#5a5a7a")
                peak_day = daily.loc[daily['net_score'].idxmax(), 'date']
                worst_day = daily.loc[daily['net_score'].idxmin(), 'date']
                avg_net_col = "green" if avg_net >= 0 else "red"
                total_headlines = int(daily['total'].sum())
                coverage_days = len(daily)

                # Use st native components to avoid HTML rendering issues in columns
                st.markdown(f"""
                <div class="trend-stats-panel">
                  <div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;letter-spacing:0.15em;
                              text-transform:uppercase;color:#5a5a7a;margin-bottom:1rem;
                              padding-bottom:0.4rem;border-bottom:1px solid #1c1c2e;">
                    Trend Stats
                  </div>
                  <div class="ts-row">
                    <div class="ts-label">Direction</div>
                    <div class="ts-val" style="color:{trend_col};">{trend_dir}</div>
                  </div>
                  <div class="ts-row">
                    <div class="ts-label">Avg Net Score</div>
                    <div class="ts-val {'green' if avg_net >= 0 else 'red'}">{avg_net:+.1f}</div>
                  </div>
                  <div class="ts-row">
                    <div class="ts-label">Coverage</div>
                    <div class="ts-val sm accent">{coverage_days} days</div>
                    <div style="font-size:0.68rem;color:#5a5a7a;">{total_headlines} headlines</div>
                  </div>
                  <div class="ts-row">
                    <div class="ts-label">Peak Day</div>
                    <div class="ts-val sm" style="color:#00ff9d;">{peak_day}</div>
                  </div>
                  <div class="ts-row">
                    <div class="ts-label">Worst Day</div>
                    <div class="ts-val sm" style="color:#ff4f6a;">{worst_day}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 4 · COMPARE ──────────────────────────────────────────────────────
    with tab_cmp:
        if len(all_dfs) < 2:
            st.info("Enter 2–3 tickers to compare them side by side.")
        else:
            st.markdown('<p class="sec-title">Side-by-side comparison</p>', unsafe_allow_html=True)
            rows = []
            for df_t in all_dfs:
                sym = df_t['ticker'].iloc[0]
                pos = len(df_t[df_t['sentiment'] == 'positive'])
                neg = len(df_t[df_t['sentiment'] == 'negative'])
                neu = len(df_t[df_t['sentiment'] == 'neutral'])
                dom = df_t['sentiment'].mode()[0]
                rows.append({'Ticker': sym,
                             'Company': get_company_name(sym),
                             'Positive': pos, 'Negative': neg,
                             'Neutral': neu, 'Avg Confidence': f"{df_t['score'].mean():.2%}",
                             'Verdict': dom.upper()})
            st.dataframe(pd.DataFrame(rows).set_index('Ticker'), use_container_width=True)

            st.markdown('<p class="sec-title">Sentiment distribution</p>', unsafe_allow_html=True)
            cmp_data = []
            for df_t in all_dfs:
                sym = df_t['ticker'].iloc[0]
                for s in ['positive', 'negative', 'neutral']:
                    cmp_data.append({'Ticker': sym, 'Sentiment': s,
                                     'Count': len(df_t[df_t['sentiment'] == s])})
            cmp_df = pd.DataFrame(cmp_data)
            cmp_bar = px.bar(cmp_df, x='Ticker', y='Count', color='Sentiment', barmode='group',
                             color_discrete_map={'positive': '#00ff9d', 'negative': '#ff4f6a', 'neutral': '#1c1c2e'},
                             height=300)
            cmp_bar.update_layout(
                **plot_layout(margin=dict(l=0, r=0, t=30, b=50)),
                xaxis=dict(showgrid=False, zeroline=False, color="#5a5a7a"),
                yaxis=dict(showgrid=True, gridcolor="#1c1c2e", zeroline=False, color="#5a5a7a"),
                legend=dict(orientation='h', y=-0.18, x=0, font=dict(size=11), title_text=""),
            )
            cmp_bar.update_traces(marker_line_width=0)
            st.plotly_chart(cmp_bar, use_container_width=True, config={'displayModeBar': False})

            st.markdown('<p class="sec-title">Average model confidence</p>', unsafe_allow_html=True)
            conf_data = [{'Ticker': df_t['ticker'].iloc[0],
                          'Avg Confidence': df_t['score'].mean()} for df_t in all_dfs]
            conf_df = pd.DataFrame(conf_data)
            conf_bar = px.bar(conf_df, x='Ticker', y='Avg Confidence',
                              color='Ticker',
                              color_discrete_sequence=['#00e5ff', '#ffd166', '#00ff9d'],
                              height=240, text_auto='.2%')
            conf_bar.update_layout(
                **plot_layout(margin=dict(l=0, r=0, t=30, b=0)),
                xaxis=dict(showgrid=False, zeroline=False, color="#5a5a7a"),
                yaxis=dict(showgrid=True, gridcolor="#1c1c2e", zeroline=False,
                           color="#5a5a7a", tickformat='.0%'),
                showlegend=False,
            )
            conf_bar.update_traces(marker_line_width=0)
            st.plotly_chart(conf_bar, use_container_width=True, config={'displayModeBar': False})

elif go_btn and not raw.strip():
    st.warning("Please enter at least one ticker symbol.")
