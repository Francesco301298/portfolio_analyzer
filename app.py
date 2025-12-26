"""Portfolio Analyzer Pro - v7.1 with Full Deep-dive and Backtest"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import warnings
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import t as t_dist
import requests

from portfolio_analyzer import AdvancedPortfolioAnalyzer

from core.metrics import calculate_portfolio_metrics
from core.metrics import calculate_robust_metrics

from core.optimization import optimize_portfolio_weights
from core.optimization import run_walk_forward_analysis
from core.optimization import get_hrp_dendrogram_data
from core.optimization import cvar_optimization
from core.optimization import compute_portfolio_cvar
##from core.optimization import compute_portfolio_cvar_fast

from core.statistics import compute_autocorrelation
from core.statistics import invariance_test_ellipsoid
from core.statistics import ks_test
from core.statistics import fit_garch
from core.statistics import fit_locdisp_mlfp
from core.statistics import fit_dcc_t
from core.statistics import compute_flexible_probabilities
from core.statistics import extract_garch_residuals
from core.statistics import fit_var1
from core.statistics import cointegration_fp

from core.rebalancing import calculate_portfolio_with_rebalancing
from core.rebalancing import calculate_all_portfolios_with_costs

from core.backtesting import combinatorial_purged_cv
from core.backtesting import run_cpcv_backtest
from core.backtesting import compute_pbo

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

st.set_page_config(page_title="Portfolio Analyzer Pro", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #1a1a24;
    --accent-primary: #6366f1;
    --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-color: rgba(99,102,241,0.2);
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.main, .stApp { background: var(--bg-primary); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border-right: 1px solid var(--border-color);
}

.stButton > button {
    width: 100%;
    background: var(--accent-gradient) !important;
    color: #ffffff !important;
    padding: 0.875rem 1.5rem;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 12px;
    border: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.1);
    color: #ffffff !important;
}

.dashboard-card {
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    padding: 1.5rem;
    border-radius: 16px;
    margin: 0.5rem 0;
    border: 1px solid var(--border-color);
}

.dashboard-card h3 {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.75rem;
}

.dashboard-card ul, .dashboard-card ol, .dashboard-card p {
    color: var(--text-secondary);
    line-height: 1.6;
    font-size: 0.9rem;
}

[data-testid="stMetricValue"] {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 1.5rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.7rem !important;
}

h1 {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 2rem !important;
}

h2 {
    color: var(--text-primary) !important;
    font-weight: 700;
    font-size: 1.4rem !important;
    border-bottom: 2px solid;
    border-image: var(--accent-gradient) 1;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

h3, h4, h5 { color: var(--accent-primary) !important; font-weight: 600; }
p, li, span, label { color: var(--text-secondary); }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 0.4rem;
    gap: 0.25rem;
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary);
    font-weight: 600;
    border-radius: 8px;
    font-size: 0.85rem;
    padding: 0.5rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-gradient) !important;
    color: white !important;
}

.hero-section { text-align: center; padding: 1.5rem 0; }

.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-divider {
    width: 100px;
    height: 3px;
    background: var(--accent-gradient);
    margin: 1rem auto;
    border-radius: 2px;
}

.footer-section {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border-radius: 16px;
    margin-top: 2rem;
    border: 1px solid var(--border-color);
}

.styled-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border-radius: 12px;
    overflow: hidden;
    margin: 1rem 0;
    font-size: 0.85rem;
}

.styled-table th {
    background: var(--accent-gradient);
    color: white !important;
    padding: 10px 8px;
    text-align: left;
    font-weight: 600;
    font-size: 0.75rem;
}

.styled-table td {
    padding: 8px;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-secondary);
    font-size: 0.8rem;
}

.styled-table tr:hover { background: rgba(99,102,241,0.1); }

.stCheckbox label { color: var(--text-secondary) !important; }
.stAlert { border-radius: 8px; }

@media screen and (max-width: 640px) {
    .hero-title { font-size: 1.5rem !important; }
    .dashboard-card { padding: 1rem; }
    .styled-table { font-size: 0.7rem; }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
}
</style>
""", unsafe_allow_html=True)

# Yahoo Finance Search
@st.cache_data(ttl=300)
def search_yahoo_finance(query):
    if not query or len(query) < 1:
        return []
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search"
        params = {'q': query, 'quotesCount': 15, 'newsCount': 0, 'listsCount': 0}
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = []
            for quote in data.get('quotes', []):
                ticker = quote.get('symbol', '')
                name = quote.get('shortname') or quote.get('longname') or ticker
                exchange = quote.get('exchange', '')
                quote_type = quote.get('quoteType', '')
                if quote_type in ['EQUITY', 'ETF', 'INDEX', 'CRYPTOCURRENCY', 'MUTUALFUND', 'CURRENCY']:
                    results.append({'symbol': ticker, 'name': name, 'exchange': exchange, 'type': quote_type})
            return results
    except:
        pass
    return []

# Ticker Database
TICKER_INFO = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet (Google)", "AMZN": "Amazon",
    "META": "Meta (Facebook)", "NVDA": "NVIDIA", "AMD": "AMD", "INTC": "Intel",
    "CRM": "Salesforce", "ADBE": "Adobe", "NFLX": "Netflix", "PYPL": "PayPal",
    "QCOM": "Qualcomm", "AVGO": "Broadcom", "PLTR": "Palantir", "CRWD": "CrowdStrike",
    "SNOW": "Snowflake", "NET": "Cloudflare", "CSCO": "Cisco", "ORCL": "Oracle",
    "IBM": "IBM", "SHOP": "Shopify", "SQ": "Block (Square)", "UBER": "Uber",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo", "C": "Citigroup",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "BLK": "BlackRock", "SCHW": "Charles Schwab",
    "AXP": "American Express", "V": "Visa", "MA": "Mastercard", "COF": "Capital One",
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "PFE": "Pfizer", "ABBV": "AbbVie",
    "TMO": "Thermo Fisher", "ABT": "Abbott Labs", "MRK": "Merck", "LLY": "Eli Lilly",
    "AMGN": "Amgen", "BMY": "Bristol-Myers Squibb", "GILD": "Gilead Sciences", "ISRG": "Intuitive Surgical",
    "TSLA": "Tesla", "HD": "Home Depot", "MCD": "McDonald's", "NKE": "Nike",
    "SBUX": "Starbucks", "LOW": "Lowe's", "TGT": "Target", "COST": "Costco",
    "WMT": "Walmart", "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "DIS": "Disney", "ABNB": "Airbnb", "BKNG": "Booking Holdings",
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips", "SLB": "Schlumberger",
    "EOG": "EOG Resources", "MPC": "Marathon Petroleum", "OXY": "Occidental Petroleum",
    "DVN": "Devon Energy", "HAL": "Halliburton", "KMI": "Kinder Morgan",
    "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ Composite",
    "^RUT": "Russell 2000", "^FTSE": "FTSE 100", "^GDAXI": "DAX (Germany)",
    "^N225": "Nikkei 225", "^STOXX50E": "Euro Stoxx 50",
    "SPY": "SPDR S&P 500 ETF", "VOO": "Vanguard S&P 500", "VTI": "Vanguard Total Market",
    "QQQ": "Invesco NASDAQ 100", "IWM": "iShares Russell 2000", "VEA": "Vanguard FTSE Developed",
    "VWO": "Vanguard FTSE Emerging", "EEM": "iShares MSCI Emerging", "EFA": "iShares MSCI EAFE",
    "ACWI": "iShares MSCI ACWI", "XLK": "Technology Select SPDR", "XLV": "Health Care Select SPDR",
    "XLF": "Financial Select SPDR", "XLE": "Energy Select SPDR", "XLI": "Industrial Select SPDR",
    "XLY": "Consumer Discret. SPDR", "XLP": "Consumer Staples SPDR", "XLU": "Utilities Select SPDR",
    "VNQ": "Vanguard Real Estate", "ARKK": "ARK Innovation ETF",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana", "ADA-USD": "Cardano", "XRP-USD": "Ripple (XRP)",
    "DOGE-USD": "Dogecoin", "DOT-USD": "Polkadot", "AVAX-USD": "Avalanche",
    "TLT": "iShares 20+ Year Treasury", "IEF": "iShares 7-10 Year Treasury",
    "SHY": "iShares 1-3 Year Treasury", "AGG": "iShares Core US Aggregate",
    "BND": "Vanguard Total Bond", "LQD": "iShares Investment Grade Corp",
    "HYG": "iShares High Yield Corp", "TIP": "iShares TIPS Bond",
    "GLD": "SPDR Gold Shares", "IAU": "iShares Gold Trust", "SLV": "iShares Silver Trust",
    "GDX": "VanEck Gold Miners", "GDXJ": "VanEck Junior Gold Miners",
    "USO": "United States Oil Fund", "UNG": "United States Natural Gas",
    "DBA": "Invesco DB Agriculture", "DBC": "Invesco DB Commodity Index",
    "PDBC": "Invesco Optimum Yield Diversified Commodity",
    "CPER": "United States Copper Index", "WEAT": "Teucrium Wheat Fund",
    "CORN": "Teucrium Corn Fund", "SOYB": "Teucrium Soybean Fund",
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
    "USDCHF=X": "USD/CHF", "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD",
    "NZDUSD=X": "NZD/USD", "EURGBP=X": "EUR/GBP", "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY",
}

# Single Stocks organizzate per settore
SINGLE_STOCKS = {
    "🇺🇸 US Tech": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AMD","INTC","CRM","ADBE","NFLX","PYPL","PLTR","CRWD","SNOW","NET"],
    "🇺🇸 US Finance": ["JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP","V","MA","COF"],
    "🇺🇸 US Healthcare": ["JNJ","UNH","PFE","ABBV","TMO","ABT","MRK","LLY","AMGN","BMY","GILD","ISRG"],
    "🇺🇸 US Consumer": ["TSLA","HD","MCD","NKE","SBUX","LOW","TGT","COST","WMT","PG","KO","PEP"],
    "🇺🇸 US Energy": ["XOM","CVX","COP","SLB","EOG","MPC","OXY","DVN","HAL","KMI"],
}

# Forex con formattazione display
FOREX_TICKERS = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "USDCHF=X": "USD/CHF",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "NZDUSD=X": "NZD/USD",
    "EURGBP=X": "EUR/GBP",
    "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY",
}

TICKER_DATABASE = {
    "📊 Indices": ["^GSPC","^DJI","^IXIC","^RUT","^FTSE","^GDAXI","^N225","^STOXX50E"],
    "📈 ETF Broad": ["SPY","VOO","VTI","QQQ","IWM","VEA","VWO","EEM","EFA","ACWI"],
    "📈 ETF Sector": ["XLK","XLV","XLF","XLE","XLI","XLY","XLP","XLU","VNQ"],
    "💎 Crypto": ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","ADA-USD","XRP-USD","DOGE-USD"],
    "🏛️ Bonds": ["TLT","IEF","SHY","AGG","BND","LQD","HYG","TIP"],
    "🛢️ Commodities": ["GLD","IAU","SLV","GDX","GDXJ","USO","UNG","DBA","DBC","PDBC","CPER","WEAT","CORN","SOYB"],
    "💱 Forex": list(FOREX_TICKERS.keys()),
}
def get_display_name(ticker):
    return TICKER_INFO.get(ticker, ticker)

def update_ticker_info(ticker, name):
    if ticker not in TICKER_INFO:
        TICKER_INFO[ticker] = name

CHART_COLORS = ['#FF6B6B','#4ECDC4','#FFE66D','#95E1D3','#F38181','#AA96DA','#FCBAD3','#A8D8EA','#FF9F43','#6C5CE7']

def apply_plotly_theme(fig):
    fig.update_layout(
        title_text="", title=dict(text=""),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', family='Inter', size=12),
        legend=dict(bgcolor='rgba(26,26,36,0.95)', bordercolor='rgba(99,102,241,0.5)', borderwidth=1, font=dict(color='#F8FAFC', size=11)),
        xaxis=dict(gridcolor='rgba(99,102,241,0.15)', linecolor='rgba(99,102,241,0.3)', tickfont=dict(color='#E2E8F0', size=10)),
        yaxis=dict(gridcolor='rgba(99,102,241,0.15)', linecolor='rgba(99,102,241,0.3)', tickfont=dict(color='#E2E8F0', size=10)),
        hoverlabel=dict(bgcolor='#1E1E2E', bordercolor='#6366F1', font=dict(color='#F8FAFC', size=11)),
        margin=dict(t=20, b=40, l=50, r=20)
    )
    return fig

def create_styled_table(df, title=""):
    html = '<div style="overflow-x:auto;"><table class="styled-table">'
    if title:
        html += f'<caption style="color:#94a3b8;padding:8px;font-size:0.9rem;font-weight:600">{title}</caption>'
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for val in row:
            html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html

    
# Session State
defaults = {
    'analyzer': None, 'analysis_complete': False, 'selected_tickers': [],
    'benchmark': '^GSPC', 'use_benchmark': True, 'benchmark_returns': None,
    'run_analysis': False, 'alerts': [], 'yf_search_results': [], 'yf_selected': [],
    # NEW: Transaction costs defaults
    'portfolios_with_costs': None, 'cost_config': None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
# Part 2: Header and Sidebar

# Header
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">📊 PORTFOLIO ANALYZER PRO</h1>
    <p class="hero-subtitle">Advanced Multi-Asset Optimization System</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    selection_method = st.radio("Selection Method", ["📋 Database", "🔍 Yahoo Search", "✍️ Manual"], horizontal=True, key="sel_method")
    
    selected_symbols = []
    
    if selection_method == "📋 Database":
        st.markdown("#### 📚 Select from Categories")
        search_filter = st.text_input("🔍 Filter", placeholder="Filter database...", key="db_filter")
        
        # Single Stocks con struttura gerarchica
        with st.expander("📈 Single Stocks", expanded=False):
            for sector, tickers in SINGLE_STOCKS.items():
                with st.expander(sector, expanded=False):
                    if search_filter:
                        filtered = [t for t in tickers if search_filter.upper() in t.upper() or search_filter.upper() in get_display_name(t).upper()]
                    else:
                        filtered = tickers
                    
                    if filtered:
                        cols = st.columns(2)
                        for idx, ticker in enumerate(filtered):
                            with cols[idx % 2]:
                                if st.checkbox(get_display_name(ticker), key=f"db_stock_{sector}_{ticker}"):
                                    selected_symbols.append(ticker)
        
        # Altre categorie (Indices, ETFs, Crypto, Bonds, Commodities, Forex)
        for category, tickers in TICKER_DATABASE.items():
            with st.expander(category, expanded=False):
                if search_filter:
                    filtered = [t for t in tickers if search_filter.upper() in t.upper() or search_filter.upper() in get_display_name(t).upper()]
                else:
                    filtered = tickers
                
                if filtered:
                    cols = st.columns(2)
                    for idx, ticker in enumerate(filtered):
                        with cols[idx % 2]:
                            if st.checkbox(get_display_name(ticker), key=f"db_{category}_{ticker}"):
                                selected_symbols.append(ticker)    
    
    elif selection_method == "🔍 Yahoo Search":
        st.markdown("#### 🌐 Search Yahoo Finance")
        st.caption("Search any stock, ETF, index, crypto worldwide!")
        
        yf_query = st.text_input("🔍 Search", placeholder="e.g., Tesla, Bitcoin, FTSE...", key="yf_search_input")
        
        if st.button("🔎 Search", key="yf_search_btn", use_container_width=True):
            if yf_query and len(yf_query) >= 1:
                with st.spinner("Searching..."):
                    results = search_yahoo_finance(yf_query)
                    st.session_state.yf_search_results = results
        
        if st.session_state.yf_search_results:
            st.markdown("---")
            st.markdown(f"**Results ({len(st.session_state.yf_search_results)})**")
            
            for i, result in enumerate(st.session_state.yf_search_results):
                symbol = result['symbol']
                name = result['name']
                type_emoji = {"EQUITY": "📈", "ETF": "📊", "INDEX": "📉", "CRYPTOCURRENCY": "💎"}.get(result['type'], "📄")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{type_emoji} **{symbol}**")
                    st.caption(f"{name[:25]}...")
                with col2:
                    if st.button("➕", key=f"add_yf_{i}_{symbol}"):
                        if symbol not in st.session_state.yf_selected:
                            st.session_state.yf_selected.append(symbol)
                            update_ticker_info(symbol, name)
                            st.rerun()
        
        if st.session_state.yf_selected:
            st.markdown("#### ✅ Selected")
            for symbol in st.session_state.yf_selected:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"• **{symbol}**")
                with col2:
                    if st.button("❌", key=f"rem_yf_{symbol}"):
                        st.session_state.yf_selected.remove(symbol)
                        st.rerun()
            selected_symbols = st.session_state.yf_selected.copy()
    
    else:
        st.markdown("#### ✍️ Manual Entry")
        manual_input = st.text_area("Enter tickers", height=100, placeholder="AAPL, MSFT, GOOGL", key="manual_tickers")
        if manual_input:
            import re
            selected_symbols = [s.strip().upper() for s in re.split('[,\n]', manual_input) if s.strip()]
    
    if selection_method != "🔍 Yahoo Search" and st.session_state.yf_selected:
        for sym in st.session_state.yf_selected:
            if sym not in selected_symbols:
                selected_symbols.append(sym)
    
    selected_symbols = list(dict.fromkeys(selected_symbols))
    # ============ CUSTOM WEIGHTS SECTION ============
    st.markdown("---")
    st.markdown("#### ⚖️ Custom Portfolio")
    
    enable_custom_weights = st.checkbox(
        "Define your own weights",
        value=False,
        key="enable_custom_weights",
        help="Create a custom portfolio with your chosen allocation"
    )
    
    custom_weights_valid = False
    custom_weights_dict = {}
    
    if enable_custom_weights and selected_symbols:
        st.caption("Assign weights to each asset (must sum to 100%)")
        
        # Initialize weights in session state if needed
        if 'custom_weights' not in st.session_state:
            st.session_state.custom_weights = {}
        
        # Clean up weights for removed assets
        st.session_state.custom_weights = {
            k: v for k, v in st.session_state.custom_weights.items() 
            if k in selected_symbols
        }
        
        # Default: equal weight for new assets
        default_weight = 100.0 / len(selected_symbols)
        
        # Weight input method selection
        weight_method = st.radio(
            "Input method",
            ["Sliders", "Manual entry"],
            horizontal=True,
            key="weight_input_method"
        )
        
        total_weight = 0.0
        temp_weights = {}
        
        if weight_method == "Sliders":
            for ticker in selected_symbols:
                current_val = st.session_state.custom_weights.get(ticker, default_weight)
                weight = st.slider(
                    f"{get_display_name(ticker)}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(current_val),
                    step=0.5,
                    key=f"weight_slider_{ticker}",
                    format="%.1f%%"
                )
                temp_weights[ticker] = weight
                total_weight += weight
        else:  # Manual entry
            cols = st.columns(2)
            for idx, ticker in enumerate(selected_symbols):
                current_val = st.session_state.custom_weights.get(ticker, default_weight)
                with cols[idx % 2]:
                    weight = st.number_input(
                        f"{get_display_name(ticker)}",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(current_val),
                        step=0.5,
                        key=f"weight_input_{ticker}",
                        format="%.1f"
                    )
                    temp_weights[ticker] = weight
                    total_weight += weight
        
        # Update session state
        st.session_state.custom_weights = temp_weights
        
        # Validation and display
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if abs(total_weight - 100.0) < 0.01:
                st.success(f"✅ Total: {total_weight:.1f}%")
                custom_weights_valid = True
                custom_weights_dict = temp_weights.copy()
            elif total_weight > 100.0:
                st.error(f"❌ Total: {total_weight:.1f}% (exceeds 100%)")
            else:
                st.warning(f"⚠️ Total: {total_weight:.1f}% (below 100%)")
        
        with col2:
            # Quick actions
            if st.button("Reset to Equal", key="reset_weights", use_container_width=True):
                eq_weight = 100.0 / len(selected_symbols)
                st.session_state.custom_weights = {t: eq_weight for t in selected_symbols}
                st.rerun()
        
        # Show allocation preview
        if custom_weights_valid:
            with st.expander("Preview allocation", expanded=False):
                for ticker, weight in sorted(temp_weights.items(), key=lambda x: -x[1]):
                    if weight > 0:
                        bar_width = int(weight / 2)  # Scale for display
                        bar = "█" * bar_width
                        st.markdown(f"**{get_display_name(ticker)}**: {weight:.1f}% {bar}")
    
    elif enable_custom_weights and not selected_symbols:
        st.info("👆 Select assets first to define weights")
    
    # Store validation result for later use
    st.session_state.custom_weights_valid = custom_weights_valid
    st.session_state.custom_weights_dict = custom_weights_dict    
    st.markdown("---")
    st.markdown("#### 📅 Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2018, 1, 1), min_value=datetime(2000, 1, 1), key="start")
    with col2:
        end_date = st.date_input("End Date", value=datetime.now(), key="end")
    
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1, key="rf")
    window_years = st.slider("Rolling Window (years)", 1, 10, 3, key="window")
    
    # ============ NEW: Transaction Costs & Rebalancing ============
    st.markdown("#### 💰 Transaction Costs")
    
    enable_costs = st.checkbox("Enable transaction costs", value=True, key="enable_costs")
    
    if enable_costs:
        transaction_cost_bps = st.slider(
            "Cost per trade (bps)", 
            min_value=0, 
            max_value=100, 
            value=10, 
            step=1,
            key="tx_cost",
            help="Basis points charged per transaction. 10 bps = 0.10%"
        )
        st.caption("💡 Typical: 5-10 bps for ETFs, 10-50 bps for single stocks")
    else:
        transaction_cost_bps = 0
    
    st.markdown("#### 🔄 Rebalancing Strategy")
    
    rebalance_type = st.radio(
        "Rebalancing method",
        ["Buy & Hold","Threshold-based","Calendar-based"],
        horizontal=True,
        key="rebal_type"
    )
    
    if rebalance_type == "Calendar-based":
        rebalance_freq = st.selectbox(
            "Frequency",
            options=["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
            index=2,  # Monthly default
            key="rebal_freq"
        )
        rebalance_freq_map = {
            "Daily": "D", "Weekly": "W", "Monthly": "M", 
            "Quarterly": "Q", "Annually": "A"
        }
        rebalance_freq_code = rebalance_freq_map[rebalance_freq]
        rebalance_threshold = None
        
    elif rebalance_type == "Threshold-based":
        rebalance_threshold = st.slider(
            "Rebalance when drift exceeds (%)",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="rebal_thresh"
        ) / 100
        rebalance_freq_code = None
        st.caption("💡 Rebalances only when any asset drifts beyond target by this %")
        
    else:  # Buy & Hold
        rebalance_freq_code = None
        rebalance_threshold = None
        st.caption("💡 No rebalancing - weights drift freely based on performance")
    
    # Store in session state for access during analysis    
    st.markdown("#### 📊 Benchmark")
    use_benchmark = st.checkbox("Compare with benchmark", value=True, key="use_bench")
    benchmark_options = ["^GSPC", "^DJI", "^IXIC", "SPY", "QQQ", "VTI"]
    benchmark_ticker = st.selectbox("Select Benchmark", benchmark_options, format_func=lambda x: get_display_name(x), disabled=not use_benchmark, key="bench_select")
    
    st.markdown("#### 🔔 Alerts")
    enable_alerts = st.checkbox("Enable risk alerts", value=False, key="alerts_on")  # OFF by default
    col1, col2 = st.columns(2)
    with col1:
        max_dd_threshold = st.number_input("Max DD (%)", 5, 50, 20, disabled=not enable_alerts, key="max_dd")
    with col2:
        min_sharpe_threshold = st.number_input("Min Sharpe", 0.0, 2.0, 0.5, 0.1, disabled=not enable_alerts, key="min_sharpe")
    
    st.markdown("---")
    st.info(f"📦 **{len(selected_symbols)}** assets selected")
    
    if selected_symbols:
        with st.expander("View selected", expanded=False):
            for t in selected_symbols[:15]:
                st.markdown(f"• {get_display_name(t)} ({t})")
            if len(selected_symbols) > 15:
                st.markdown(f"*...and {len(selected_symbols) - 15} more*")
    
    st.markdown("---")
    
    if st.button("🚀 START ANALYSIS", use_container_width=True, key="start_btn"):
        if len(selected_symbols) < 2:
            st.error("⚠️ Select at least 2 assets!")
        else:
            st.session_state.selected_tickers = selected_symbols
            st.session_state.run_analysis = True
            st.session_state.analyzer = None
            st.rerun()
    
    if st.session_state.analyzer is not None:
        if st.button("🔄 RESET", use_container_width=True, key="reset_btn"):
            st.session_state.analyzer = None
            st.session_state.analysis_complete = False
            st.session_state.selected_tickers = []
            st.session_state.benchmark_returns = None
            st.session_state.run_analysis = False
            st.session_state.yf_search_results = []
            st.session_state.yf_selected = []
            st.rerun()

# Landing Page
if not st.session_state.run_analysis and st.session_state.analyzer is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='dashboard-card'>
            <h3>📝 Create Your Portfolio</h3>
            <p>Select your assets from our database, search Yahoo Finance, or enter tickers manually. Configure date range, risk-free rate, and transaction costs.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='dashboard-card'>
            <h3>📊 Manage Your Portfolio</h3>
            <p>Analyze individual assets, explore correlations, review statistics, and monitor seasonality patterns. Deep-dive into risk drivers with main econometric models.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='dashboard-card'>
            <h3>⚡ Optimize Your Portfolio</h3>
            <p>Compare 7 optimization strategies, backtest with Walk-Forward validation, visualize the Efficient Frontier, and export your results.</p>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("## 🎲 Quick Examples")
    
    examples = {
        "🔥 Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        "💰 Dividend": ["JNJ", "PG", "KO", "PEP", "MCD", "WMT"],
        "🌍 Global ETFs": ["SPY", "EFA", "EEM", "GLD", "TLT", "VNQ"],
        "🛢️ Commodities": ["GLD", "SLV", "USO", "UNG", "DBC", "CPER"]
    }

    cols = st.columns(4)
    for idx, (name, tickers) in enumerate(examples.items()):
        with cols[idx]:
            if st.button(name, use_container_width=True, key=f"ex_{idx}"):
                st.session_state.selected_tickers = tickers
                st.session_state.run_analysis = True
                st.rerun()
            # Mostra lista asset con nomi completi, uno per riga
            asset_list_html = "<br>".join([f"• {get_display_name(t)}" for t in tickers])
            st.markdown(f"<p style='font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem;'>{asset_list_html}</p>", unsafe_allow_html=True)


# Part 3: Analysis Logic

if st.session_state.run_analysis or st.session_state.analyzer is not None:
    
    if st.session_state.run_analysis and st.session_state.analyzer is None:
        symbols = st.session_state.selected_tickers
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing...")
            progress_bar.progress(10)
            
            status_text.text("📥 Downloading from Yahoo Finance...")
            progress_bar.progress(20)
            
            analyzer = AdvancedPortfolioAnalyzer(
                symbols,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                risk_free_rate=risk_free_rate / 100
            )
            
            if YF_AVAILABLE:
                analyzer.download_data()
            else:
                st.error("❌ yfinance not available")
                st.stop()
            
            progress_bar.progress(50)
            status_text.text("📊 Calculating returns...")
            analyzer.calculate_returns()
            symbols = analyzer.symbols
            progress_bar.progress(65)
            
            status_text.text("💼 Optimizing portfolios (7 strategies)...")
            analyzer.build_all_portfolios()
            # ============ ADD CUSTOM PORTFOLIO IF VALID ============
            if st.session_state.get('custom_weights_valid', False) and st.session_state.get('custom_weights_dict'):
                custom_weights_dict = st.session_state.custom_weights_dict
                
                # Build weights array in correct order (matching analyzer.symbols)
                custom_weights_array = np.array([
                    custom_weights_dict.get(ticker, 0) / 100.0 
                    for ticker in analyzer.symbols
                ])
                
                # Calculate portfolio returns
                custom_returns = analyzer.returns.dot(custom_weights_array)
                
                # Calculate metrics
                custom_metrics = calculate_portfolio_metrics(custom_returns, risk_free_rate/100)
                
                # Calculate cumulative return
                custom_cumulative = (1 + custom_returns).cumprod()
                custom_cum_return = custom_cumulative.iloc[-1] - 1
                
                # Calculate max drawdown
                custom_rolling_max = custom_cumulative.expanding().max()
                custom_drawdown = (custom_cumulative - custom_rolling_max) / custom_rolling_max
                custom_max_dd = custom_drawdown.min()
                
                # Calculate Calmar ratio
                n_years = len(custom_returns) / 252
                custom_ann_return = (1 + custom_cum_return) ** (1 / n_years) - 1 if n_years > 0 else 0
                custom_calmar = custom_ann_return / abs(custom_max_dd) if custom_max_dd != 0 else 0
                
                # Add to portfolios dict
                analyzer.portfolios['custom'] = {
                    'name': 'Your Portfolio',
                    'weights': custom_weights_array,
                    'returns': custom_returns,
                    'cumulative_return': custom_cum_return,
                    'annualized_return': custom_metrics['return'],
                    'annualized_volatility': custom_metrics['volatility'],
                    'sharpe_ratio': custom_metrics['sharpe'],
                    'sortino_ratio': custom_metrics['sortino'],
                    'max_drawdown': custom_max_dd,
                    'calmar_ratio': custom_calmar
                }            
            progress_bar.progress(85)
            
            benchmark_returns = None
            if use_benchmark and benchmark_ticker and YF_AVAILABLE:
                try:
                    status_text.text(f"📈 Loading benchmark...")
                    benchmark_df = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
                    if not benchmark_df.empty:
                        if isinstance(benchmark_df.columns, pd.MultiIndex):
                            benchmark_prices = benchmark_df['Close'][benchmark_ticker]
                        else:
                            benchmark_prices = benchmark_df['Close']
                        benchmark_prices = pd.Series(benchmark_prices.values.flatten(), index=benchmark_df.index)
                        benchmark_returns = benchmark_prices.pct_change().dropna()
                except:
                    benchmark_returns = None
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            st.session_state.analyzer = analyzer
            st.session_state.benchmark_returns = benchmark_returns
            st.session_state.benchmark = benchmark_ticker
            st.session_state.use_benchmark = use_benchmark
            st.session_state.analysis_complete = True
            st.session_state.run_analysis = False
            
            # ============ NEW: Calculate portfolios with costs ============
            # Get values from widgets (use local variables, not session_state for widget-bound keys)
            _enable_costs = enable_costs if 'enable_costs' in dir() else st.session_state.get('enable_costs', True)
            _tx_cost = transaction_cost_bps if 'transaction_cost_bps' in dir() else st.session_state.get('tx_cost', 10)
            _rebal_freq = rebalance_freq_code if 'rebalance_freq_code' in dir() else None
            _rebal_thresh = rebalance_threshold if 'rebalance_threshold' in dir() else None
            
            if _enable_costs or _rebal_freq or _rebal_thresh:
                st.session_state.portfolios_with_costs = calculate_all_portfolios_with_costs(
                    analyzer=analyzer,
                    rebalance_freq=_rebal_freq,
                    rebalance_threshold=_rebal_thresh,
                    cost_bps=_tx_cost if _enable_costs else 0,
                    rf_rate=risk_free_rate / 100
                )
                # Store config for display
                st.session_state.cost_config = {
                    'enabled': _enable_costs,
                    'bps': _tx_cost,
                    'rebal_freq': _rebal_freq,
                    'rebal_thresh': _rebal_thresh
                }
            else:
                st.session_state.portfolios_with_costs = None
                st.session_state.cost_config = None            
            if enable_alerts:
                st.session_state.alerts = []
                for name, portfolio in analyzer.portfolios.items():
                    if abs(portfolio['max_drawdown']) > (max_dd_threshold / 100):
                        st.session_state.alerts.append({'type': 'warning', 'portfolio': portfolio['name'], 'message': f"High DD: {portfolio['max_drawdown']*100:.1f}%"})
                    if portfolio['sharpe_ratio'] < min_sharpe_threshold:
                        st.session_state.alerts.append({'type': 'info', 'portfolio': portfolio['name'], 'message': f"Low Sharpe: {portfolio['sharpe_ratio']:.2f}"})
            
            progress_bar.empty()
            status_text.empty()
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")
            st.session_state.run_analysis = False
            st.stop()
    
    if st.session_state.analyzer is not None:
        analyzer = st.session_state.analyzer
        benchmark_returns = st.session_state.benchmark_returns
        symbols = analyzer.symbols
        rf_rate = risk_free_rate / 100
        
        if st.session_state.alerts:
            with st.expander(f"🔔 {len(st.session_state.alerts)} Alerts", expanded=False):
                for alert in st.session_state.alerts:
                    if alert['type'] == 'warning':
                        st.warning(f"**{alert['portfolio']}**: {alert['message']}")
                    else:
                        st.info(f"**{alert['portfolio']}**: {alert['message']}")
        
        # KPI Dashboard
        st.markdown("## 📊 KPI Dashboard")
        portfolios_list = list(analyzer.portfolios.values())
        
        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            best_return = max(portfolios_list, key=lambda x: x['annualized_return'])
            st.metric("🏆 Best Return", f"{best_return['annualized_return']*100:.1f}%", delta=best_return['name'].split()[0])
        with kpi_cols[1]:
            best_sharpe = max(portfolios_list, key=lambda x: x['sharpe_ratio'])
            st.metric("⭐ Best Sharpe", f"{best_sharpe['sharpe_ratio']:.2f}", delta=best_sharpe['name'].split()[0])
        with kpi_cols[2]:
            min_vol = min(portfolios_list, key=lambda x: x['annualized_volatility'])
            st.metric("🛡️ Min Vol", f"{min_vol['annualized_volatility']*100:.1f}%", delta=min_vol['name'].split()[0])
        with kpi_cols[3]:
            best_dd = max(portfolios_list, key=lambda x: x['max_drawdown'])
            st.metric("📉 Min DD", f"{best_dd['max_drawdown']*100:.1f}%", delta=best_dd['name'].split()[0])
        with kpi_cols[4]:
            st.metric("📦 Strategies", f"{len(portfolios_list)}")
        
        st.markdown("---")
        
        # Tabs
        if st.session_state.use_benchmark and benchmark_returns is not None:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Overview", "💼 Portfolios", "📈 Performance", 
                "🔬 Deep-dive", "🧪 Backtest", "📐 Frontier", "🎯 Benchmark", "📥 Export"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Overview", "💼 Portfolios", "📈 Performance",
                "🔬 Deep-dive", "🧪 Backtest", "📐 Frontier", "📥 Export"
            ])
            tab8 = None            # TAB 1: OVERVIEW
        with tab1:
            st.markdown("### 📊 Overview")
            
            col1, col2 = st.columns([2.5, 1])
            
            with col1:
                st.markdown("#### 📈 Asset Performance (Base 100)")
                normalized = (analyzer.data / analyzer.data.iloc[0]) * 100
                fig = go.Figure()
                for i, ticker in enumerate(normalized.columns):
                    fig.add_trace(go.Scatter(
                        x=normalized.index, y=normalized[ticker], name=get_display_name(ticker), mode='lines',
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                        hovertemplate=f'<b>{get_display_name(ticker)}</b><br>Value: %{{y:.2f}}<extra></extra>'
                    ))
                fig.update_layout(height=450, hovermode='x unified', xaxis_title="Date", yaxis_title="Value (Base 100)",
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Statistics")
                final_values = (analyzer.data.iloc[-1] / analyzer.data.iloc[0] - 1) * 100
                
                st.markdown("**🔥 Top 3**")
                for ticker, perf in final_values.nlargest(3).items():
                    st.success(f"**{get_display_name(ticker)}**: +{perf:.1f}%")
                
                st.markdown("**❄️ Bottom 3**")
                for ticker, perf in final_values.nsmallest(3).items():
                    if perf < 0:
                        st.error(f"**{get_display_name(ticker)}**: {perf:.1f}%")
                    else:
                        st.warning(f"**{get_display_name(ticker)}**: +{perf:.1f}%")
                
                st.markdown("---")
                st.metric("📅 Days", f"{len(analyzer.data)}")
                st.metric("📦 Assets", f"{len(symbols)}")
            
            # ============ NEW: Asset Statistics Table ============
            st.markdown("#### 📋 Individual Asset Statistics")
            
            asset_stats_data = []
            for ticker in symbols:
                asset_returns = analyzer.returns[ticker]
                
                # Calculate metrics
                ann_return = asset_returns.mean() * 252
                ann_vol = asset_returns.std() * np.sqrt(252)
                sharpe = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else 0
                
                # Sortino
                downside = asset_returns[asset_returns < 0]
                downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
                sortino = (ann_return - rf_rate) / downside_std if downside_std > 0 else 0
                
                # Max Drawdown
                cumulative = (1 + asset_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_dd = drawdown.min()
                
                # Calmar
                calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
                
                asset_stats_data.append({
                    'Asset': get_display_name(ticker),
                    'Ticker': ticker,
                    'Return': f"{ann_return*100:.2f}%",
                    'Vol': f"{ann_vol*100:.2f}%",
                    'Sharpe': f"{sharpe:.3f}",
                    'Sortino': f"{sortino:.3f}",
                    'Max DD': f"{max_dd*100:.2f}%",
                    'Calmar': f"{calmar:.3f}"
                })
            
            # Sort by Sharpe ratio (numeric extraction for sorting)
            asset_stats_df = pd.DataFrame(asset_stats_data)
            asset_stats_df['Sharpe_num'] = asset_stats_df['Sharpe'].astype(float)
            asset_stats_df = asset_stats_df.sort_values('Sharpe_num', ascending=False).drop('Sharpe_num', axis=1)
            
            # Add rank
            asset_stats_df.insert(0, '#', range(1, len(asset_stats_df) + 1))
            
            st.markdown(create_styled_table(asset_stats_df, "Ranked by Sharpe Ratio"), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Correlation
            st.markdown("#### 🔥 Correlation Heatmap")
            corr_matrix = analyzer.returns.corr()
            corr_display = corr_matrix.copy()
            corr_display.index = [get_display_name(t) for t in corr_display.index]
            corr_display.columns = [get_display_name(t) for t in corr_display.columns]
            
            fig = px.imshow(corr_display, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig.update_layout(height=400)
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        
# TAB 2: PORTFOLIOS
        with tab2:
            st.markdown("### 💼 Portfolio Analysis")
            
            # ============================================================
            # SECTION 1: THEORETICAL PERFORMANCE (NO COSTS)
            # ============================================================
            st.markdown("""
            ## 📊 Theoretical Performance
            
            This section shows portfolio performance **before transaction costs and rebalancing friction**. 
            Think of this as the "best case scenario" - what each strategy would achieve in a frictionless world.
            
            Use this to understand the **pure characteristics** of each optimization approach.
            """)
            
            st.markdown("#### 🏆 Strategy Ranking")
            
            comparison_data = []
            for idx, (key, p) in enumerate(sorted(analyzer.portfolios.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True), 1):
                comparison_data.append({
                    '#': idx, 
                    'Strategy': p['name'], 
                    'Return': f"{p['annualized_return']*100:.2f}%",
                    'Volatility': f"{p['annualized_volatility']*100:.2f}%", 
                    'Sharpe': f"{p['sharpe_ratio']:.3f}",
                    'Sortino': f"{p['sortino_ratio']:.3f}", 
                    'Max DD': f"{p['max_drawdown']*100:.2f}%",
                    'Calmar': f"{p['calmar_ratio']:.3f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.markdown(create_styled_table(comparison_df, "Ranked by Sharpe Ratio (Theoretical)"), unsafe_allow_html=True)
            
            with st.expander("📖 Understanding the Metrics"):
                st.markdown("""
                | Metric | What it measures | Good values |
                |--------|------------------|-------------|
                | **Return** | Annualized average return | Higher is better |
                | **Volatility** | Annualized standard deviation of returns | Lower = more stable |
                | **Sharpe Ratio** | Return per unit of total risk | > 1.0 good, > 2.0 excellent |
                | **Sortino Ratio** | Return per unit of *downside* risk | Higher than Sharpe = positive skew |
                | **Max Drawdown** | Largest peak-to-trough decline | < 20% conservative, < 40% moderate |
                | **Calmar Ratio** | Return divided by Max Drawdown | > 1.0 good |
                
                **Which metric matters most?**
                - **Risk-averse investors**: Focus on Max Drawdown and Volatility
                - **Return-focused investors**: Focus on Return and Sharpe
                - **Balanced approach**: Sortino and Calmar (penalize only downside)
                """)
            

            # ============================================================
            # SECTION 2: MORE ON HIERARCHICAL RISK PARITY
            # ============================================================
            with st.expander("🌳 More on Hierarchical Risk Parity (HRP)"):
                st.markdown("""
                ## Understanding Hierarchical Risk Parity
                
                Hierarchical Risk Parity (HRP) is a portfolio optimization technique developed by 
                **Marcos López de Prado** in 2016. Unlike traditional methods like Markowitz optimization, 
                HRP doesn't try to find the "optimal" portfolio by solving a mathematical equation. 
                Instead, it uses **machine learning** (hierarchical clustering) to build a diversified 
                portfolio that respects the natural structure of asset relationships.
                
                ### Why HRP?
                
                Traditional portfolio optimization has well-known problems:
                
                - **Instability**: Small changes in inputs lead to dramatically different portfolios
                - **Concentration**: Often puts all eggs in few baskets
                - **Overfitting**: Optimizes perfectly on past data, disappoints in real trading
                
                HRP addresses these issues by taking a completely different approach: instead of 
                optimizing, it **organizes** assets into a hierarchy and allocates risk accordingly.
                
                ---
                
                ### How HRP Works: The Three Steps
                
                **Step 1: Tree Clustering**
                
                First, we measure how "similar" each pair of assets is based on their correlation. 
                Assets that move together (high correlation) are considered similar. We then build 
                a tree (dendrogram) where similar assets are grouped together.
                
                The distance between assets is calculated as:
                
                $$d_{i,j} = \\sqrt{\\frac{1}{2}(1 - \\rho_{i,j})}$$
                
                where $\\rho_{i,j}$ is the correlation between assets $i$ and $j$.
                
                **Step 2: Quasi-Diagonalization**
                
                We reorganize the assets in the correlation matrix so that the largest correlations lie around the diagonal. 
                This way, assets will end up close to those similar to them 
                and far apart from very different ones and we will be able to visualize the clusters in a correlation matrix
                
                **Step 3: Recursive Bisection**
                
                Finally, we allocate weights by repeatedly splitting the portfolio in half and giving 
                more weight to the lower variance half. This continues until each asset 
                has its final weight.
                
                The allocation formula at each split is:
                
                $$\\alpha = \\frac{V_{right}}{V_{left} + V_{right}}$$
                
                where $V$ is the variance of each cluster. The cluster with **lower variance gets more weight**.
                """)
                
                st.markdown("---")
                st.markdown("### 📊 Dendrogram: Your Portfolio's Tree")
                st.markdown("""
                The dendrogram below shows how assets in your portfolio are related. Assets that are 
                connected at lower heights are more similar (higher correlation). The structure reveals 
                natural "clusters" of assets that tend to move together.
                """)
                
                try:
                    from scipy.cluster.hierarchy import dendrogram
                    
                    link, symbols, corr_matrix, distance_matrix = get_hrp_dendrogram_data(analyzer)
                    
                    # Create dendrogram figure
                    fig_dendro = go.Figure()
                    
                    # Calculate dendrogram using scipy (for coordinates)
                    dendro_data = dendrogram(link, labels=symbols, no_plot=True)
                    
                    # Extract coordinates
                    icoord = np.array(dendro_data['icoord'])
                    dcoord = np.array(dendro_data['dcoord'])
                    
                    # Plot dendrogram lines
                    for i in range(len(icoord)):
                        # Horizontal lines
                        fig_dendro.add_trace(go.Scatter(
                            x=icoord[i],
                            y=dcoord[i],
                            mode='lines',
                            line=dict(color='#6366F1', width=2),
                            hoverinfo='skip',
                            showlegend=False
                        ))
                    
                    # Add asset labels
                    leaf_labels = dendro_data['ivl']
                    leaf_positions = [5 + 10*i for i in range(len(leaf_labels))]
                    
                    # Get display names for labels
                    display_labels = [get_display_name(label) if len(get_display_name(label)) <= 12 
                                     else get_display_name(label)[:10] + '..' for label in leaf_labels]
                    
                    fig_dendro.add_trace(go.Scatter(
                        x=leaf_positions,
                        y=[-0.02] * len(leaf_labels),
                        mode='text',
                        text=display_labels,
                        textposition='bottom center',
                        textfont=dict(size=9, color='#E2E8F0'),
                        hovertext=[f"{get_display_name(label)} ({label})" for label in leaf_labels],
                        hoverinfo='text',
                        showlegend=False
                    ))
                    
                    fig_dendro.update_layout(
                        height=400,
                        xaxis=dict(
                            showticklabels=False,
                            showgrid=False,
                            zeroline=False,
                            title=""
                        ),
                        yaxis=dict(
                            title="Distance (lower = more similar)",
                            showgrid=True,
                            gridcolor='rgba(99,102,241,0.15)',
                            zeroline=False
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=20, b=80, l=60, r=20)
                    )
                    
                    st.plotly_chart(fig_dendro, use_container_width=True)
                    
                    st.markdown("""
                    **How to read the dendrogram:**
                    - Assets at the **bottom** are individual holdings
                    - **Vertical lines** show when assets/clusters merge
                    - **Height of merge** indicates dissimilarity (lower = more correlated)
                    - Assets connected at **low heights** tend to move together
                    """)
                    
                    st.markdown("---")
                    st.markdown("### 🔍 Cluster Analysis")
                    
                    # Identify main clusters (cut tree at appropriate height)
                    from scipy.cluster.hierarchy import fcluster
                    
                    # Determine optimal number of clusters (between 2 and min(5, n_assets))
                    max_clusters = min(5, len(symbols))
                    n_clusters = min(3, max_clusters)  # Default to 3 clusters
                    
                    cluster_labels = fcluster(link, n_clusters, criterion='maxclust')
                    
                    # Group assets by cluster
                    clusters = {}
                    for i, (symbol, cluster_id) in enumerate(zip(symbols, cluster_labels)):
                        if cluster_id not in clusters:
                            clusters[cluster_id] = []
                        clusters[cluster_id].append(symbol)
                    
                    # Display clusters
                    st.markdown(f"Based on correlation structure, your {len(symbols)} assets form **{len(clusters)} main groups**:")
                    
                    cluster_cols = st.columns(min(len(clusters), 3))
                    
                    cluster_colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#AA96DA']
                    
                    for idx, (cluster_id, assets) in enumerate(sorted(clusters.items())):
                        with cluster_cols[idx % len(cluster_cols)]:
                            color = cluster_colors[idx % len(cluster_colors)]
                            
                            # Calculate average correlation within cluster
                            if len(assets) > 1:
                                cluster_corrs = []
                                for i, a1 in enumerate(assets):
                                    for a2 in assets[i+1:]:
                                        cluster_corrs.append(corr_matrix.loc[a1, a2])
                                avg_corr = np.mean(cluster_corrs)
                            else:
                                avg_corr = 1.0
                            
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, {color}22, {color}11); 
                                        border-left: 3px solid {color}; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                                <strong style='color: {color};'>Group {idx + 1}</strong><br>
                                <span style='color: #94a3b8; font-size: 0.85rem;'>
                                    {len(assets)} assets · Avg correlation: {avg_corr:.2f}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for asset in assets:
                                st.markdown(f"<span style='color: #E2E8F0; font-size: 0.9rem;'>• {get_display_name(asset)}</span>", 
                                           unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### ⚖️ HRP Weight Allocation Logic")
                    
                    # Show HRP weights if available
                    if 'hrp' in analyzer.portfolios:
                        hrp_weights = analyzer.portfolios['hrp']['weights']
                        
                        # Create weight breakdown by cluster
                        cluster_weights = {}
                        for cluster_id, assets in clusters.items():
                            cluster_weight = sum(hrp_weights[symbols.index(a)] for a in assets)
                            cluster_weights[cluster_id] = cluster_weight
                        
                        st.markdown("""
                        HRP allocates weights based on the principle: **lower risk clusters get more weight**. 
                        Here's how the weight is distributed across the identified groups:
                        """)
                        
                        # Cluster weight visualization
                        fig_cluster_weights = go.Figure()
                        
                        cluster_names = [f"Group {i+1}" for i in range(len(clusters))]
                        weights_pct = [cluster_weights[i+1] * 100 for i in range(len(clusters))]
                        
                        fig_cluster_weights.add_trace(go.Bar(
                            x=cluster_names,
                            y=weights_pct,
                            marker_color=cluster_colors[:len(clusters)],
                            text=[f"{w:.1f}%" for w in weights_pct],
                            textposition='outside',
                            textfont=dict(color='#E2E8F0', size=11)
                        ))
                        
                        fig_cluster_weights.update_layout(
                            height=300,
                            yaxis_title="Weight Allocation (%)",
                            xaxis_title="",
                            showlegend=False
                        )
                        fig_cluster_weights = apply_plotly_theme(fig_cluster_weights)
                        st.plotly_chart(fig_cluster_weights, use_container_width=True)
                        
                        # Individual asset weights within HRP
                        st.markdown("**Individual Asset Weights (HRP):**")
                        
                        hrp_weight_data = []
                        for i, symbol in enumerate(symbols):
                            cluster_id = cluster_labels[i]
                            hrp_weight_data.append({
                                'Asset': get_display_name(symbol),
                                'Ticker': symbol,
                                'Group': f"Group {cluster_id}",
                                'Weight': f"{hrp_weights[i]*100:.2f}%"
                            })
                        
                        hrp_weight_df = pd.DataFrame(hrp_weight_data)
                        hrp_weight_df = hrp_weight_df.sort_values('Weight', ascending=False)
                        
                        st.markdown(create_styled_table(hrp_weight_df, "HRP Portfolio Weights"), unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### 📚 Key Takeaways")
                    
                    st.markdown("""
                    <div class='dashboard-card'>
                    <p><strong>Why HRP often works well in practice:</strong></p>
                    <ul>
                        <li><strong>No forecasting required</strong>: Unlike Markowitz, HRP doesn't need expected returns (notoriously hard to predict)</li>
                        <li><strong>Stable allocations</strong>: Small changes in correlations don't cause dramatic portfolio shifts</li>
                        <li><strong>Natural diversification</strong>: The hierarchical structure prevents over-concentration</li>
                        <li><strong>Respects market structure</strong>: Assets that behave similarly are treated as a group</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class='dashboard-card'>
                    <p><strong>Limitations to keep in mind:</strong></p>
                    <ul>
                        <li>HRP doesn't optimize for any specific goal (max return, min risk)</li>
                        <li>It assumes historical correlations persist into the future</li>
                        <li>The clustering is sensitive to the time period used</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("""
                    📖 **Reference**: López de Prado, M. (2016). "Building Diversified Portfolios that 
                    Outperform Out-of-Sample". *Journal of Portfolio Management*, 42(4), 59-69.
                    """)
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not generate HRP visualization: {str(e)}")
                    st.markdown("""
                    The dendrogram visualization requires the scipy library. 
                    The HRP optimization still works, but the visual breakdown is unavailable.
                    """)            

            # SECTION 3: CVaR OPTIMIZATION DEEP-DIVE
            # ============================================================
            if 'cvar' in analyzer.portfolios:
                with st.expander("📉 Understanding CVaR Optimization (Rockafellar & Uryasev, 2000)", expanded=False):
                    st.markdown("""
                    ## What is CVaR and Why Does It Matter?
                    
                    **CVaR (Conditional Value-at-Risk)** is a risk measure that focuses on **tail risk** —
                    the average loss in the worst-case scenarios.
                    
                    ### The Problem with Variance and VaR
                    
                    **Traditional Mean-Variance (Markowitz)**:
                    - Penalizes **all volatility** equally (upside and downside)
                    - Assumes normal distributions (fails in crashes)
                    - Sharpe ratio doesn't capture tail risk
                    
                    **Value-at-Risk (VaR)**:
                    - Only tells you the threshold: *"5% chance of losing more than X%"*
                    - **Ignores how much you lose beyond that threshold**
                    - Not sub-additive → diversification can actually increase VaR
                    
                    ### CVaR: The Solution
                    
                    CVaR answers: *"If I'm in the worst 5% of scenarios, what's my **average** loss?"*
                    
                    **Mathematical definition:**
                    
                    $$\\text{CVaR}_{\\alpha}(X) = \\mathbb{E}[X | X \\geq \\text{VaR}_{\\alpha}(X)]$$
                    
                    - **α = 0.95** → CVaR is the average of the worst 5% of returns
                    - **α = 0.99** → CVaR is the average of the worst 1% (more extreme tail)
                    
                    ---
                    
                    ### Why CVaR is a **Coherent Risk Measure**
                    
                    Artzner et al. (1999) defined four axioms that any "reasonable" risk measure should satisfy:
                    
                    1. **Monotonicity**: If portfolio A always loses less than B, then Risk(A) ≤ Risk(B)
                    2. **Sub-additivity**: Risk(A + B) ≤ Risk(A) + Risk(B) (diversification never hurts)
                    3. **Positive homogeneity**: Doubling position size doubles risk
                    4. **Translation invariance**: Adding cash reduces risk proportionally
                                        
                    ---
                    
                    ### The Rockafellar & Uryasev (2000) Breakthrough
                                        
                    The paper finds that CVaR can be computed by minimizing a **smooth convex function**:
                    
                    $$F_{\\alpha}(x, \\zeta) = \\zeta + \\frac{1}{1-\\alpha} \\mathbb{E}[[L(x,y) - \\zeta]^+]$$
                    
                    where:
                    - $x$ = portfolio weights
                    - $\\zeta$ = auxiliary variable (becomes VaR at optimum)
                    - $L(x,y) = -R^T x$ = portfolio loss
                    - $[z]^+ = \\max(z, 0)$
                    
                    **Key theorem (Paper, Theorem 1):**
                    
                    $$\\text{CVaR}_{\\alpha}(x) = \\min_{\\zeta \\in \\mathbb{R}} F_{\\alpha}(x, \\zeta)$$
                    
                    This transforms CVaR optimization into a **Linear Program** when using historical scenarios:
                    
                    $$
                    \\begin{align}
                    \\min_{x, \\zeta, u} \\quad & \\zeta + \\frac{1}{T(1-\\alpha)} \\sum_{t=1}^T u_t \\\\
                    \\text{s.t.} \\quad & u_t \\geq -R_t^T x - \\zeta, \\quad \\forall t \\\\
                    & u_t \\geq 0, \\quad \\forall t \\\\
                    & \\sum_i x_i = 1, \\quad x_i \\geq 0
                    \\end{align}
                    $$
                    
                    **Intuition:**
                    - $u_t$ captures how much scenario $t$ exceeds threshold $\\zeta$
                    - The LP finds optimal $(x, \\zeta)$ that minimizes average tail loss
                    - At optimum, $\\zeta^* = \\text{VaR}_{\\alpha}(x^*)$ and objective = $\\text{CVaR}_{\\alpha}(x^*)$
                    
                    ---
                    
                    ### Comparison: CVaR vs Your Other Strategies
                    
                    Let's see how CVaR compares to the portfolios you already have:
                    """)
                    
                    # Display CVaR-specific metrics
                    cvar_portfolio = analyzer.portfolios['cvar']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("CVaR (daily)", f"{cvar_portfolio['cvar_value']*100:.3f}%",
                                 help="Average loss in worst 5% of days")
                    with col2:
                        st.metric("VaR (daily)", f"{cvar_portfolio['var_value']*100:.3f}%",
                                 help="5th percentile threshold")
                    with col3:
                        st.metric("Confidence Level", f"{cvar_portfolio['cvar_alpha']:.0%}")
                    with col4:
                        st.metric("Annualized CVaR", f"{cvar_portfolio['cvar_value']*np.sqrt(252)*100:.2f}%",
                                 help="CVaR scaled to annual terms (approximate)")
                    
                    st.markdown("---")
                    
                    # Compare CVaR strategy with others
                    st.markdown("#### 🔍 CVaR vs Other Strategies")
                    
                    comparison_metrics = []
                    for key, p in analyzer.portfolios.items():
                        # Calculate actual CVaR for each portfolio
                        port_returns = p['returns']
                        var_95 = port_returns.quantile(0.05)
                        cvar_95 = port_returns[port_returns <= var_95].mean()
                        
                        comparison_metrics.append({
                            'Strategy': p['name'],
                            'Sharpe': f"{p['sharpe_ratio']:.3f}",
                            'Max DD': f"{p['max_drawdown']*100:.2f}%",
                            'VaR (5%)': f"{var_95*100:.3f}%",
                            'CVaR (5%)': f"{cvar_95*100:.3f}%",
                            'CVaR Rank': cvar_95  # For sorting
                        })
                    
                    comparison_df = pd.DataFrame(comparison_metrics)
                    comparison_df = comparison_df.sort_values('CVaR Rank', ascending=False)  # Lower CVaR = better
                    comparison_df['#'] = range(1, len(comparison_df) + 1)
                    comparison_df = comparison_df.drop('CVaR Rank', axis=1)
                    comparison_df = comparison_df[['#', 'Strategy', 'Sharpe', 'Max DD', 'VaR (5%)', 'CVaR (5%)']]
                    
                    st.markdown(create_styled_table(comparison_df, "Ranked by CVaR (lower = better tail risk)"), unsafe_allow_html=True)
                    
                    cvar_rank = comparison_df[comparison_df['Strategy'] == cvar_portfolio['name']]['#'].values[0]
                    
                    if cvar_rank == 1:
                        st.success(f"🏆 **CVaR strategy has the best (lowest) tail risk** among all strategies!")
                    elif cvar_rank <= 3:
                        st.info(f"📊 **CVaR strategy ranks #{cvar_rank}** in tail risk control.")
                    else:
                        st.warning(f"⚠️ **CVaR strategy ranks #{cvar_rank}** — other strategies have lower tail risk.")
                    
                    st.markdown("---")
                    
                    st.markdown("#### 💼 When to Use CVaR Optimization")
                    
                    st.markdown("""
                    <div class='dashboard-card'>
                    <p><strong>CVaR is ideal for:</strong></p>
                    <ul>
                        <li><strong>Risk-averse investors</strong>: You care more about avoiding large losses than maximizing returns</li>
                        <li><strong>Liability-driven investing</strong>: You have obligations to meet and can't afford tail losses</li>
                        <li><strong>Crisis periods</strong>: When market correlations spike and tail risk dominates</li>
                        <li><strong>Regulatory compliance</strong>: Basel III uses CVaR (Expected Shortfall) for bank capital</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class='dashboard-card'>
                    <p><strong>Trade-offs to consider:</strong></p>
                    <ul>
                        <li><strong>Lower expected returns</strong>: CVaR portfolios sacrifice upside to control downside</li>
                        <li><strong>Estimation risk</strong>: CVaR estimated on historical data may not predict future tail events</li>
                        <li><strong>Ignores upside</strong>: Doesn't capture positive skewness (big gains)</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    st.markdown("#### 📚 Key References")
                    
                    st.markdown("""
                    1. **Rockafellar, R.T. & Uryasev, S. (2000).** "Optimization of Conditional Value-at-Risk." 
                       *Journal of Risk*, 2(3), 21-41.
                    
                    2. **Artzner, P., Delbaen, F., Eber, J.M., & Heath, D. (1999).** "Coherent Measures of Risk." 
                       *Mathematical Finance*, 9(3), 203-228.
                    
                    3. **Basel Committee on Banking Supervision (2016).** "Minimum Capital Requirements for Market Risk."
                    """)
                              
            
            st.markdown("---")            
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Returns by Strategy")
                port_data = sorted(analyzer.portfolios.values(), key=lambda x: x['annualized_return'], reverse=True)
                fig = go.Figure(data=[go.Bar(
                    x=[p['name'] for p in port_data], 
                    y=[p['annualized_return'] * 100 for p in port_data],
                    marker_color=CHART_COLORS[:len(port_data)],
                    text=[f"{p['annualized_return']*100:.1f}%" for p in port_data],
                    textposition='outside', 
                    textfont=dict(color='#E2E8F0', size=9)
                )])
                fig.update_layout(height=380, xaxis_tickangle=-45, yaxis_title="Annualized Return (%)")
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### ⚖️ Risk vs Return")
                fig = go.Figure()
                for i, p in enumerate(analyzer.portfolios.values()):
                    fig.add_trace(go.Scatter(
                        x=[p['annualized_volatility'] * 100], 
                        y=[p['annualized_return'] * 100],
                        mode='markers+text', 
                        name=p['name'],
                        marker=dict(size=16, color=CHART_COLORS[i % len(CHART_COLORS)]),
                        text=[p['name'].split()[0]], 
                        textposition='top center', 
                        textfont=dict(color='#E2E8F0', size=8),
                        hovertemplate=f"<b>{p['name']}</b><br>Return: %{{y:.2f}}%<br>Volatility: %{{x:.2f}}%<extra></extra>"
                    ))
                fig.update_layout(height=380, xaxis_title="Volatility (%)", yaxis_title="Return (%)", showlegend=False)
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            st.caption("💡 **Ideal position**: Top-left corner (high return, low volatility). The closer to this corner, the better the risk-adjusted performance.")
            
            st.markdown("---")
            
            # ============================================================
            # SECTION 2: PORTFOLIO DEEP-DIVE
            # ============================================================
            st.markdown("""
            ## 🔍 Portfolio Deep-Dive
            
            Select a strategy to explore its composition and characteristics in detail.
            """)
            
            portfolio_keys = list(analyzer.portfolios.keys())
            selected_p = st.selectbox(
                "Select strategy to analyze", 
                portfolio_keys, 
                format_func=lambda x: analyzer.portfolios[x]['name'], 
                key="port_detail"
            )
            portfolio = analyzer.portfolios[selected_p]
            
            # Metrics row
            st.markdown("#### 📈 Key Metrics (Theoretical)")
            mcols = st.columns(6)
            metrics_list = [
                ("Return", f"{portfolio['annualized_return']*100:.2f}%", "Annualized"),
                ("Volatility", f"{portfolio['annualized_volatility']*100:.2f}%", "Annualized"),
                ("Sharpe", f"{portfolio['sharpe_ratio']:.3f}", "Risk-adjusted"),
                ("Sortino", f"{portfolio['sortino_ratio']:.3f}", "Downside-adjusted"),
                ("Max DD", f"{portfolio['max_drawdown']*100:.2f}%", "Worst decline"),
                ("Calmar", f"{portfolio['calmar_ratio']:.3f}", "Return/MaxDD")
            ]
            for col, (label, value, help_text) in zip(mcols, metrics_list):
                col.metric(label, value, help=help_text)
            
            st.markdown("#### ⚖️ Asset Allocation")
            
            weights_data = []
            for ticker, weight in zip(symbols, portfolio['weights']):
                if weight > 0.001:
                    weights_data.append({
                        'Asset': get_display_name(ticker), 
                        'Ticker': ticker, 
                        'Weight': weight * 100
                    })
            weights_df = pd.DataFrame(weights_data).sort_values('Weight', ascending=False)
            
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                table_data = [{'Asset': r['Asset'], 'Ticker': r['Ticker'], 'Weight': f"{r['Weight']:.2f}%"} for _, r in weights_df.iterrows()]
                st.markdown(create_styled_table(pd.DataFrame(table_data), "Portfolio Weights"), unsafe_allow_html=True)
                
                # Concentration metrics
                top_3_weight = weights_df.head(3)['Weight'].sum()
                n_assets_90 = len(weights_df[weights_df['Weight'].cumsum() <= 90]) + 1
                
                st.markdown(f"""
                **Concentration Analysis:**
                - Top 3 assets: **{top_3_weight:.1f}%** of portfolio
                - Assets needed for 90% weight: **{n_assets_90}**
                - {"⚠️ High concentration" if top_3_weight > 60 else "✅ Well diversified"}
                """)
            
            with col2:
                fig = go.Figure(data=[go.Pie(
                    labels=weights_df['Asset'], 
                    values=weights_df['Weight'], 
                    hole=0.4,
                    marker_colors=CHART_COLORS[:len(weights_df)], 
                    textinfo='percent', 
                    textposition='outside',
                    textfont=dict(color='#E2E8F0', size=10),
                    hovertemplate='<b>%{label}</b><br>%{value:.2f}%<extra></extra>'
                )])
                fig.update_layout(
                    height=320, 
                    showlegend=False,
                    annotations=[dict(
                        text=portfolio['name'].split()[0], 
                        x=0.5, y=0.5, 
                        font_size=11, 
                        font_color='#E2E8F0', 
                        showarrow=False
                    )]
                )
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ============================================================
            # SECTION 3: SEASONALITY ANALYSIS
            # ============================================================
            st.markdown("""
            ## 📅 Seasonality Analysis
            
            Does this portfolio perform better in certain months? This analysis uses **theoretical returns** 
            (before costs) to identify **pure market patterns** that may repeat.
            """)
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                portfolio_returns = analyzer.portfolios[selected_p]['returns']
                available_years = sorted(portfolio_returns.index.year.unique(), reverse=True)
                
                n_years_display = st.slider(
                    "Number of years to display",
                    min_value=1,
                    max_value=min(len(available_years), 15),
                    value=min(5, len(available_years)),
                    key="seasonality_years"
                )
            
            with col2:
                selected_years = st.multiselect(
                    "Or select specific years",
                    options=available_years,
                    default=None,
                    key="seasonality_specific_years"
                )
            
            if selected_years:
                years_to_analyze = sorted(selected_years)
            else:
                years_to_analyze = available_years[:n_years_display]
            
            mask = portfolio_returns.index.year.isin(years_to_analyze)
            filtered_returns = portfolio_returns[mask]
            
            if len(filtered_returns) > 20:
                seasonality_df = pd.DataFrame({
                    'Return': filtered_returns.values,
                    'Date': filtered_returns.index,
                    'Year': filtered_returns.index.year,
                    'Month': filtered_returns.index.month
                })
                
                # Year-over-Year Performance
                st.markdown("#### 📈 Year-over-Year Performance (Base 100)")
                st.caption("Compare how the portfolio evolved throughout each year.")
                
                fig_yearly = go.Figure()
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                current_year = datetime.now().year
                sorted_years = sorted(years_to_analyze, reverse=True)
                
                for i, year in enumerate(sorted_years):
                    year_data = seasonality_df[seasonality_df['Year'] == year].copy()
                    if len(year_data) > 0:
                        year_data = year_data.sort_values('Date')
                        year_data['Cumulative'] = (1 + year_data['Return']).cumprod() * 100
                        year_data['DayOfYear'] = year_data['Date'].dt.dayofyear
                        
                        is_current_year = (year == current_year)
                        
                        if is_current_year:
                            line_style = dict(color='#FFE66D', width=4)
                            opacity = 1.0
                        else:
                            color_idx = (i if year != current_year else i + 1) % len(CHART_COLORS)
                            line_style = dict(color=CHART_COLORS[color_idx], width=1.5)
                            opacity = 0.5
                        
                        fig_yearly.add_trace(go.Scatter(
                            x=year_data['DayOfYear'],
                            y=year_data['Cumulative'],
                            name=str(year),
                            mode='lines',
                            line=line_style,
                            opacity=opacity,
                            hovertemplate=f'<b>{year}</b><br>Day: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                        ))
                        
                        if is_current_year and len(year_data) > 0:
                            last_point = year_data.iloc[-1]
                            fig_yearly.add_trace(go.Scatter(
                                x=[last_point['DayOfYear']],
                                y=[last_point['Cumulative']],
                                mode='markers+text',
                                marker=dict(size=12, color='#FFE66D', line=dict(width=2, color='white')),
                                text=[f"{last_point['Cumulative']:.1f}"],
                                textposition='top right',
                                textfont=dict(color='#FFE66D', size=11),
                                showlegend=False,
                                hovertemplate=f'<b>📍 LATEST ({current_year})</b><br>Value: %{{y:.2f}}<extra></extra>'
                            ))
                
                month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
                
                fig_yearly.update_layout(
                    height=450,
                    xaxis=dict(tickmode='array', tickvals=month_starts, ticktext=month_names, title="Month"),
                    yaxis_title="Cumulative Value (Base 100)",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10))
                )
                fig_yearly.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_yearly = apply_plotly_theme(fig_yearly)
                st.plotly_chart(fig_yearly, use_container_width=True)
                
                if current_year in years_to_analyze:
                    current_year_data = seasonality_df[seasonality_df['Year'] == current_year]
                    if len(current_year_data) > 0:
                        ytd_return = ((1 + current_year_data['Return']).prod() - 1) * 100
                        last_date = current_year_data['Date'].max().strftime('%d %b %Y')
                        
                        if ytd_return >= 0:
                            st.success(f"📍 **{current_year} YTD**: {ytd_return:+.2f}% (as of {last_date})")
                        else:
                            st.error(f"📍 **{current_year} YTD**: {ytd_return:+.2f}% (as of {last_date})")
                
                # Average Monthly Performance
                st.markdown("#### 📊 Average Monthly Performance")
                
                monthly_stats = seasonality_df.groupby(['Year', 'Month']).agg(
                    avg_daily_return=('Return', 'mean'),
                    trading_days=('Return', 'count')
                ).reset_index()
                
                monthly_stats['Monthly_Return'] = ((1 + monthly_stats['avg_daily_return']) ** monthly_stats['trading_days'] - 1) * 100
                
                avg_monthly = monthly_stats.groupby('Month')['Monthly_Return'].mean().reset_index()
                month_name_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
                avg_monthly['Month_Name'] = avg_monthly['Month'].map(month_name_map)
                
                colors = ['#4ECDC4' if x >= 0 else '#FF6B6B' for x in avg_monthly['Monthly_Return']]
                
                fig_monthly_avg = go.Figure(data=[go.Bar(
                    x=avg_monthly['Month_Name'],
                    y=avg_monthly['Monthly_Return'],
                    marker_color=colors,
                    text=[f"{x:+.2f}%" for x in avg_monthly['Monthly_Return']],
                    textposition='outside',
                    textfont=dict(color='#E2E8F0', size=10),
                    hovertemplate='<b>%{x}</b><br>Avg Return: %{y:.2f}%<extra></extra>'
                )])
                
                fig_monthly_avg.update_layout(height=350, xaxis_title="Month", yaxis_title="Average Monthly Return (%)")
                fig_monthly_avg.add_hline(y=0, line_color="rgba(255,255,255,0.5)")
                fig_monthly_avg = apply_plotly_theme(fig_monthly_avg)
                st.plotly_chart(fig_monthly_avg, use_container_width=True)
                
                # Summary Statistics
                col1, col2, col3, col4 = st.columns(4)
                
                best_month = avg_monthly.loc[avg_monthly['Monthly_Return'].idxmax()]
                worst_month = avg_monthly.loc[avg_monthly['Monthly_Return'].idxmin()]
                
                with col1:
                    st.metric("🏆 Best Month", best_month['Month_Name'], f"{best_month['Monthly_Return']:+.2f}%")
                with col2:
                    st.metric("📉 Worst Month", worst_month['Month_Name'], f"{worst_month['Monthly_Return']:+.2f}%")
                with col3:
                    positive_months = (avg_monthly['Monthly_Return'] > 0).sum()
                    st.metric("✅ Positive Months", f"{positive_months}/12")
                with col4:
                    seasonality_strength = avg_monthly['Monthly_Return'].std()
                    st.metric("📊 Seasonality", f"{seasonality_strength:.2f}%", 
                             "High" if seasonality_strength > 3 else "Moderate" if seasonality_strength > 1.5 else "Low")
            else:
                st.warning("⚠️ Not enough data for seasonality analysis.")
            
            st.markdown("---")

            # ============================================================
            # SECTION 4: TRANSACTION COSTS & REBALANCING IMPACT
            # ============================================================
            st.markdown("""
            ## 💰 Transaction Costs & Rebalancing Impact
            
            The theoretical returns above assume a frictionless world. In reality, **every trade costs money**.
            This section shows how transaction costs and rebalancing frequency affect your actual returns.
            """)
            
            # Academic reference
            with st.expander("📚 Methodology & Academic References"):
                st.markdown("""
                ### Transaction Cost Model
                
                This analysis implements **proportional transaction costs** following the methodology from:
                
                > **DeMiguel, V., Garlappi, L., & Uppal, R. (2009)**  
                > *"Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?"*  
                > Review of Financial Studies, 22(5), 1915-1953
                
                > **Kirby, C., & Ostdiek, B. (2012)**  
                > *"It's All in the Timing: Simple Active Portfolio Strategies that Outperform Naive Diversification"*  
                > Journal of Financial and Quantitative Analysis, 47(2), 437-467
                
                ---
                
                ### How Costs Are Calculated
                
                **Turnover** measures how much of the portfolio is traded during rebalancing:
                
                $$\\tau_{p,t} = \\sum_{i=1}^{N} |\\omega_{i,t}^{target} - \\tilde{\\omega}_{i,t}|$$
                
                where $\\tilde{\\omega}_{i,t}$ is the weight of asset $i$ after price drift (before rebalancing).
                
                **Transaction cost** reduces portfolio value multiplicatively:
                
                $$V_t^{after} = V_t^{before} \\times (1 - c \\times \\tau_{p,t})$$
                
                where $c$ is the proportional cost rate (e.g., 10 bps = 0.001).
                
                This multiplicative approach correctly captures the **compounding effect** of costs over time,
                which is critical for long-term analysis.
                
                ---
                
                ### Typical Cost Ranges
                
                | Instrument | Typical Cost (bps) |
                |------------|-------------------|
                | Large-cap ETFs (SPY, QQQ) | 2-5 |
                | Broad market ETFs | 5-10 |
                | Individual stocks (liquid) | 5-15 |
                | Individual stocks (less liquid) | 15-30 |
                | International / EM | 20-50 |
                
                *Note: Costs include bid-ask spread + broker commissions*
                """)
            
            # Check if we have cost-adjusted data
            has_costs = st.session_state.portfolios_with_costs is not None
            
            if has_costs and st.session_state.cost_config:
                config = st.session_state.cost_config
                
                # Display current configuration
                if config['rebal_freq']:
                    freq_names = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'A': 'Annually'}
                    rebal_desc = freq_names.get(config['rebal_freq'], config['rebal_freq'])
                elif config['rebal_thresh']:
                    rebal_desc = f"Threshold ({config['rebal_thresh']*100:.0f}% drift)"
                else:
                    rebal_desc = "Buy & Hold"
                
                st.info(f"**Current Settings:** {config['bps']} bps per trade | Rebalancing: {rebal_desc}")
                
                portfolios_data = st.session_state.portfolios_with_costs
                
                # ===== IMPACT SUMMARY =====
                st.markdown("#### 📉 Cost Impact Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                avg_cost_drag = np.mean([v['cost_drag'] for v in portfolios_data.values()])
                avg_annual_turnover = np.mean([v['annual_turnover'] for v in portfolios_data.values()])
                total_rebalances = np.mean([v['n_rebalances'] for v in portfolios_data.values()])
                
                cost_drags = [(k, v['cost_drag']) for k, v in portfolios_data.items()]
                most_affected = max(cost_drags, key=lambda x: x[1])
                least_affected = min(cost_drags, key=lambda x: x[1])
                
                with col1:
                    st.metric(
                        "Avg Annual Cost Drag", 
                        f"{avg_cost_drag*100:.2f}%",
                        help="Average annual return lost to transaction costs"
                    )
                
                with col2:
                    st.metric(
                        "Avg Annual Turnover", 
                        f"{avg_annual_turnover*100:.0f}%",
                        help="Average portfolio turnover per year"
                    )
                
                with col3:
                    st.metric(
                        "Most Affected", 
                        portfolios_data[most_affected[0]]['name'].split()[0],
                        delta=f"-{most_affected[1]*100:.2f}%/yr",
                        delta_color="inverse"
                    )
                
                with col4:
                    st.metric(
                        "Least Affected", 
                        portfolios_data[least_affected[0]]['name'].split()[0],
                        delta=f"-{least_affected[1]*100:.2f}%/yr",
                        delta_color="inverse"
                    )
                
                # ===== COMPARISON TABLE =====
                st.markdown("#### 📊 Gross vs Net Performance")
                
                cost_comparison_data = []
                for idx, (key, p) in enumerate(sorted(portfolios_data.items(), key=lambda x: x[1]['sharpe_net'], reverse=True), 1):
                    # Calculate rank change
                    gross_rank = sorted(portfolios_data.keys(), key=lambda x: portfolios_data[x]['sharpe_gross'], reverse=True).index(key) + 1
                    net_rank = idx
                    rank_change = gross_rank - net_rank
                    rank_indicator = f"↑{rank_change}" if rank_change > 0 else f"↓{abs(rank_change)}" if rank_change < 0 else "="
                    
                    cost_comparison_data.append({
                        '#': f"{idx} ({rank_indicator})",
                        'Strategy': p['name'],
                        'Gross Return': f"{p['ann_return_gross']*100:.2f}%",
                        'Net Return': f"{p['ann_return_net']*100:.2f}%",
                        'Cost Drag': f"-{p['cost_drag']*100:.2f}%",
                        'Gross Sharpe': f"{p['sharpe_gross']:.3f}",
                        'Net Sharpe': f"{p['sharpe_net']:.3f}",
                        'Annual Turn.': f"{p['annual_turnover']*100:.0f}%",
                        '# Rebal': p['n_rebalances']
                    })
                
                cost_comparison_df = pd.DataFrame(cost_comparison_data)
                st.markdown(create_styled_table(cost_comparison_df, "Ranked by Net Sharpe (after costs)"), unsafe_allow_html=True)
                
                st.caption("💡 **Rank changes** show how strategies move when costs are included. ↑ = improved, ↓ = worse")
                
                # ===== VISUALIZATIONS =====
                st.markdown("#### 📊 Visual Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gross vs Net Returns - FIXED: use full strategy names to avoid overlap
                    fig_compare = go.Figure()
                    
                    sorted_keys = sorted(portfolios_data.keys(), key=lambda x: portfolios_data[x]['ann_return_gross'], reverse=True)
                    # Use shorter but unique names
                    def get_short_name(name):
                        """Get a short but unique name for the strategy."""
                        if "Maximum Sharpe" in name:
                            return "Max Sharpe"
                        elif "Maximum Return" in name:
                            return "Max Return"
                        elif "Minimum Volatility" in name:
                            return "Min Vol"
                        elif "Hierarchical Risk Parity" in name or "HRP" in name:
                            return "HRP"                            
                        elif "Risk Parity" in name:
                            return "Risk Parity"
                        elif "Equally Weighted" in name:
                            return "Equal"
                        elif "Markowitz" in name:
                            return "Markowitz"
                        else:
                            return name.split()[0]
                    
                    strategies = [get_short_name(portfolios_data[k]['name']) for k in sorted_keys]
                    gross_returns = [portfolios_data[k]['ann_return_gross']*100 for k in sorted_keys]
                    net_returns = [portfolios_data[k]['ann_return_net']*100 for k in sorted_keys]
                    
                    fig_compare.add_trace(go.Bar(
                        name='Gross',
                        x=strategies,
                        y=gross_returns,
                        marker_color='rgba(99, 102, 241, 0.5)',
                        text=[f"{v:.1f}%" for v in gross_returns],
                        textposition='outside',
                        textfont=dict(size=9, color='#94a3b8')
                    ))
                    
                    fig_compare.add_trace(go.Bar(
                        name='Net',
                        x=strategies,
                        y=net_returns,
                        marker_color='rgba(78, 205, 196, 0.9)',
                        text=[f"{v:.1f}%" for v in net_returns],
                        textposition='inside',
                        textfont=dict(size=9, color='white')
                    ))
                    
                    fig_compare.update_layout(
                        barmode='overlay',
                        height=350,
                        yaxis_title="Annualized Return (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    fig_compare = apply_plotly_theme(fig_compare)
                    st.plotly_chart(fig_compare, use_container_width=True)
                    st.caption("Faded = Gross | Solid = Net")
                
                with col2:
                    # Cost Efficiency Scatter
                    fig_efficiency = go.Figure()
                    
                    for i, (k, p) in enumerate(portfolios_data.items()):
                        annual_turn = p.get('annual_turnover', p['total_turnover'] / (len(analyzer.returns) / 252))
                        short_name = get_short_name(p['name'])
                        fig_efficiency.add_trace(go.Scatter(
                            x=[annual_turn*100],
                            y=[p['sharpe_net']],
                            mode='markers+text',
                            name=p['name'],
                            marker=dict(size=16, color=CHART_COLORS[i % len(CHART_COLORS)]),
                            text=[short_name],
                            textposition='top center',
                            textfont=dict(size=9, color='#E2E8F0'),
                            hovertemplate=f"<b>{p['name']}</b><br>Turnover: %{{x:.0f}}%/yr<br>Net Sharpe: %{{y:.3f}}<extra></extra>"
                        ))
                    
                    fig_efficiency.update_layout(
                        height=350,
                        xaxis_title="Annual Turnover (%)",
                        yaxis_title="Net Sharpe Ratio",
                        showlegend=False
                    )
                    fig_efficiency = apply_plotly_theme(fig_efficiency)
                    st.plotly_chart(fig_efficiency, use_container_width=True)
                    st.caption("Top-left = most efficient (high Sharpe, low turnover)")
                    
                # ===== CUMULATIVE IMPACT =====
                st.markdown("#### 📈 Cumulative Cost Impact Over Time")
                
                # Select portfolio for detailed view
                detail_portfolio = st.selectbox(
                    "Select strategy for detailed analysis",
                    options=list(portfolios_data.keys()),
                    format_func=lambda x: portfolios_data[x]['name'],
                    key="cost_detail_select"
                )
                
                p_data = portfolios_data[detail_portfolio]
                
                fig_cumulative = go.Figure()
                
                # Gross cumulative
                fig_cumulative.add_trace(go.Scatter(
                    x=p_data['cumulative_gross'].index,
                    y=(p_data['cumulative_gross'] - 1) * 100,
                    name='Gross (no costs)',
                    mode='lines',
                    line=dict(color='rgba(99, 102, 241, 0.7)', width=2, dash='dash'),
                    hovertemplate='<b>Gross</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                ))
                
                # Net cumulative
                fig_cumulative.add_trace(go.Scatter(
                    x=p_data['cumulative_net'].index,
                    y=(p_data['cumulative_net'] - 1) * 100,
                    name='Net (after costs)',
                    mode='lines',
                    line=dict(color='#4ECDC4', width=2.5),
                    fill='tonexty',
                    fillcolor='rgba(255, 107, 107, 0.2)',
                    hovertemplate='<b>Net</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                ))
                
                # Mark rebalancing events
                if len(p_data['rebalance_dates']) > 0 and len(p_data['rebalance_dates']) <= 50:
                    rebal_y = [(p_data['cumulative_net'].loc[d] - 1) * 100 for d in p_data['rebalance_dates'] if d in p_data['cumulative_net'].index]
                    rebal_x = [d for d in p_data['rebalance_dates'] if d in p_data['cumulative_net'].index]
                    
                    fig_cumulative.add_trace(go.Scatter(
                        x=rebal_x,
                        y=rebal_y,
                        mode='markers',
                        name='Rebalancing',
                        marker=dict(size=6, color='#FF6B6B', symbol='circle'),
                        hovertemplate='<b>🔄 Rebalance</b><br>Date: %{x}<extra></extra>'
                    ))
                
                fig_cumulative.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig_cumulative.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_cumulative = apply_plotly_theme(fig_cumulative)
                st.plotly_chart(fig_cumulative, use_container_width=True)
                
                st.caption("**Shaded area** represents cumulative value lost to transaction costs over time.")
                
                # Summary metrics for selected portfolio
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    final_gross = (p_data['cumulative_gross'].iloc[-1] - 1) * 100
                    st.metric("Total Gross Return", f"{final_gross:.2f}%")
                with col2:
                    final_net = (p_data['cumulative_net'].iloc[-1] - 1) * 100
                    st.metric("Total Net Return", f"{final_net:.2f}%")
                with col3:
                    total_cost = final_gross - final_net
                    st.metric("Total Cost Impact", f"-{total_cost:.2f}%")
                with col4:
                    n_years = len(p_data['cumulative_net']) / 252
                    st.metric("Period", f"{n_years:.1f} years")
                
                # ===== KEY INSIGHTS =====
                with st.expander("💡 Key Insights & Recommendations"):
                    best_gross = max(portfolios_data.items(), key=lambda x: x[1]['sharpe_gross'])
                    best_net = max(portfolios_data.items(), key=lambda x: x[1]['sharpe_net'])
                    lowest_turnover = min(portfolios_data.items(), key=lambda x: x[1]['annual_turnover'])
                    highest_turnover = max(portfolios_data.items(), key=lambda x: x[1]['annual_turnover'])
                    winner_changed = best_gross[0] != best_net[0]
                    
                    st.markdown(f"""
                    ### Analysis Results
                    
                    **Best Strategy (Theoretical):** {best_gross[1]['name']} (Sharpe: {best_gross[1]['sharpe_gross']:.3f})
                    
                    **Best Strategy (After Costs):** {best_net[1]['name']} (Sharpe: {best_net[1]['sharpe_net']:.3f})
                    
                    {"⚠️ **Winner changed!** Transaction costs reversed the ranking." if winner_changed else "✅ **Consistent winner.** Best strategy holds after costs."}
                    
                    ---
                    
                    ### Recommendations
                    
                    1. **Lowest cost option:** {lowest_turnover[1]['name']} ({lowest_turnover[1]['annual_turnover']*100:.0f}% annual turnover)
                    
                    2. **Best risk-adjusted after costs:** {best_net[1]['name']} (Net Sharpe: {best_net[1]['sharpe_net']:.3f})
                    
                    3. **Potential savings:** Switching from {highest_turnover[1]['name']} to {lowest_turnover[1]['name']} 
                       saves ~**{(highest_turnover[1]['cost_drag'] - lowest_turnover[1]['cost_drag'])*100:.2f}%** annually
                    
                    ---
                    
                    ### General Guidelines
                    
                    - **Cost drag > 1%/year**: Consider less frequent rebalancing
                    - **Turnover > 200%/year**: Strategy may be too active for retail implementation
                    - **Rank changes**: If your preferred strategy drops significantly, costs matter for you
                    """)
            
            else:
                st.warning("""
                ⚠️ **Cost analysis not available.** 
                
                To enable transaction cost analysis:
                1. Check "Enable transaction costs" in the sidebar
                2. Set your estimated cost per trade (in basis points)
                3. Choose a rebalancing strategy
                4. Re-run the analysis
                """)

        # ============== TAB 3 ============== #
        with tab3:
            st.markdown("### 📈 Performance Analysis")
            
            st.markdown("""
            This section analyzes the performance of selected strategies from different angles:
            cumulative capital growth, drawdown phases, and the consistency of returns 
            and volatility over time through rolling windows.
            """)
            
            st.markdown("---")
            
            # Strategy selector
            portfolio_keys = list(analyzer.portfolios.keys())
            sel_perf = st.multiselect(
                "Select strategies to compare", 
                portfolio_keys, 
                default=portfolio_keys[:4], 
                format_func=lambda x: analyzer.portfolios[x]['name'], 
                key="perf_unified_sel"
            )
            
            if sel_perf:
                # ============================================================
                # SECTION 1: CUMULATIVE PERFORMANCE
                # ============================================================
                st.markdown("#### 📊 Capital Growth (Base 100)")
                
                st.markdown("""
                <div class='dashboard-card'>
                <p><strong>How to read this chart:</strong> it shows how an initial investment of 100 
                would have grown over time for each strategy. A line reaching 150 means the capital 
                grew by 50%. The steeper the upward slope, the better the performance during that period.</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig_cum = go.Figure()
                
                for i, p_name in enumerate(sel_perf):
                    p = analyzer.portfolios[p_name]
                    cum_val = (1 + p['returns']).cumprod() * 100
                    
                    fig_cum.add_trace(go.Scatter(
                        x=cum_val.index, 
                        y=cum_val.values, 
                        name=p['name'], 
                        mode='lines',
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                        hovertemplate=f'<b>{p["name"]}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ))
                
                fig_cum.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.3)", 
                                annotation_text="Initial capital", annotation_position="right")
                
                fig_cum.update_layout(
                    height=420, 
                    hovermode='x unified', 
                    xaxis_title="Date", 
                    yaxis_title="Value (Base 100)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig_cum = apply_plotly_theme(fig_cum)
                st.plotly_chart(fig_cum, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                perf_data = [(p_name, analyzer.portfolios[p_name]) for p_name in sel_perf]
                best_cum = max(perf_data, key=lambda x: x[1]['cumulative_return'])
                worst_cum = min(perf_data, key=lambda x: x[1]['cumulative_return'])
                
                with col1:
                    st.metric(
                        "🏆 Best Performance", 
                        best_cum[1]['name'].split()[0],
                        f"+{best_cum[1]['cumulative_return']*100:.1f}%"
                    )
                with col2:
                    st.metric(
                        "📉 Worst Performance", 
                        worst_cum[1]['name'].split()[0],
                        f"{worst_cum[1]['cumulative_return']*100:+.1f}%"
                    )
                with col3:
                    avg_return = np.mean([analyzer.portfolios[p]['annualized_return'] for p in sel_perf])
                    st.metric("📊 Average Return", f"{avg_return*100:.1f}%", "annualized")
                with col4:
                    st.metric("📅 Period", f"{len(analyzer.returns)} days")
                
                st.markdown("---")
                
                # ============================================================
                # SECTION 2: DRAWDOWN ANALYSIS
                # ============================================================
                st.markdown("#### 📉 Drawdown Analysis")
                
                st.markdown("""
                <div class='dashboard-card'>
                <p><strong>How to read this chart:</strong> drawdown measures how much the portfolio 
                has declined from its historical peak. A drawdown of -20% means that at that moment 
                you had lost 20% from the peak. It's a measure of investment "pain": deep and prolonged 
                drawdowns are psychologically difficult to endure.</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig_dd = go.Figure()
                
                for i, p_name in enumerate(sel_perf):
                    p = analyzer.portfolios[p_name]
                    cum_val = (1 + p['returns']).cumprod()
                    roll_max = cum_val.expanding().max()
                    dd = (cum_val - roll_max) / roll_max * 100
                    
                    color = CHART_COLORS[i % len(CHART_COLORS)]
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    
                    fig_dd.add_trace(go.Scatter(
                        x=dd.index, 
                        y=dd.values, 
                        name=p['name'], 
                        mode='lines',
                        line=dict(color=color, width=2), 
                        fill='tozeroy', 
                        fillcolor=f'rgba({r},{g},{b},0.2)',
                        hovertemplate=f'<b>{p["name"]}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>'
                    ))
                
                fig_dd.update_layout(
                    height=350, 
                    hovermode='x unified', 
                    xaxis_title="Date", 
                    yaxis_title="Drawdown (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig_dd = apply_plotly_theme(fig_dd)
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Drawdown summary table
                dd_summary = []
                for p_name in sel_perf:
                    p = analyzer.portfolios[p_name]
                    cum_val = (1 + p['returns']).cumprod()
                    roll_max = cum_val.expanding().max()
                    dd_series = (cum_val - roll_max) / roll_max
                    
                    max_dd_idx = dd_series.idxmin()
                    max_dd_val = dd_series.min()
                    
                    post_dd = cum_val[cum_val.index >= max_dd_idx]
                    peak_before_dd = roll_max[max_dd_idx]
                    recovered = post_dd[post_dd >= peak_before_dd]
                    
                    if len(recovered) > 0:
                        recovery_date = recovered.index[0]
                        recovery_days = (recovery_date - max_dd_idx).days
                        recovery_str = f"{recovery_days} days"
                    else:
                        recovery_str = "Not recovered"
                    
                    dd_summary.append({
                        'Strategy': p['name'],
                        'Max Drawdown': f"{max_dd_val*100:.2f}%",
                        'Max DD Date': max_dd_idx.strftime('%Y-%m-%d'),
                        'Recovery Time': recovery_str
                    })
                
                st.markdown(create_styled_table(pd.DataFrame(dd_summary), "Drawdown Summary"), unsafe_allow_html=True)
                
                st.markdown("---")
                
                # ============================================================
                # SECTION 3: ROLLING ANALYSIS
                # ============================================================
                st.markdown("#### 🔄 Rolling Analysis (Moving Windows)")
                
                st.markdown(f"""
                <div class='dashboard-card'>
                <p><strong>What are rolling metrics?</strong> Instead of calculating a single number for the entire 
                period, rolling metrics calculate values over a moving "window" of <strong>{window_years} years</strong> 
                that slides through time. This shows how performance and risk have changed across different 
                market phases, revealing whether a strategy is consistent or only worked in certain periods.</p>
                </div>
                """, unsafe_allow_html=True)
                
                window = window_years * 252
                
                if len(analyzer.returns) < window:
                    st.warning(f"⚠️ Insufficient data for rolling analysis. At least {window_years} years of data ({window} days) required, but only {len(analyzer.returns)} available.")
                else:
                    rolling_data = {}
                    
                    for p_name in sel_perf:
                        p = analyzer.portfolios[p_name]
                        rets = p['returns']
                        
                        roll_ret = rets.rolling(window=window).apply(
                            lambda x: (1 + x).prod() ** (252 / len(x)) - 1 if len(x) == window else np.nan
                        )
                        
                        roll_vol = rets.rolling(window=window).std() * np.sqrt(252)
                        
                        rolling_data[p['name']] = pd.DataFrame({
                            'Return': roll_ret,
                            'Volatility': roll_vol
                        }).dropna()
                    
                    # Rolling Returns Chart
                    st.markdown("##### 📈 Rolling Returns (annualized)")
                    st.caption(f"Annualized return over {window_years}-year rolling window")
                    
                    fig_roll_ret = go.Figure()
                    
                    for i, (name, data) in enumerate(rolling_data.items()):
                        fig_roll_ret.add_trace(go.Scatter(
                            x=data.index, 
                            y=data['Return'] * 100, 
                            name=name,
                            mode='lines',
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                            hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
                        ))
                    
                    fig_roll_ret.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    fig_roll_ret.update_layout(
                        height=400, 
                        hovermode='x unified', 
                        xaxis_title="Date",
                        yaxis_title="Return (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    fig_roll_ret = apply_plotly_theme(fig_roll_ret)
                    st.plotly_chart(fig_roll_ret, use_container_width=True)
                    
                    # Rolling Volatility Chart
                    st.markdown("##### 📊 Rolling Volatility (annualized)")
                    st.caption(f"Annualized volatility over {window_years}-year rolling window")
                    
                    fig_roll_vol = go.Figure()
                    
                    for i, (name, data) in enumerate(rolling_data.items()):
                        fig_roll_vol.add_trace(go.Scatter(
                            x=data.index, 
                            y=data['Volatility'] * 100, 
                            name=name,
                            mode='lines',
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                            hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Volatility: %{{y:.2f}}%<extra></extra>'
                        ))
                    
                    fig_roll_vol.update_layout(
                        height=400, 
                        hovermode='x unified', 
                        xaxis_title="Date",
                        yaxis_title="Volatility (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    fig_roll_vol = apply_plotly_theme(fig_roll_vol)
                    st.plotly_chart(fig_roll_vol, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Rolling statistics summary
                    st.markdown("##### 📋 Rolling Statistics")
                    
                    rolling_stats = []
                    for p_name in sel_perf:
                        p = analyzer.portfolios[p_name]
                        if p['name'] in rolling_data:
                            data = rolling_data[p['name']]
                            
                            rolling_stats.append({
                                'Strategy': p['name'],
                                'Avg Return': f"{data['Return'].mean()*100:.2f}%",
                                'Min Return': f"{data['Return'].min()*100:.2f}%",
                                'Max Return': f"{data['Return'].max()*100:.2f}%",
                                'Avg Vol': f"{data['Volatility'].mean()*100:.2f}%",
                                'Min Vol': f"{data['Volatility'].min()*100:.2f}%",
                                'Max Vol': f"{data['Volatility'].max()*100:.2f}%"
                            })
                    
                    if rolling_stats:
                        st.markdown(create_styled_table(pd.DataFrame(rolling_stats), f"Statistics over {window_years}-year window"), unsafe_allow_html=True)
                    
                    with st.expander("📖 How to interpret rolling analysis"):
                        st.markdown(f"""
                        ### What to look for in rolling charts
                        
                        **Rolling Returns:**
                        - Stable lines above zero = consistent performance over time
                        - High variability = the strategy only works in certain market regimes
                        - Prolonged periods below zero = phases that are psychologically difficult to endure
                        
                        **Rolling Volatility:**
                        - Flat lines = predictable and stable risk
                        - Sudden spikes = the strategy becomes riskier during stress periods
                        - Increasing volatility over time = potential warning signal
                        
                        ### Comparing strategies
                        
                        An ideal strategy shows:
                        - Consistently positive rolling returns
                        - Stable and contained rolling volatility
                        - Low correlation between volatility spikes and return drops
                        
                        ### Important note
                        
                        Rolling metrics are calculated "looking backward" - the value on December 31, 2023 
                        reflects the previous {window_years} years, it does not predict the future.
                        """)

            # ============================================================
            # SECTION 4: ROLLING ANALYSIS - GROSS VS NET (COST IMPACT)
            # ============================================================
            if st.session_state.portfolios_with_costs is not None:
                st.markdown("---")
                
                with st.expander("🔬 Advanced: Rolling Cost Impact Analysis"):
                    st.markdown("""
                    This section compares **theoretical (gross)** vs **after-cost (net)** rolling metrics 
                    for a selected strategy. Use this to identify periods where transaction costs 
                    had the largest impact on performance.
                    """)
                    
                    portfolios_data = st.session_state.portfolios_with_costs
                    
                    # Strategy selector
                    cost_roll_portfolio = st.selectbox(
                        "Select strategy to analyze",
                        options=list(portfolios_data.keys()),
                        format_func=lambda x: portfolios_data[x]['name'],
                        key="cost_rolling_select"
                    )
                    
                    p_data = portfolios_data[cost_roll_portfolio]
                    
                    # Check if we have enough data
                    window = window_years * 252
                    
                    if len(p_data['returns_gross']) < window:
                        st.warning(f"⚠️ Insufficient data for rolling analysis. Need at least {window_years} years.")
                    else:
                        # Calculate rolling metrics for both gross and net
                        returns_gross = p_data['returns_gross']
                        returns_net = p_data['returns_net']
                        
                        # Rolling returns
                        roll_ret_gross = returns_gross.rolling(window=window).apply(
                            lambda x: (1 + x).prod() ** (252 / len(x)) - 1 if len(x) == window else np.nan
                        ).dropna()
                        
                        roll_ret_net = returns_net.rolling(window=window).apply(
                            lambda x: (1 + x).prod() ** (252 / len(x)) - 1 if len(x) == window else np.nan
                        ).dropna()
                        
                        # Rolling volatility
                        roll_vol_gross = returns_gross.rolling(window=window).std().dropna() * np.sqrt(252)
                        roll_vol_net = returns_net.rolling(window=window).std().dropna() * np.sqrt(252)
                        
                        # Align indices
                        common_idx = roll_ret_gross.index.intersection(roll_ret_net.index)
                        roll_ret_gross = roll_ret_gross.loc[common_idx]
                        roll_ret_net = roll_ret_net.loc[common_idx]
                        
                        common_idx_vol = roll_vol_gross.index.intersection(roll_vol_net.index)
                        roll_vol_gross = roll_vol_gross.loc[common_idx_vol]
                        roll_vol_net = roll_vol_net.loc[common_idx_vol]
                        
                        # Rolling cost drag
                        roll_cost_drag = roll_ret_gross - roll_ret_net
                        
                        # Chart 1: Rolling Returns Comparison
                        st.markdown(f"##### 📈 Rolling Returns: Gross vs Net ({window_years}-year window)")
                        
                        fig_roll_compare = go.Figure()
                        
                        fig_roll_compare.add_trace(go.Scatter(
                            x=roll_ret_gross.index,
                            y=roll_ret_gross * 100,
                            name='Gross (theoretical)',
                            mode='lines',
                            line=dict(color='rgba(99, 102, 241, 0.7)', width=2, dash='dash'),
                            hovertemplate='<b>Gross</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                        ))
                        
                        fig_roll_compare.add_trace(go.Scatter(
                            x=roll_ret_net.index,
                            y=roll_ret_net * 100,
                            name='Net (after costs)',
                            mode='lines',
                            line=dict(color='#4ECDC4', width=2.5),
                            hovertemplate='<b>Net</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                        ))
                        
                        fig_roll_compare.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                        fig_roll_compare.update_layout(
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Annualized Return (%)",
                            hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                        )
                        fig_roll_compare = apply_plotly_theme(fig_roll_compare)
                        st.plotly_chart(fig_roll_compare, use_container_width=True)
                        
                        # Chart 2: Rolling Cost Drag
                        st.markdown(f"##### 💸 Rolling Cost Drag ({window_years}-year window)")
                        st.caption("Shows the annualized return lost to transaction costs over rolling periods.")
                        
                        fig_cost_drag = go.Figure()
                        
                        fig_cost_drag.add_trace(go.Scatter(
                            x=roll_cost_drag.index,
                            y=roll_cost_drag * 100,
                            name='Cost Drag',
                            mode='lines',
                            fill='tozeroy',
                            line=dict(color='#FF6B6B', width=2),
                            fillcolor='rgba(255, 107, 107, 0.3)',
                            hovertemplate='<b>Cost Drag</b><br>Date: %{x}<br>Drag: %{y:.2f}%/yr<extra></extra>'
                        ))
                        
                        fig_cost_drag.update_layout(
                            height=300,
                            xaxis_title="Date",
                            yaxis_title="Cost Drag (%/year)",
                            hovermode='x unified'
                        )
                        fig_cost_drag = apply_plotly_theme(fig_cost_drag)
                        st.plotly_chart(fig_cost_drag, use_container_width=True)
                        
                        # Chart 3: Rolling Volatility Comparison
                        st.markdown(f"##### 📊 Rolling Volatility: Gross vs Net ({window_years}-year window)")
                        st.caption("Volatility should be nearly identical - small differences come from timing of cost deductions.")
                        
                        fig_vol_compare = go.Figure()
                        
                        fig_vol_compare.add_trace(go.Scatter(
                            x=roll_vol_gross.index,
                            y=roll_vol_gross * 100,
                            name='Gross',
                            mode='lines',
                            line=dict(color='rgba(99, 102, 241, 0.7)', width=2, dash='dash'),
                            hovertemplate='<b>Gross</b><br>Date: %{x}<br>Vol: %{y:.2f}%<extra></extra>'
                        ))
                        
                        fig_vol_compare.add_trace(go.Scatter(
                            x=roll_vol_net.index,
                            y=roll_vol_net * 100,
                            name='Net',
                            mode='lines',
                            line=dict(color='#4ECDC4', width=2.5),
                            hovertemplate='<b>Net</b><br>Date: %{x}<br>Vol: %{y:.2f}%<extra></extra>'
                        ))
                        
                        fig_vol_compare.update_layout(
                            height=350,
                            xaxis_title="Date",
                            yaxis_title="Annualized Volatility (%)",
                            hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                        )
                        fig_vol_compare = apply_plotly_theme(fig_vol_compare)
                        st.plotly_chart(fig_vol_compare, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("##### 📋 Cost Impact Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Avg Cost Drag",
                                f"{roll_cost_drag.mean()*100:.2f}%/yr"
                            )
                        with col2:
                            st.metric(
                                "Max Cost Drag",
                                f"{roll_cost_drag.max()*100:.2f}%/yr",
                                help="Worst rolling period for cost impact"
                            )
                        with col3:
                            st.metric(
                                "Min Cost Drag",
                                f"{roll_cost_drag.min()*100:.2f}%/yr",
                                help="Best rolling period for cost impact"
                            )
                        with col4:
                            cost_drag_vol = roll_cost_drag.std() * 100
                            st.metric(
                                "Cost Drag Volatility",
                                f"{cost_drag_vol:.2f}%",
                                help="How variable is the cost impact over time"
                            )
                        
                        # Interpretation
                        st.markdown("""
                        ---
                        **How to interpret:**
                        - **Stable cost drag** (flat red area) = consistent trading activity across market conditions
                        - **Variable cost drag** (spiky red area) = rebalancing triggered more in certain periods
                        - **Higher cost drag during volatility** = threshold-based rebalancing triggers more often in turbulent markets
                        - **Gross ≈ Net volatility** = costs mainly affect returns, not risk profile
                        """)
        
        # TAB 4: DEEP-DIVE STATISTICS
        with tab4:
            st.markdown("### 🔬 Deep-Dive Statistics")
            st.markdown("""
            Statistical analysis of individual assets and portfolio dynamics. 
            This tab helps you understand the **risk drivers** behind your portfolio returns.
            """)
            
            # Create two main sections
            analysis_mode = st.radio(
                "Select Analysis Mode",
                ["📈 Single Asset Analysis", "🔗 Portfolio Correlation Dynamics", "🔄 Mean-Reversion & Dynamics"],
                horizontal=True,
                key="deepdive_mode"
            )
            
            st.markdown("---")
            
            # ================================================================
            # SECTION 1: SINGLE ASSET ANALYSIS
            # ================================================================
            if analysis_mode == "📈 Single Asset Analysis":
                
                # Asset selector
                asset_options = [(t, get_display_name(t)) for t in symbols]
                selected_asset = st.selectbox(
                    "Select Asset for Analysis",
                    options=[t[0] for t in asset_options],
                    format_func=lambda x: get_display_name(x),
                    key="deepdive_asset"
                )
                
                if selected_asset:
                    # Get price data for selected asset
                    prices = analyzer.data[selected_asset]
                    
                    # Calculate risk driver (log-values)
                    x_stock = np.log(prices.values)
                    dates_idx = prices.index
                    
                    # Calculate log-returns (compounded returns)
                    delta_x = np.diff(x_stock)
                    delta_x_abs = np.abs(delta_x)
                    
                    st.markdown("---")
                    
                    # ===== SECTION 1.1: RISK DRIVER =====
                    st.markdown(f"#### 📈 Risk Driver: Log-values of {get_display_name(selected_asset)}")
                    
                    with st.expander("💡 What is a Risk Driver?", expanded=False):
                        st.markdown("""
                        **Risk drivers** are the fundamental quantities that determine asset prices. 
                        For stocks, we use **log-prices** because:
                        
                        1. **Returns become additive**: Log-returns over multiple periods simply sum up
                        2. **Percentage changes**: Small log-returns ≈ percentage returns
                        3. **Statistical properties**: Log-returns are more likely to be stationary
                        
                        The relationship is: $x_t = \log(P_t)$ where $P_t$ is the price.
                        
                        📖 *Reference: Meucci, A. (2005). "Risk and Asset Allocation." Springer.*
                        """)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=dates_idx, y=x_stock, mode='lines',
                            line=dict(color=CHART_COLORS[0], width=2),
                            hovertemplate='Date: %{x}<br>Log-value: %{y:.4f}<extra></extra>'
                        ))
                        fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Log-values")
                        fig = apply_plotly_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Basic statistics
                        st.markdown("##### 📊 Summary Statistics")
                        
                        # Calculate statistics
                        mean_ret = np.mean(delta_x)
                        std_ret = np.std(delta_x)
                        skew_ret = stats.skew(delta_x)
                        kurt_ret = stats.kurtosis(delta_x)
                        
                        stats_data = {
                            'Metric': ['Mean Log-Return', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max'],
                            'Value': [
                                f"{mean_ret:.6f}",
                                f"{std_ret:.6f}",
                                f"{skew_ret:.4f}",
                                f"{kurt_ret:.4f}",
                                f"{np.min(delta_x):.4f}",
                                f"{np.max(delta_x):.4f}"
                            ]
                        }
                        st.markdown(create_styled_table(pd.DataFrame(stats_data)), unsafe_allow_html=True)
                        
                        # Quick interpretation
                        st.markdown("##### 🎯 Quick Interpretation")
                        
                        ann_return = mean_ret * 252 * 100
                        ann_vol = std_ret * np.sqrt(252) * 100
                        
                        st.markdown(f"• **Annualized Return**: {ann_return:.2f}%")
                        st.markdown(f"• **Annualized Volatility**: {ann_vol:.2f}%")
                        
                        if skew_ret < -0.5:
                            st.markdown("• **Skewness**: ⚠️ Negative skew → larger left tail (crash risk)")
                        elif skew_ret > 0.5:
                            st.markdown("• **Skewness**: ✅ Positive skew → larger right tail")
                        else:
                            st.markdown("• **Skewness**: ~ Symmetric distribution")
                        
                        if kurt_ret > 1:
                            st.markdown(f"• **Kurtosis**: ⚠️ Fat tails (excess={kurt_ret:.2f}) → extreme events more likely than Normal")
                        else:
                            st.markdown("• **Kurtosis**: ~ Near-normal tails")
                    
                    st.markdown("---")
                    
                    # ===== SECTION 1.2: LOG-RETURNS =====
                    st.markdown("#### 📉 Compounded Returns (Log-Returns)")
                    
                    with st.expander("💡 Why Log-Returns?", expanded=False):
                        st.markdown("""
                        **Log-returns** (also called *continuously compounded returns*) are defined as:
                        
                        $$r_t = \log(P_t) - \log(P_{t-1}) = \log(P_t / P_{t-1})$$
                        
                        **Advantages over simple returns:**
                        - **Time-additivity**: $r_{t:t+k} = r_t + r_{t+1} + ... + r_{t+k}$
                        - **Symmetry**: A +10% followed by -10% doesn't return to original price with simple returns, but is symmetric with log-returns
                        - **Statistical modeling**: More amenable to statistical analysis
                        
                        For small returns, log-returns ≈ simple returns.
                        """)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates_idx[1:], y=delta_x, mode='markers',
                        marker=dict(size=3, color=CHART_COLORS[1], opacity=0.7),
                        hovertemplate='Date: %{x}<br>Return: %{y:.4f}<extra></extra>'
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    fig.add_hline(y=np.mean(delta_x), line_dash="dot", line_color="#4ECDC4",
                                annotation_text="Mean", annotation_position="right")
                    fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Log-Return")
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # ===== SECTION 1.3: INVARIANCE TESTS =====
                    st.markdown("#### 🧪 Invariance Tests")
                    
                    with st.expander("💡 What are Invariance Tests?", expanded=False):
                        st.markdown("""
                        **Invariants** are variables whose distribution doesn't change over time (IID: independent and identically distributed).
                        
                        **Why do we care?**
                        - Most statistical models assume IID data
                        - If returns are NOT IID, we need more sophisticated models (like GARCH)
                        - Autocorrelation in returns → potential predictability (or model misspecification)
                        - Autocorrelation in |returns| → **volatility clustering** (common in financial data)
                        
                        **The tests:**
                        - We compute autocorrelation at various lags
                        - If autocorrelations fall within the confidence bands, the series is consistent with IID
                        
                        📖 *Reference: Meucci, A. (2010). "Quant Nugget 2: Linear vs. Compounded Returns"*
                        """)
                    
                    l_bar = st.slider("Maximum lag for autocorrelation", 5, 50, 25, key="lag_slider")
                    conf_lev = 0.95
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Ellipsoid test on returns
                        st.markdown("##### Autocorrelation of Returns")
                        acf, conf_int, test_passed = invariance_test_ellipsoid(delta_x, l_bar, conf_lev)
                        
                        fig = go.Figure()
                        
                        # Color bars based on significance
                        colors = ['#FF6B6B' if abs(a) > conf_int else CHART_COLORS[2] for a in acf]
                        
                        fig.add_trace(go.Bar(
                            x=list(range(1, l_bar+1)), y=acf,
                            marker_color=colors, name='ACF'
                        ))
                        fig.add_hline(y=conf_int, line_dash="dash", line_color="#FFE66D",
                                    annotation_text=f"+{conf_lev*100:.0f}% CI")
                        fig.add_hline(y=-conf_int, line_dash="dash", line_color="#FFE66D",
                                    annotation_text=f"-{conf_lev*100:.0f}% CI")
                        fig.add_hline(y=0, line_color="rgba(255,255,255,0.5)")
                        fig.update_layout(height=300, xaxis_title="Lag", yaxis_title="Autocorrelation",
                                        yaxis_range=[-0.3, 0.3])
                        fig = apply_plotly_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if test_passed:
                            st.success("✅ Returns appear IID (all ACF within confidence bands)")
                            st.markdown("*→ Simple models may be adequate for this asset*")
                        else:
                            st.warning("⚠️ Some autocorrelations exceed confidence bands")
                            st.markdown("*→ Returns show some predictability or the sample has anomalies*")
                    
                    with col2:
                        # Ellipsoid test on absolute returns (volatility clustering)
                        st.markdown("##### Autocorrelation of |Returns| (Volatility Clustering)")
                        acf_abs, conf_int_abs, test_passed_abs = invariance_test_ellipsoid(delta_x_abs, l_bar, conf_lev)
                        
                        fig = go.Figure()
                        
                        colors_abs = ['#FF6B6B' if abs(a) > conf_int_abs else CHART_COLORS[3] for a in acf_abs]
                        
                        fig.add_trace(go.Bar(
                            x=list(range(1, l_bar+1)), y=acf_abs,
                            marker_color=colors_abs, name='ACF |r|'
                        ))
                        fig.add_hline(y=conf_int_abs, line_dash="dash", line_color="#FFE66D")
                        fig.add_hline(y=-conf_int_abs, line_dash="dash", line_color="#FFE66D")
                        fig.add_hline(y=0, line_color="rgba(255,255,255,0.5)")
                        fig.update_layout(height=300, xaxis_title="Lag", yaxis_title="Autocorrelation",
                                        yaxis_range=[-0.1, 0.5])
                        fig = apply_plotly_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if test_passed_abs:
                            st.success("✅ No volatility clustering detected")
                            st.markdown("*→ Volatility appears constant over time*")
                        else:
                            st.info("ℹ️ **Volatility clustering detected**")
                            st.markdown("""
                            *→ Large moves tend to follow large moves (regardless of direction)*
                            
                            *→ This is typical for financial assets and motivates GARCH modeling below*
                            """)
                    
                    st.markdown("---")
                    
                    # ===== SECTION 1.4: DISTRIBUTION ANALYSIS =====
                    st.markdown("#### 📊 Distribution Analysis")
                    
                    with st.expander("💡 Why Test for Normality?", expanded=False):
                        st.markdown("""
                        **Many financial models assume Normal (Gaussian) returns:**
                        - Mean-variance optimization (Markowitz)
                        - Black-Scholes option pricing
                        - Traditional VaR calculations
                        
                        **Reality check:**
                        - Financial returns typically have **fat tails** (more extreme events than Normal predicts)
                        - They often show **negative skewness** (crashes are more severe than rallies)
                        - This is why CVaR and robust methods are important!
                        
                        **The tests:**
                        - **Kolmogorov-Smirnov**: Compares empirical CDF to theoretical Normal CDF
                        - **Jarque-Bera**: Tests if skewness and kurtosis match Normal distribution
                        - **Shapiro-Wilk**: Most powerful test for small samples
                        
                        *p-value > 0.05 means we cannot reject normality (doesn't prove it's Normal!)*
                        """)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram with normal overlay
                        st.markdown("##### Return Distribution vs Normal")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=delta_x, nbinsx=50, name='Returns',
                            marker_color=CHART_COLORS[4], opacity=0.7, histnorm='probability density'
                        ))
                        
                        # Normal overlay
                        x_range = np.linspace(delta_x.min(), delta_x.max(), 100)
                        normal_pdf = stats.norm.pdf(x_range, np.mean(delta_x), np.std(delta_x))
                        fig.add_trace(go.Scatter(
                            x=x_range, y=normal_pdf, mode='lines',
                            name='Normal', line=dict(color='#FFFFFF', width=2)
                        ))
                        
                        fig.update_layout(
                            height=300, xaxis_title="Log-Return", yaxis_title="Density",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02)
                        )
                        fig = apply_plotly_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Normality tests
                        st.markdown("##### Normality Tests")
                        
                        ks_stat, ks_pval = ks_test(delta_x)
                        jb_stat, jb_pval = stats.jarque_bera(delta_x)
                        
                        if len(delta_x) > 5000:
                            sw_sample = np.random.choice(delta_x, 5000, replace=False)
                        else:
                            sw_sample = delta_x
                        sw_stat, sw_pval = stats.shapiro(sw_sample)
                        
                        test_results = {
                            'Test': ['Kolmogorov-Smirnov', 'Jarque-Bera', 'Shapiro-Wilk'],
                            'Statistic': [f"{ks_stat:.4f}", f"{jb_stat:.2f}", f"{sw_stat:.4f}"],
                            'p-value': [f"{ks_pval:.4f}", f"{jb_pval:.4f}", f"{sw_pval:.4f}"],
                            'Normal?': [
                                "✅ Yes" if ks_pval > 0.05 else "❌ No",
                                "✅ Yes" if jb_pval > 0.05 else "❌ No",
                                "✅ Yes" if sw_pval > 0.05 else "❌ No"
                            ]
                        }
                        st.markdown(create_styled_table(pd.DataFrame(test_results)), unsafe_allow_html=True)
                        
                        # Summary interpretation
                        n_normal = sum([ks_pval > 0.05, jb_pval > 0.05, sw_pval > 0.05])
                        
                        st.markdown("##### 🎯 Interpretation")
                        if n_normal == 3:
                            st.success("All tests suggest returns are consistent with Normal distribution")
                            st.markdown("*→ Standard mean-variance methods may be appropriate*")
                        elif n_normal >= 1:
                            st.warning("Mixed results - some evidence against Normality")
                            st.markdown("*→ Consider robust methods (CVaR) for risk management*")
                        else:
                            st.error("Strong evidence against Normality")
                            st.markdown("""
                            *→ Returns have fat tails and/or skewness*
                            
                            *→ Mean-variance may underestimate tail risk*
                            
                            *→ **Recommendation**: Use CVaR optimization and GARCH modeling*
                            """)
                    
                    st.markdown("---")
                    
                    # ===== SECTION 1.5: GARCH ANALYSIS =====
                    if ARCH_AVAILABLE:
                        st.markdown("#### ⚡ GARCH(1,1) Volatility Model")
                        
                        with st.expander("💡 What is GARCH?", expanded=False):
                            st.markdown("""
                            **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity) models 
                            capture **time-varying volatility** in financial returns.
                            
                            **The GARCH(1,1) model:**
                            
                            $$r_t = \\mu + \\varepsilon_t, \\quad \\varepsilon_t = \\sigma_t z_t, \\quad z_t \\sim N(0,1)$$
                            
                            $$\\sigma_t^2 = \\omega + \\alpha \\varepsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$
                            
                            **Parameters:**
                            - **ω (omega)**: Long-run variance constant
                            - **α (alpha)**: Reaction to recent shocks (ARCH term)
                            - **β (beta)**: Persistence of past volatility (GARCH term)
                            - **α + β**: Total persistence (< 1 for stationarity)
                            
                            **Why use GARCH?**
                            - Captures volatility clustering
                            - Produces **standardized residuals** (quasi-invariants) that are closer to IID
                            - Better risk forecasting than constant volatility models
                            
                            📖 *Reference: Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"*
                            """)
                        
                        with st.spinner("Fitting GARCH(1,1) model..."):
                            params, std_resid, cond_vol = fit_garch(delta_x)
                        
                        if params is not None:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("##### Model Parameters")
                                
                                persistence = params['alpha[1]'] + params['beta[1]']
                                
                                param_data = {
                                    'Parameter': ['μ (mean)', 'ω (constant)', 'α (ARCH)', 'β (GARCH)', 'Persistence (α+β)'],
                                    'Value': [
                                        f"{params['mu']:.6f}",
                                        f"{params['omega']:.8f}",
                                        f"{params['alpha[1]']:.4f}",
                                        f"{params['beta[1]']:.4f}",
                                        f"{persistence:.4f}"
                                    ]
                                }
                                st.markdown(create_styled_table(pd.DataFrame(param_data)), unsafe_allow_html=True)
                                
                                # Interpretation
                                st.markdown("##### 🎯 Parameter Interpretation")
                                
                                if params['alpha[1]'] > 0.15:
                                    st.markdown(f"• **α = {params['alpha[1]']:.3f}**: High reactivity to shocks")
                                else:
                                    st.markdown(f"• **α = {params['alpha[1]']:.3f}**: Moderate reactivity")
                                
                                if params['beta[1]'] > 0.9:
                                    st.markdown(f"• **β = {params['beta[1]']:.3f}**: Very persistent volatility")
                                elif params['beta[1]'] > 0.7:
                                    st.markdown(f"• **β = {params['beta[1]']:.3f}**: Moderately persistent")
                                else:
                                    st.markdown(f"• **β = {params['beta[1]']:.3f}**: Low persistence")
                                
                                if persistence > 0.98:
                                    st.warning(f"⚠️ **Persistence = {persistence:.3f}**: Near unit root → volatility shocks are very long-lasting")
                                elif persistence > 0.9:
                                    st.info(f"ℹ️ **Persistence = {persistence:.3f}**: Typical for stocks → shocks decay over weeks/months")
                                else:
                                    st.success(f"✅ **Persistence = {persistence:.3f}**: Faster mean-reversion of volatility")
                            
                            with col2:
                                st.markdown("##### Conditional Volatility Over Time")
                                
                                # Annualized conditional volatility
                                ann_cond_vol = cond_vol * np.sqrt(252) * 100
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=dates_idx[1:], y=ann_cond_vol,
                                    mode='lines', line=dict(color=CHART_COLORS[5], width=1.5),
                                    hovertemplate='Date: %{x}<br>Ann. Vol: %{y:.2f}%<extra></extra>'
                                ))
                                
                                # Add mean line
                                fig.add_hline(y=np.mean(ann_cond_vol), line_dash="dash", line_color="#4ECDC4",
                                            annotation_text=f"Mean: {np.mean(ann_cond_vol):.1f}%")
                                
                                fig.update_layout(height=280, xaxis_title="Date", yaxis_title="Annualized Vol (%)")
                                fig = apply_plotly_theme(fig)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Current vs historical vol
                                current_vol = ann_cond_vol[-1]
                                vol_percentile = stats.percentileofscore(ann_cond_vol, current_vol)
                                
                                st.markdown(f"**Current volatility**: {current_vol:.1f}% ({vol_percentile:.0f}th percentile)")
                                
                                if vol_percentile > 80:
                                    st.warning("⚠️ Volatility is elevated compared to history")
                                elif vol_percentile < 20:
                                    st.success("✅ Volatility is low compared to history")
                                else:
                                    st.info("ℹ️ Volatility is in normal range")
                            
                            # Standardized residuals section
                            st.markdown("##### 📐 Standardized Residuals (Quasi-Invariants)")
                            
                            with st.expander("💡 What are Quasi-Invariants?", expanded=False):
                                st.markdown("""
                                **Quasi-invariants** are the standardized residuals from the GARCH model:
                                
                                $$z_t = \\varepsilon_t / \\sigma_t = (r_t - \\mu) / \\sigma_t$$
                                
                                If the GARCH model is correct, these should be approximately **IID**.
                                
                                **Why do we care?**
                                - They are the "true" invariants after removing volatility dynamics
                                - They can be used for portfolio simulation (Copula-Marginal approach)
                                - If they're not IID, we need a more complex model
                                
                                📖 *Reference: Meucci, A. (2009). "Managing Diversification." Risk Magazine.*
                                """)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=dates_idx[1:], y=std_resid, mode='markers',
                                    marker=dict(size=2, color=CHART_COLORS[6], opacity=0.6)
                                ))
                                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                                fig.add_hline(y=2, line_dash="dot", line_color="#FF6B6B", annotation_text="+2σ")
                                fig.add_hline(y=-2, line_dash="dot", line_color="#FF6B6B", annotation_text="-2σ")
                                fig.update_layout(height=250, xaxis_title="Date", yaxis_title="Std. Residual")
                                fig = apply_plotly_theme(fig)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # ACF of squared standardized residuals
                                st.markdown("**ACF of ε² (should be ~0 if GARCH is adequate)**")
                                acf_resid2, conf_resid, resid_iid = invariance_test_ellipsoid(std_resid**2, 20, 0.95)
                                
                                fig = go.Figure()
                                colors_resid = ['#FF6B6B' if abs(a) > conf_resid else CHART_COLORS[7] for a in acf_resid2]
                                fig.add_trace(go.Bar(
                                    x=list(range(1, 21)), y=acf_resid2,
                                    marker_color=colors_resid, name='ACF ε²'
                                ))
                                fig.add_hline(y=conf_resid, line_dash="dash", line_color="#FFE66D")
                                fig.add_hline(y=-conf_resid, line_dash="dash", line_color="#FFE66D")
                                fig.update_layout(height=250, xaxis_title="Lag", yaxis_title="ACF of ε²",
                                                yaxis_range=[-0.15, 0.15])
                                fig = apply_plotly_theme(fig)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            if resid_iid:
                                st.success("✅ GARCH(1,1) adequately captures volatility dynamics - residuals appear IID")
                            else:
                                st.warning("⚠️ Some residual autocorrelation remains - consider GARCH(1,2) or EGARCH")
                            
                            # Practical implications box
                            st.markdown("---")
                            st.markdown("##### 💼 Practical Implications for Portfolio Management")
                            
                            implications = []
                            
                            if persistence > 0.95:
                                implications.append("• **Volatility regimes persist**: When vol is high, expect it to stay high for weeks")
                            
                            if current_vol > np.percentile(ann_cond_vol, 75):
                                implications.append("• **Current high volatility**: Consider reducing position size or hedging")
                            elif current_vol < np.percentile(ann_cond_vol, 25):
                                implications.append("• **Current low volatility**: Good entry point, but vol can spike suddenly")
                            
                            if not test_passed_abs:
                                implications.append("• **Volatility clustering confirmed**: GARCH forecasts are more reliable than constant vol")
                            
                            if n_normal < 2:
                                implications.append("• **Non-normal returns**: Use CVaR instead of VaR for risk measurement")
                            
                            if implications:
                                for imp in implications:
                                    st.markdown(imp)
                            else:
                                st.markdown("• Asset shows typical characteristics - standard models should work well")
                        
                        else:
                            st.warning("⚠️ Could not fit GARCH model to this data")
                    else:
                        st.info("ℹ️ Install `arch` package for GARCH analysis: `pip install arch`")
        # ================================================================
            # SECTION 2: PORTFOLIO CORRELATION DYNAMICS
            # ================================================================
            elif:  # analysis_mode == "🔗 Portfolio Correlation Dynamics"
                
                st.markdown("""
                ### What is this analysis for?
                
                This section answers a critical question for portfolio management: 
                **"How correlated are my assets right now, and is this different from normal?"**
                
                Why does this matter? Because:
                - **High correlations** → Your assets move together → Less diversification → Higher portfolio risk
                - **Low correlations** → Your assets move independently → More diversification → Lower portfolio risk
                
                The problem is that correlations **change over time**, especially during market stress when 
                you need diversification the most.
                """)
                
                with st.expander("📚 The Science Behind This Analysis", expanded=False):
                    st.markdown("""
                    We use the **Dynamic Conditional Correlation (DCC)** model from Engle (2002), 
                    which is the industry standard for modeling time-varying correlations.
                    
                    **Why not just use rolling correlation?**
                    
                    Rolling correlation (e.g., 60-day window) has problems:
                    1. All days in the window count equally (day 1 = day 60)
                    2. Old observations "fall off" abruptly
                    3. Mixes volatility effects with true correlation changes

                    **The DCC (Dynamic Conditional Correlation) model:**
                    
                    Engle (2002) proposed modeling correlations as time-varying:
                    
                    $$Q_t = (1-a-b)\\bar{\\rho} + a \\cdot \\varepsilon_{t-1}\\varepsilon_{t-1}' + b \\cdot Q_{t-1}$$
                    
                    $$R_t = \\text{diag}(Q_t)^{-1/2} \\cdot Q_t \\cdot \\text{diag}(Q_t)^{-1/2}$$
                    
                    Where:
                    - $\\bar{\\rho}$: Long-run (unconditional) correlation
                    - $a$: Reaction to recent shocks
                    - $b$: Persistence of correlation
                    - $R_t$: Time-varying correlation matrix
                    
                    **This approach incorporates:**
                    1. **Flexible Probabilities**: Recent data gets more weight (exponential decay)
                    2. **GARCH filtering**: Separates volatility from correlation
                    3. **Parametric model**: Smooth estimates with mean-reversion
                    
                    📖 *References:*
                    - *Engle, R. (2002). "Dynamic Conditional Correlation." Journal of Business & Economic Statistics, 20(3), 339-350.*
                    - *Meucci, A. (2010). "Historical Scenarios with Fully Flexible Probabilities." GARP Risk Professional.*
                    """)
                
                st.markdown("---")
                
                # Asset selection for DCC
                st.markdown("#### 🎯 Select Assets for Analysis")
                
                available_assets = [(t, get_display_name(t)) for t in symbols]
                
                selected_dcc_assets = st.multiselect(
                    "Choose 2-10 assets",
                    options=[t[0] for t in available_assets],
                    default=[],
                    format_func=lambda x: get_display_name(x),
                    key="dcc_assets",
                    help="The analysis works best with 3-10 assets. More assets = richer analysis but longer computation."
                )
                
                if len(selected_dcc_assets) < 2:
                    st.info("👆 Select at least 2 assets to analyze how their correlations evolve over time.")
                    
                elif len(selected_dcc_assets) > 10:
                    st.warning("⚠️ Please select 10 or fewer assets for reliable estimation.")
                    
                else:
                    # Configuration
                    with st.expander("⚙️ Model Settings", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            tau_hl = st.slider(
                                "Half-life (days)",
                                min_value=30, max_value=252, value=120, step=10,
                                key="dcc_halflife",
                                help="How quickly old data loses importance. 120 days = data from 4 months ago has half the weight of today."
                            )
                            st.caption(f"With τ={tau_hl}: data from {tau_hl} days ago has 50% weight, {tau_hl*2} days ago has 25% weight")
                        
                        with col2:
                            nu_marginal = st.selectbox(
                                "Distribution assumption",
                                options=[("Normal (standard)", 1000), ("Student-t ν=6 (fat tails)", 6), ("Student-t ν=4 (fatter tails)", 4)],
                                format_func=lambda x: x[0],
                                index=0,
                                key="dcc_nu",
                                help="Student-t captures extreme events better but needs more data."
                            )
                            nu = nu_marginal[1]
                    
                    if st.button("🚀 Run Analysis", use_container_width=True, key="run_dcc"):
                        
                        n_assets = len(selected_dcc_assets)
                        
                        with st.spinner(f"Analyzing correlation dynamics for {n_assets} assets..."):
                            
                            try:
                                # Prepare returns data
                                returns_dcc = analyzer.returns[selected_dcc_assets].dropna()
                                t_bar = len(returns_dcc)
                                
                                # Compute flexible probabilities
                                p_flex = compute_flexible_probabilities(t_bar, tau_hl)
                                
                                # Extract GARCH residuals
                                eps, cond_vols, garch_params = extract_garch_residuals(returns_dcc, p_flex)
                                
                                # Compute MLFP location-dispersion for each marginal
                                mu_marg = np.zeros(n_assets)
                                sigma2_marg = np.zeros(n_assets)
                                
                                for i in range(n_assets):
                                    mu_marg[i], sigma2_marg[i] = fit_locdisp_mlfp(eps[:, i], p=p_flex, nu=nu)
                                
                                # Transform to standard normal (copula approach)
                                eps_tilde = np.zeros_like(eps)
                                for i in range(n_assets):
                                    u = t_dist.cdf(eps[:, i], df=nu, loc=mu_marg[i], scale=np.sqrt(sigma2_marg[i]))
                                    u = np.clip(u, 1e-7, 1 - 1e-7)
                                    eps_tilde[:, i] = t_dist.ppf(u, df=1000)
                                
                                # Compute unconditional correlation
                                _, sigma2_eps_tilde = fit_locdisp_mlfp(eps_tilde, p=p_flex, nu=1000)
                                rho2_uncond = np.diag(1/np.sqrt(np.diag(sigma2_eps_tilde))) @ sigma2_eps_tilde @ np.diag(1/np.sqrt(np.diag(sigma2_eps_tilde)))
                                
                                # Fit DCC model
                                dcc_params, r2_t, eps_dcc, q2_final = fit_dcc_t(eps_tilde, p_flex, rho2=rho2_uncond)
                                
                                # Store results in session state
                                st.session_state.dcc_results = {
                                    'params': dcc_params,
                                    'r2_t': r2_t,
                                    'rho2_uncond': rho2_uncond,
                                    'eps': eps,
                                    'eps_tilde': eps_tilde,
                                    'cond_vols': cond_vols,
                                    'garch_params': garch_params,
                                    'dates': returns_dcc.index,
                                    'assets': selected_dcc_assets,
                                    'q2_final': q2_final
                                }
                                
                                st.success("✅ Analysis complete!")
                                
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {str(e)}")
                                st.session_state.dcc_results = None
                    
                    # Display results if available
                    if 'dcc_results' in st.session_state and st.session_state.dcc_results is not None:
                        
                        dcc = st.session_state.dcc_results
                        r2_t = dcc['r2_t']
                        rho2_uncond = dcc['rho2_uncond']
                        dates = dcc['dates']
                        assets = dcc['assets']
                        n_assets = len(assets)
                        c, a, b = dcc['params']
                        
                        st.markdown("---")
                        
                        # ================================================================
                        # KEY FINDING 1: CURRENT CORRELATION REGIME
                        # ================================================================
                        st.markdown("## 📊 Key Finding 1: What's Your Current Correlation Regime?")
                        
                        # Compute average correlation (current vs long-run)
                        avg_corr_current = np.mean(r2_t[-1][np.triu_indices(n_assets, k=1)])
                        avg_corr_longrun = np.mean(rho2_uncond[np.triu_indices(n_assets, k=1)])
                        corr_diff = avg_corr_current - avg_corr_longrun
                        
                        # Compute historical average correlation series for percentile
                        avg_corr_series = np.zeros(len(dates))
                        for t in range(len(dates)):
                            upper_tri_t = r2_t[t][np.triu_indices(n_assets, k=1)]
                            avg_corr_series[t] = np.mean(upper_tri_t)
                        
                        corr_percentile = stats.percentileofscore(avg_corr_series, avg_corr_current)
                        
                        # Display regime
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Current Avg. Correlation",
                                f"{avg_corr_current:.2f}",
                                delta=f"{corr_diff:+.2f} vs long-run",
                                delta_color="inverse"  # Red when positive (bad for diversification)
                            )
                        
                        with col2:
                            st.metric(
                                "Long-run Average",
                                f"{avg_corr_longrun:.2f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Current Percentile",
                                f"{corr_percentile:.0f}%",
                                help="Where current correlation stands vs history. 90% = correlations are higher than 90% of historical observations."
                            )
                        
                        # INTERPRETATION BOX
                        st.markdown("#### 🎯 What does this mean for your portfolio?")
                        
                        if corr_diff > 0.10:
                            st.error(f"""
                            **⚠️ HIGH CORRELATION REGIME**
                            
                            Your portfolio's average correlation ({avg_corr_current:.2f}) is **significantly above** 
                            the long-run average ({avg_corr_longrun:.2f}).
                            
                            **Practical implications:**
                            - Your assets are currently moving together more than usual
                            - **Diversification is less effective right now**
                            - Portfolio volatility is likely **higher than your historical estimates suggest**
                            - VaR/CVaR calculated with historical correlation may **underestimate current risk**
                            
                            **Possible actions:**
                            1. Reduce overall portfolio size to maintain the same risk level
                            2. Consider adding assets with lower correlation to your current holdings
                            3. Review hedge ratios if you're hedging (they may need adjustment)
                            """)
                            
                        elif corr_diff < -0.10:
                            st.success(f"""
                            **✅ LOW CORRELATION REGIME**
                            
                            Your portfolio's average correlation ({avg_corr_current:.2f}) is **significantly below** 
                            the long-run average ({avg_corr_longrun:.2f}).
                            
                            **Practical implications:**
                            - Your assets are moving more independently than usual
                            - **Diversification is working well right now**
                            - Portfolio volatility is likely **lower than historical estimates suggest**
                            
                            **Possible actions:**
                            1. This is generally a favorable environment for diversified portfolios
                            2. You may have room to take slightly more risk if desired
                            3. Be aware this regime may not persist (see persistence analysis below)
                            """)
                            
                        else:
                            st.info(f"""
                            **ℹ️ NORMAL CORRELATION REGIME**
                            
                            Your portfolio's average correlation ({avg_corr_current:.2f}) is **near** 
                            the long-run average ({avg_corr_longrun:.2f}).
                            
                            **Practical implications:**
                            - Correlations are behaving as expected historically
                            - Your historical risk estimates should be reasonable
                            - Standard portfolio assumptions are appropriate
                            """)
                        
                        st.markdown("---")
                        
                        # ================================================================
                        # KEY FINDING 2: HOW PERSISTENT IS THIS REGIME?
                        # ================================================================
                        st.markdown("## ⏳ Key Finding 2: How Long Will This Regime Last?")
                        
                        persistence = a + b
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Persistence (a+b)", f"{persistence:.3f}")
                        with col2:
                            st.metric("Shock Reaction (a)", f"{a:.3f}")
                        with col3:
                            st.metric("Memory (b)", f"{b:.3f}")
                        
                        # Calculate half-life of correlation shocks
                        if persistence < 1:
                            shock_halflife = np.log(0.5) / np.log(persistence)
                            shock_halflife_days = abs(shock_halflife)
                        else:
                            shock_halflife_days = float('inf')
                        
                        st.markdown("#### 🎯 What does this mean?")
                        
                        if persistence > 0.98:
                            st.warning(f"""
                            **Very High Persistence ({persistence:.3f})**
                            
                            Correlation shocks take a **very long time to fade** (half-life ≈ {shock_halflife_days:.0f} days).
                            
                            **Practical implication:** If correlations are currently elevated, **don't expect them to 
                            quickly return to normal**. The current regime is likely to persist for weeks or months.
                            
                            *This is typical for equity markets during stress periods (Engle, 2002).*
                            """)
                        elif persistence > 0.90:
                            st.info(f"""
                            **High Persistence ({persistence:.3f})**
                            
                            Correlation shocks fade gradually (half-life ≈ {shock_halflife_days:.0f} days).
                            
                            **Practical implication:** Correlation regimes last weeks to months. If you observe 
                            elevated correlations today, expect them to remain elevated for some time.
                            
                            *This is typical for most equity markets.*
                            """)
                        else:
                            st.success(f"""
                            **Moderate Persistence ({persistence:.3f})**
                            
                            Correlation shocks fade relatively quickly (half-life ≈ {shock_halflife_days:.0f} days).
                            
                            **Practical implication:** Correlations mean-revert faster than typical. Current 
                            deviations from the long-run average are likely temporary.
                            """)
                        
                        st.markdown("---")
                        
                        # ================================================================
                        # KEY FINDING 3: CORRELATION OVER TIME
                        # ================================================================
                        st.markdown("## 📈 Key Finding 3: How Have Correlations Evolved?")
                        
                        st.markdown("""
                        This chart shows the **average correlation across all asset pairs** over time.
                        It's the single best summary of "how correlated is my portfolio right now?"
                        """)
                        
                        # Average portfolio correlation over time
                        fig_avg = go.Figure()
                        
                        fig_avg.add_trace(go.Scatter(
                            x=dates, y=avg_corr_series,
                            mode='lines',
                            name='Portfolio Correlation',
                            line=dict(color=CHART_COLORS[0], width=2),
                            fill='tozeroy',
                            fillcolor='rgba(78, 205, 196, 0.2)',
                            hovertemplate='Date: %{x}<br>Avg Correlation: %{y:.3f}<extra></extra>'
                        ))
                        
                        # Long-run average
                        fig_avg.add_hline(
                            y=avg_corr_longrun, line_dash="dash", line_color="#FFE66D",
                            annotation_text=f"Long-run avg: {avg_corr_longrun:.2f}"
                        )
                        
                        # High correlation threshold
                        high_threshold = avg_corr_longrun + np.std(avg_corr_series)
                        fig_avg.add_hline(
                            y=high_threshold, line_dash="dot", line_color="#FF6B6B",
                            annotation_text=f"High regime: >{high_threshold:.2f}"
                        )
                        
                        fig_avg.update_layout(
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Average Pairwise Correlation",
                            yaxis_range=[0, 1.0],
                            title="Portfolio-Level Correlation Over Time"
                        )
                        fig_avg = apply_plotly_theme(fig_avg)
                        st.plotly_chart(fig_avg, use_container_width=True)
                        
                        # Identify high-correlation periods
                        high_corr_periods = avg_corr_series > high_threshold
                        pct_high_corr = np.mean(high_corr_periods) * 100
                        
                        st.markdown(f"""
                        **Reading this chart:**
                        - **Yellow dashed line**: Long-run average correlation
                        - **Red dotted line**: Threshold for "high correlation regime" (mean + 1 std)
                        - Historically, your portfolio has been in a high-correlation regime **{pct_high_corr:.1f}%** of the time
                        """)
                        
                        st.markdown("---")
                        
                        # ================================================================
                        # KEY FINDING 4: PAIR-BY-PAIR ANALYSIS
                        # ================================================================
                        st.markdown("## 🔍 Key Finding 4: Which Pairs Are Most Correlated?")
                        
                        col1, col2 = st.columns([1.3, 1])
                        
                        with col1:
                            # Heatmap of CURRENT correlations
                            st.markdown("##### Current Correlation Matrix")
                            
                            asset_names = [get_display_name(a) for a in assets]
                            current_corr_matrix = r2_t[-1]
                            
                            fig_heatmap = go.Figure(data=go.Heatmap(
                                z=current_corr_matrix,
                                x=asset_names,
                                y=asset_names,
                                colorscale='RdBu_r',
                                zmid=0,
                                zmin=-1, zmax=1,
                                text=np.round(current_corr_matrix, 2),
                                texttemplate='%{text}',
                                textfont=dict(size=10, color='white'),
                                hovertemplate='%{x} vs %{y}<br>Current ρ: %{z:.3f}<extra></extra>'
                            ))
                            
                            fig_heatmap.update_layout(
                                height=400,
                                xaxis=dict(tickangle=45),
                                yaxis=dict(autorange='reversed'),
                                title="Correlations RIGHT NOW"
                            )
                            fig_heatmap = apply_plotly_theme(fig_heatmap)
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        with col2:
                            st.markdown("##### Pairs to Watch")
                            
                            # Find highest and lowest correlation pairs
                            pairs_data = []
                            for i in range(n_assets):
                                for j in range(i+1, n_assets):
                                    current_rho = r2_t[-1, i, j]
                                    longrun_rho = rho2_uncond[i, j]
                                    diff = current_rho - longrun_rho
                                    pairs_data.append({
                                        'Pair': f"{get_display_name(assets[i])} / {get_display_name(assets[j])}",
                                        'Current ρ': current_rho,
                                        'Long-run ρ': longrun_rho,
                                        'Difference': diff,
                                        'i': i, 'j': j
                                    })
                            
                            pairs_df = pd.DataFrame(pairs_data)
                            
                            # Highest current correlations
                            st.markdown("**🔴 Highest correlations (least diversification):**")
                            top_corr = pairs_df.nlargest(3, 'Current ρ')
                            for _, row in top_corr.iterrows():
                                color = "🔴" if row['Current ρ'] > 0.7 else "🟡"
                                st.markdown(f"{color} **{row['Pair']}**: {row['Current ρ']:.2f}")
                            
                            st.markdown("")
                            
                            # Biggest increases vs long-run
                            st.markdown("**⬆️ Largest increases vs normal:**")
                            top_increase = pairs_df.nlargest(3, 'Difference')
                            for _, row in top_increase.iterrows():
                                if row['Difference'] > 0.05:
                                    st.markdown(f"⚠️ **{row['Pair']}**: +{row['Difference']:.2f} (now {row['Current ρ']:.2f} vs normal {row['Long-run ρ']:.2f})")
                            
                            if all(top_increase['Difference'] <= 0.05):
                                st.markdown("No significant increases detected.")
                        
                        st.markdown("---")
                        
                        # ================================================================
                        # DETAILED PAIR ANALYSIS
                        # ================================================================
                        st.markdown("## 📉 Detailed: Single Pair Over Time")
                        
                        st.markdown("Select a specific pair to see how their correlation has changed:")
                        
                        if n_assets > 2:
                            col1, col2 = st.columns(2)
                            with col1:
                                asset1_idx = st.selectbox(
                                    "First asset",
                                    options=list(range(n_assets)),
                                    format_func=lambda x: get_display_name(assets[x]),
                                    key="dcc_asset1"
                                )
                            with col2:
                                asset2_options = [i for i in range(n_assets) if i != asset1_idx]
                                asset2_idx = st.selectbox(
                                    "Second asset",
                                    options=asset2_options,
                                    format_func=lambda x: get_display_name(assets[x]),
                                    key="dcc_asset2"
                                )
                        else:
                            asset1_idx, asset2_idx = 0, 1
                        
                        # Extract pairwise correlations over time
                        corr_series_pair = r2_t[:, asset1_idx, asset2_idx]
                        uncond_corr_pair = rho2_uncond[asset1_idx, asset2_idx]
                        current_corr_pair = corr_series_pair[-1]
                        
                        fig_pair = go.Figure()
                        
                        fig_pair.add_trace(go.Scatter(
                            x=dates, y=corr_series_pair,
                            mode='lines',
                            name='Conditional Correlation',
                            line=dict(color=CHART_COLORS[1], width=2),
                            hovertemplate='Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>'
                        ))
                        
                        fig_pair.add_hline(
                            y=uncond_corr_pair, line_dash="dash", line_color="#FFE66D",
                            annotation_text=f"Long-run: {uncond_corr_pair:.2f}"
                        )
                        
                        fig_pair.update_layout(
                            height=350,
                            xaxis_title="Date",
                            yaxis_title="Correlation",
                            yaxis_range=[-0.2, 1.0],
                            title=f"Correlation: {get_display_name(assets[asset1_idx])} vs {get_display_name(assets[asset2_idx])}"
                        )
                        fig_pair = apply_plotly_theme(fig_pair)
                        st.plotly_chart(fig_pair, use_container_width=True)
                        
                        # Pair-specific interpretation
                        pair_diff = current_corr_pair - uncond_corr_pair
                        pair_percentile = stats.percentileofscore(corr_series_pair, current_corr_pair)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current", f"{current_corr_pair:.3f}")
                        with col2:
                            st.metric("vs Long-run", f"{pair_diff:+.3f}", delta_color="inverse")
                        with col3:
                            st.metric("Percentile", f"{pair_percentile:.0f}%")
                        
                        st.markdown("---")
                        
                        # ================================================================
                        # ELLIPSOID VISUALIZATION
                        # ================================================================
                        st.markdown("## 🔮 Visual: Correlation Ellipsoid")
                        
                        with st.expander("💡 How to read this chart", expanded=True):
                            st.markdown("""
                            This chart shows the **joint distribution** of the two selected assets:
                            
                            - **Blue ellipse**: What the distribution looks like "on average" (long-run)
                            - **Red ellipse**: What the distribution looks like "right now" (conditional)
                            - **Gray dots**: Historical observations (GARCH-filtered)
                            
                            **Interpretation:**
                            - If the **red ellipse is more elongated (stretched)** than the blue → current correlation is **HIGHER** than normal
                            - If the **red ellipse is more circular** than the blue → current correlation is **LOWER** than normal
                            - The **direction of the stretch** shows whether correlation is positive (tilted /) or negative (tilted \\)
                            """)
                        
                        # Use same asset pair
                        eps_pair = dcc['eps_tilde'][:, [asset1_idx, asset2_idx]]
                        
                        rho_uncond_pair = rho2_uncond[np.ix_([asset1_idx, asset2_idx], [asset1_idx, asset2_idx])]
                        rho_cond_pair = r2_t[-1][np.ix_([asset1_idx, asset2_idx], [asset1_idx, asset2_idx])]
                        
                        def create_ellipse_trace(sigma2, scale=2.0, color='blue', name='Ellipse'):
                            eigenvalues, eigenvectors = np.linalg.eig(sigma2)
                            order = eigenvalues.argsort()[::-1]
                            eigenvalues = eigenvalues[order]
                            eigenvectors = eigenvectors[:, order]
                            
                            width = 2 * scale * np.sqrt(eigenvalues[0])
                            height = 2 * scale * np.sqrt(eigenvalues[1])
                            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                            
                            theta = np.linspace(0, 2*np.pi, 100)
                            ellipse_x = (width/2) * np.cos(theta)
                            ellipse_y = (height/2) * np.sin(theta)
                            
                            ang_rad = np.radians(angle)
                            x_rot = ellipse_x * np.cos(ang_rad) - ellipse_y * np.sin(ang_rad)
                            y_rot = ellipse_x * np.sin(ang_rad) + ellipse_y * np.cos(ang_rad)
                            
                            return go.Scatter(
                                x=x_rot, y=y_rot,
                                mode='lines',
                                name=name,
                                line=dict(color=color, width=3)
                            )
                        
                        fig_ellipse = go.Figure()
                        
                        fig_ellipse.add_trace(go.Scatter(
                            x=eps_pair[:, 0], y=eps_pair[:, 1],
                            mode='markers',
                            name='Historical observations',
                            marker=dict(size=3, color='rgba(255,255,255,0.3)')
                        ))
                        
                        fig_ellipse.add_trace(create_ellipse_trace(
                            rho_uncond_pair, scale=2.0, color='#6366F1', name='Long-run (unconditional)'
                        ))
                        
                        fig_ellipse.add_trace(create_ellipse_trace(
                            rho_cond_pair, scale=2.0, color='#FF6B6B', name='Current (conditional)'
                        ))
                        
                        max_range = max(np.abs(eps_pair).max() * 1.2, 3)
                        
                        fig_ellipse.update_layout(
                            height=450,
                            xaxis_title=get_display_name(assets[asset1_idx]),
                            yaxis_title=get_display_name(assets[asset2_idx]),
                            xaxis=dict(range=[-max_range, max_range], scaleanchor="y"),
                            yaxis=dict(range=[-max_range, max_range]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                        )
                        fig_ellipse = apply_plotly_theme(fig_ellipse)
                        st.plotly_chart(fig_ellipse, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # ================================================================
                        # FINAL SUMMARY & RECOMMENDATIONS
                        # ================================================================
                        st.markdown("## 💼 Summary: What Should You Do?")
                        
                        # Build recommendation based on findings
                        recommendations = []
                        warnings = []
                        
                        # Check correlation regime
                        if corr_diff > 0.10:
                            warnings.append("Correlations are elevated vs historical average")
                            recommendations.append("Consider reducing portfolio size or adding uncorrelated assets")
                            
                        if corr_diff > 0.15 and persistence > 0.95:
                            warnings.append("High correlations are likely to persist")
                            recommendations.append("Don't expect quick mean-reversion - adjust positions gradually")
                        
                        # Check for specific high-correlation pairs
                        high_corr_pairs = pairs_df[pairs_df['Current ρ'] > 0.8]
                        if len(high_corr_pairs) > 0:
                            pair_names = ", ".join(high_corr_pairs['Pair'].tolist()[:3])
                            warnings.append(f"Very high correlation between: {pair_names}")
                            recommendations.append("These pairs provide limited diversification - consider if you need both")
                        
                        # Check for correlation increases
                        big_increases = pairs_df[pairs_df['Difference'] > 0.15]
                        if len(big_increases) > 0:
                            recommendations.append("Review hedge ratios for pairs with large correlation increases")
                        
                        # Display
                        if warnings:
                            st.warning("**⚠️ Attention:**\n" + "\n".join([f"- {w}" for w in warnings]))
                        
                        if recommendations:
                            st.markdown("**📋 Recommended Actions:**")
                            for i, rec in enumerate(recommendations, 1):
                                st.markdown(f"{i}. {rec}")
                        else:
                            st.success("""
                            **✅ No immediate concerns.**
                            
                            Correlations are behaving normally. Your portfolio's diversification 
                            is working as expected based on historical patterns.
                            """)
                        
                        # Risk caveat
                        st.markdown("---")
                        st.caption("""
                        **Important:** This analysis estimates *current* correlations based on recent market behavior. 
                        It does NOT predict future returns. Correlations can change rapidly during market stress.
                        Use these insights alongside other risk management tools, not as the sole basis for decisions.
                        
                        📖 *Methodology: DCC-GARCH (Engle, 2002) with Flexible Probabilities (Meucci, 2010)*
                        """)  

            # ================================================================
            # SECTION 3: MEAN-REVERSION & DYNAMICS
            # ================================================================
            elif analysis_mode == "🔄 Mean-Reversion & Dynamics":
                
                st.markdown("""
                ### Understanding Market Dynamics
                
                This section answers fundamental questions for active portfolio management:
                
                - **Do my assets mean-revert or follow trends?**
                - **How fast do they revert?** (Can I trade it?)
                - **Are there stable relationships between assets?** (Pairs trading opportunities)
                - **Where will prices be in N days?** (Forecasting)
                
                We use **Vector Autoregression (VAR)** and **Ornstein-Uhlenbeck** models, 
                which are the workhorses of quantitative finance for modeling mean-reversion.
                """)
                
                with st.expander("📚 The Theory Behind This Analysis", expanded=False):
                    st.markdown(r"""
                    #### Ornstein-Uhlenbeck Process (Single Asset)
                    
                    The **Ornstein-Uhlenbeck (O-U) process** models mean-reverting behavior:
                    
                    $$dX_t = \theta(\mu - X_t)dt + \sigma dW_t$$
                    
                    Where:
                    - $\theta$ = **speed of mean-reversion** (higher = faster reversion)
                    - $\mu$ = **long-run mean** (equilibrium level)
                    - $\sigma$ = **volatility** of the process
                    
                    **Key metric - Half-life:**
                    $$\tau_{1/2} = \frac{\ln(2)}{\theta}$$
                    
                    This tells you how many days it takes for a deviation to halve.
                    
                    ---
                    
                    #### VAR(1) Model (Multiple Assets)
                    
                    The **Vector Autoregression** extends this to multiple assets:
                    
                    $$X_t = \mu + B(X_{t-1} - \mu) + \varepsilon_t$$
                    
                    Where:
                    - $B$ = **transition matrix** (captures cross-asset dynamics)
                    - $\mu$ = **long-run mean vector**
                    - $\varepsilon_t$ = **innovations** with covariance $\Sigma$
                    
                    **Stability condition:** All eigenvalues of $B$ must have $|\lambda| < 1$
                    
                    **Connection to O-U:** For a single asset, $B = e^{-\theta}$, so $\theta = -\ln(B)$
                    
                    ---
                    
                    #### Cointegration
                    
                    Two assets are **cointegrated** if their linear combination is stationary 
                    (mean-reverting) even if individually they are non-stationary.
                    
                    **Example:** SPY and IVV (both S&P 500 ETFs) are cointegrated. 
                    Their spread mean-reverts even though each price trends upward.
                    
                    **Trading implication:** If you find cointegrated pairs, you can trade 
                    the spread with a mean-reversion strategy.
                    
                    📖 *References:*
                    - *Meucci, A. (2005). "Risk and Asset Allocation." Springer.*
                    - *Hamilton, J.D. (1994). "Time Series Analysis." Princeton University Press.*
                    - *Engle, R. & Granger, C. (1987). "Co-integration and Error Correction."*
                    """)
                
                st.markdown("---")
                
                # ================================================================
                # ASSET SELECTION
                # ================================================================
                st.markdown("#### 🎯 Select Assets for Analysis")
                
                available_assets = [(t, get_display_name(t)) for t in symbols]
                
                selected_mr_assets = st.multiselect(
                    "Choose assets to analyze",
                    options=[t[0] for t in available_assets],
                    default=symbols[:min(3, len(symbols))],  # Default: first 3 assets
                    format_func=lambda x: get_display_name(x),
                    key="mr_assets",
                    help="Select 1 asset for pure O-U analysis, 2+ for VAR and cointegration analysis"
                )
                
                # Guidance based on selection
                if len(selected_mr_assets) == 0:
                    st.info("👆 Select at least 1 asset to begin the analysis.")
                    
                elif len(selected_mr_assets) == 1:
                    st.info("""
                    **Single Asset Mode:** You'll get Ornstein-Uhlenbeck analysis with:
                    - Mean-reversion speed and half-life
                    - Current Z-score (deviation from equilibrium)
                    - Trading signals
                    - Price forecasts
                    """)
                    
                elif len(selected_mr_assets) == 2:
                    st.info("""
                    **Pairs Mode:** You'll get:
                    - Individual O-U analysis for each asset
                    - VAR(1) joint dynamics
                    - **Cointegration test** (is the spread tradable?)
                    - Spread analysis with trading signals
                    """)
                    
                else:  # 3+ assets
                    st.info(f"""
                    **Multi-Asset Mode ({len(selected_mr_assets)} assets):** You'll get:
                    - VAR(1) system dynamics
                    - Stability analysis (eigenvalues)
                    - **Cointegration detection** (which combinations mean-revert?)
                    - Impulse response analysis
                    """)
                
                if len(selected_mr_assets) >= 1:
                    
                    # Configuration
                    with st.expander("⚙️ Model Settings", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            use_log_prices = st.checkbox(
                                "Use log-prices",
                                value=True,
                                help="Recommended for most assets. Log-prices have better statistical properties."
                            )
                        
                        with col2:
                            forecast_horizon = st.slider(
                                "Forecast horizon (days)",
                                min_value=5, max_value=60, value=20, step=5,
                                help="How far ahead to forecast"
                            )
                        
                        with col3:
                            confidence_level = st.selectbox(
                                "Confidence level",
                                options=[0.90, 0.95, 0.99],
                                index=1,
                                format_func=lambda x: f"{x*100:.0f}%",
                                help="For forecast intervals and trading signals"
                            )
                    
                    if st.button("🚀 Run Analysis", use_container_width=True, key="run_mr"):
                        
                        with st.spinner("Fitting mean-reversion models..."):
                            
                            try:
                                # Prepare data
                                n_assets = len(selected_mr_assets)
                                prices_df = analyzer.data[selected_mr_assets].dropna()
                                
                                if use_log_prices:
                                    x = np.log(prices_df.values)
                                else:
                                    x = prices_df.values
                                
                                dates = prices_df.index
                                t_bar = len(dates)
                                
                                # Uniform probabilities (can be extended to flexible probabilities)
                                p = np.ones(t_bar) / t_bar
                                
                                # ============================================
                                # FIT VAR(1) MODEL
                                # ============================================                                
                                b_hat, mu_eps_hat, sigma2_hat = fit_var1(x, p, nu=1e9)
                                
                                # Ensure proper shapes
                                if n_assets == 1:
                                    b_hat = np.array([[b_hat]])
                                    sigma2_hat = np.array([[sigma2_hat]])
                                    mu_eps_hat = np.array([mu_eps_hat])
                                else:
                                    b_hat = np.atleast_2d(b_hat)
                                    sigma2_hat = np.atleast_2d(sigma2_hat)
                                    mu_eps_hat = np.atleast_1d(mu_eps_hat)
                                
                                # Compute long-run mean
                                # From X_t = mu_eps + B @ X_{t-1}, stationary mean is:
                                # mu = (I - B)^{-1} @ mu_eps
                                try:
                                    mu_longrun = np.linalg.solve(np.eye(n_assets) - b_hat, mu_eps_hat)
                                except np.linalg.LinAlgError:
                                    mu_longrun = np.mean(x, axis=0)
                                
                                # Eigenvalue analysis for stability
                                eigenvalues = np.linalg.eigvals(b_hat)
                                max_eigenvalue = np.max(np.abs(eigenvalues))
                                is_stable = max_eigenvalue < 1
                                
                                # Compute half-lives from eigenvalues
                                half_lives = []
                                for ev in eigenvalues:
                                    if np.abs(ev) < 1 and np.abs(ev) > 0.01:
                                        hl = np.log(2) / (-np.log(np.abs(ev)))
                                        half_lives.append(hl)
                                    else:
                                        half_lives.append(np.inf)
                                
                                # Cointegration analysis (for 2+ assets)
                                c_hat, beta_hat = None, None
                                if n_assets >= 2:
                                    c_hat, beta_hat = cointegration_fp(x, p, b_threshold=0.99)
                                
                                # Store results
                                st.session_state.mr_results = {
                                    'b_hat': b_hat,
                                    'mu_eps_hat': mu_eps_hat,
                                    'sigma2_hat': sigma2_hat,
                                    'mu_longrun': mu_longrun,
                                    'eigenvalues': eigenvalues,
                                    'half_lives': half_lives,
                                    'is_stable': is_stable,
                                    'c_hat': c_hat,
                                    'beta_hat': beta_hat,
                                    'x': x,
                                    'dates': dates,
                                    'assets': selected_mr_assets,
                                    'prices_df': prices_df,
                                    'use_log': use_log_prices,
                                    'forecast_horizon': forecast_horizon,
                                    'confidence_level': confidence_level
                                }
                                
                                st.success("✅ Analysis complete!")
                                
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                                st.session_state.mr_results = None

                    # ================================================================
                    # DISPLAY RESULTS
                    # ================================================================
                    if 'mr_results' in st.session_state and st.session_state.mr_results is not None:
                        
                        mr = st.session_state.mr_results
                        n_assets = len(mr['assets'])
                        x = mr['x']
                        dates = mr['dates']
                        b_hat = mr['b_hat']
                        mu_longrun = mr['mu_longrun']
                        sigma2_hat = mr['sigma2_hat']
                        eigenvalues = mr['eigenvalues']
                        half_lives = mr['half_lives']
                        prices_df = mr['prices_df']
                        use_log = mr['use_log']
                        forecast_horizon = mr['forecast_horizon']
                        confidence_level = mr['confidence_level']
                        
                        st.markdown("---")
                        
                        # ============================================================
                        # QUICK SUMMARY - KPI CARDS
                        # ============================================================
                        st.markdown("## 🎯 Quick Summary")
                        
                        # Current values and deviations
                        x_current = x[-1, :]
                        deviations = x_current - mu_longrun
                        
                        # Compute conditional standard deviations
                        # For stationary VAR(1), unconditional variance is solution to:
                        # Sigma = B @ Sigma @ B' + Sigma_eps
                        # We use a simpler approximation: sigma_i = sqrt(sigma2_hat[i,i] / (1 - b_ii^2))
                        try:
                            sigma_unconditional = np.sqrt(np.diag(sigma2_hat) / (1 - np.diag(b_hat)**2))
                        except:
                            sigma_unconditional = np.sqrt(np.diag(sigma2_hat))
                        
                        z_scores = deviations / sigma_unconditional
                        
                        # Primary half-life (fastest mean-reversion)
                        valid_half_lives = [hl for hl in half_lives if hl != np.inf and hl > 0]
                        primary_half_life = min(valid_half_lives) if valid_half_lives else np.inf
                        
                        # KPI Cards
                        if n_assets == 1:
                            # Single asset layout
                            col1, col2, col3, col4 = st.columns(4)
                            
                            asset_name = get_display_name(mr['assets'][0])
                            
                            with col1:
                                st.metric(
                                    "Mean-Reversion Speed",
                                    f"{-np.log(np.abs(b_hat[0,0])):.4f}" if np.abs(b_hat[0,0]) < 1 else "N/A",
                                    help="θ parameter: higher = faster reversion"
                                )
                            
                            with col2:
                                if primary_half_life != np.inf:
                                    st.metric(
                                        "Half-Life",
                                        f"{primary_half_life:.1f} days",
                                        help="Time for deviation to halve"
                                    )
                                else:
                                    st.metric("Half-Life", "∞ (Random Walk)")
                            
                            with col3:
                                z = z_scores[0]
                                st.metric(
                                    "Current Z-Score",
                                    f"{z:+.2f}σ",
                                    delta="Overbought" if z > 1.5 else "Oversold" if z < -1.5 else "Neutral",
                                    delta_color="inverse" if abs(z) > 1.5 else "off"
                                )
                            
                            with col4:
                                if use_log:
                                    equilibrium_price = np.exp(mu_longrun[0])
                                    current_price = np.exp(x_current[0])
                                else:
                                    equilibrium_price = mu_longrun[0]
                                    current_price = x_current[0]
                                
                                pct_from_eq = (current_price / equilibrium_price - 1) * 100
                                st.metric(
                                    "Distance from Equilibrium",
                                    f"{pct_from_eq:+.2f}%",
                                    help=f"Equilibrium price: ${equilibrium_price:.2f}"
                                )
                        
                        else:
                            # Multi-asset layout
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "System Stability",
                                    "✅ Stable" if mr['is_stable'] else "⚠️ Unstable",
                                    f"λ_max = {np.max(np.abs(eigenvalues)):.3f}"
                                )
                            
                            with col2:
                                if primary_half_life != np.inf:
                                    st.metric(
                                        "Fastest Half-Life",
                                        f"{primary_half_life:.1f} days",
                                        help="Fastest mean-reverting direction"
                                    )
                                else:
                                    st.metric("Fastest Half-Life", "∞")
                            
                            with col3:
                                n_coint = mr['c_hat'].shape[1] if mr['c_hat'] is not None and len(mr['c_hat'].shape) > 1 else 0
                                st.metric(
                                    "Cointegration Relations",
                                    f"{n_coint} found",
                                    "Tradable spreads!" if n_coint > 0 else "No stable spreads"
                                )
                            
                            with col4:
                                max_z = np.max(np.abs(z_scores))
                                max_z_idx = np.argmax(np.abs(z_scores))
                                max_z_asset = get_display_name(mr['assets'][max_z_idx])
                                st.metric(
                                    "Largest Deviation",
                                    f"{z_scores[max_z_idx]:+.2f}σ",
                                    f"{max_z_asset}"
                                )
                        
                        st.markdown("---")
                        
                        # ============================================================
                        # SECTION 1: ORNSTEIN-UHLENBECK ANALYSIS (PER ASSET)
                        # ============================================================
                        st.markdown("## 📊 Ornstein-Uhlenbeck Analysis")
                        
                        st.markdown("""
                        For each asset, we estimate the O-U parameters and assess whether mean-reversion 
                        is **strong enough to be tradable**.
                        
                        **Rule of thumb for tradability:**
                        - Half-life < 20 days → Excellent for mean-reversion trading
                        - Half-life 20-60 days → Moderate opportunity, slower trades
                        - Half-life > 60 days → Too slow, essentially a random walk for trading purposes
                        """)
                        
                        # Create tabs for each asset if multiple
                        if n_assets > 1:
                            asset_tabs = st.tabs([f"📈 {get_display_name(a)}" for a in mr['assets']])
                        else:
                            asset_tabs = [st.container()]
                        
                        for i, (asset_tab, asset) in enumerate(zip(asset_tabs, mr['assets'])):
                            with asset_tab:
                                
                                asset_name = get_display_name(asset)
                                x_i = x[:, i]
                                
                                # Extract O-U parameters for this asset
                                b_i = b_hat[i, i] if n_assets > 1 else b_hat[0, 0]
                                mu_i = mu_longrun[i]
                                sigma_eps_i = np.sqrt(sigma2_hat[i, i]) if n_assets > 1 else np.sqrt(sigma2_hat[0, 0])
                                
                                # O-U parameters
                                if np.abs(b_i) < 1:
                                    theta_i = -np.log(np.abs(b_i))  # Mean-reversion speed
                                    half_life_i = np.log(2) / theta_i
                                    # Unconditional volatility
                                    sigma_unconditional_i = sigma_eps_i / np.sqrt(1 - b_i**2)
                                else:
                                    theta_i = 0
                                    half_life_i = np.inf
                                    sigma_unconditional_i = sigma_eps_i * np.sqrt(len(x_i))
                                
                                # Current state
                                x_current_i = x_i[-1]
                                z_score_i = (x_current_i - mu_i) / sigma_unconditional_i if sigma_unconditional_i > 0 else 0
                                
                                # Layout
                                col1, col2 = st.columns([1.5, 1])
                                
                                with col1:
                                    # Price chart with equilibrium level
                                    st.markdown(f"##### {asset_name}: Price vs Equilibrium")
                                    
                                    fig = go.Figure()
                                    
                                    # Price series
                                    if use_log:
                                        price_series = np.exp(x_i)
                                        eq_price = np.exp(mu_i)
                                        upper_band = np.exp(mu_i + 2 * sigma_unconditional_i)
                                        lower_band = np.exp(mu_i - 2 * sigma_unconditional_i)
                                    else:
                                        price_series = x_i
                                        eq_price = mu_i
                                        upper_band = mu_i + 2 * sigma_unconditional_i
                                        lower_band = mu_i - 2 * sigma_unconditional_i
                                    
                                    # ±2σ bands
                                    fig.add_trace(go.Scatter(
                                        x=dates, y=np.full(len(dates), upper_band),
                                        mode='lines', name='+2σ',
                                        line=dict(color='rgba(255,107,107,0.5)', width=1, dash='dot')
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=dates, y=np.full(len(dates), lower_band),
                                        mode='lines', name='-2σ',
                                        line=dict(color='rgba(78,205,196,0.5)', width=1, dash='dot'),
                                        fill='tonexty', fillcolor='rgba(99,102,241,0.1)'
                                    ))
                                    
                                    # Equilibrium line
                                    fig.add_trace(go.Scatter(
                                        x=dates, y=np.full(len(dates), eq_price),
                                        mode='lines', name=f'Equilibrium (μ = ${eq_price:.2f})',
                                        line=dict(color='#FFE66D', width=2, dash='dash')
                                    ))
                                    
                                    # Price
                                    fig.add_trace(go.Scatter(
                                        x=dates, y=price_series,
                                        mode='lines', name='Price',
                                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2)
                                    ))
                                    
                                    # Mark current price
                                    fig.add_trace(go.Scatter(
                                        x=[dates[-1]], y=[price_series[-1]],
                                        mode='markers', name='Current',
                                        marker=dict(size=12, color='white', line=dict(width=2, color=CHART_COLORS[i % len(CHART_COLORS)]))
                                    ))
                                    
                                    fig.update_layout(
                                        height=350,
                                        xaxis_title="Date",
                                        yaxis_title="Price ($)",
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                        hovermode='x unified'
                                    )
                                    fig = apply_plotly_theme(fig)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # O-U Parameters table
                                    st.markdown("##### O-U Parameters")
                                    
                                    params_data = {
                                        'Parameter': ['θ (speed)', 'μ (equilibrium)', 'σ (volatility)', 'Half-life', 'AR(1) coef β'],
                                        'Value': [
                                            f"{theta_i:.4f}",
                                            f"{mu_i:.4f}" + (f" (${np.exp(mu_i):.2f})" if use_log else ""),
                                            f"{sigma_eps_i:.4f}",
                                            f"{half_life_i:.1f} days" if half_life_i != np.inf else "∞",
                                            f"{b_i:.4f}"
                                        ]
                                    }
                                    st.markdown(create_styled_table(pd.DataFrame(params_data)), unsafe_allow_html=True)
                                    
                                    # Tradability assessment
                                    st.markdown("##### 🎯 Tradability Assessment")
                                    
                                    if half_life_i < 20:
                                        st.success(f"✅ **Excellent** for mean-reversion trading")
                                        st.markdown(f"Half-life of {half_life_i:.1f} days allows for frequent trading opportunities.")
                                    elif half_life_i < 60:
                                        st.info(f"ℹ️ **Moderate** mean-reversion")
                                        st.markdown(f"Half-life of {half_life_i:.1f} days → trades take weeks to play out.")
                                    elif half_life_i != np.inf:
                                        st.warning(f"⚠️ **Slow** mean-reversion")
                                        st.markdown(f"Half-life of {half_life_i:.1f} days is too slow for active trading.")
                                    else:
                                        st.error("❌ **No mean-reversion detected**")
                                        st.markdown("This asset behaves like a random walk. Mean-reversion strategies won't work.")
                                
                                # ============================================
                                # TRADING SIGNALS
                                # ============================================
                                st.markdown("##### 📡 Trading Signals")
                                
                                sig_col1, sig_col2, sig_col3 = st.columns(3)
                                
                                with sig_col1:
                                    # Z-Score Signal
                                    st.markdown("**Z-Score Signal**")
                                    
                                    z_display = z_score_i
                                    
                                    if z_display > 2:
                                        signal_color = "#FF6B6B"
                                        signal_text = "🔴 STRONG SELL"
                                        signal_desc = f"Price is {z_display:.2f}σ above equilibrium"
                                    elif z_display > 1:
                                        signal_color = "#FF9F43"
                                        signal_text = "🟠 SELL"
                                        signal_desc = f"Price is {z_display:.2f}σ above equilibrium"
                                    elif z_display < -2:
                                        signal_color = "#4ECDC4"
                                        signal_text = "🟢 STRONG BUY"
                                        signal_desc = f"Price is {abs(z_display):.2f}σ below equilibrium"
                                    elif z_display < -1:
                                        signal_color = "#95E1D3"
                                        signal_text = "🟢 BUY"
                                        signal_desc = f"Price is {abs(z_display):.2f}σ below equilibrium"
                                    else:
                                        signal_color = "#6366F1"
                                        signal_text = "⚪ NEUTRAL"
                                        signal_desc = f"Price is within ±1σ of equilibrium"
                                    
                                    st.markdown(f"""
                                    <div style='background: {signal_color}22; border-left: 4px solid {signal_color}; 
                                                padding: 1rem; border-radius: 8px;'>
                                        <strong style='color: {signal_color}; font-size: 1.2rem;'>{signal_text}</strong><br>
                                        <span style='color: #94a3b8;'>{signal_desc}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with sig_col2:
                                    # Entry/Exit Levels
                                    st.markdown("**Entry/Exit Levels**")
                                    
                                    if use_log:
                                        entry_buy = np.exp(mu_i - 1.5 * sigma_unconditional_i)
                                        entry_sell = np.exp(mu_i + 1.5 * sigma_unconditional_i)
                                        target = np.exp(mu_i)
                                        stop_loss_long = np.exp(mu_i - 3 * sigma_unconditional_i)
                                        stop_loss_short = np.exp(mu_i + 3 * sigma_unconditional_i)
                                    else:
                                        entry_buy = mu_i - 1.5 * sigma_unconditional_i
                                        entry_sell = mu_i + 1.5 * sigma_unconditional_i
                                        target = mu_i
                                        stop_loss_long = mu_i - 3 * sigma_unconditional_i
                                        stop_loss_short = mu_i + 3 * sigma_unconditional_i
                                    
                                    levels_data = {
                                        'Level': ['Buy Entry', 'Sell Entry', 'Target (μ)', 'Stop Loss (Long)', 'Stop Loss (Short)'],
                                        'Price': [f"${entry_buy:.2f}", f"${entry_sell:.2f}", f"${target:.2f}", 
                                                  f"${stop_loss_long:.2f}", f"${stop_loss_short:.2f}"]
                                    }
                                    st.markdown(create_styled_table(pd.DataFrame(levels_data)), unsafe_allow_html=True)
                                
                                with sig_col3:
                                    # Probability of Mean-Reversion
                                    st.markdown("**Reversion Probability**")
                                    
                                    if half_life_i != np.inf and half_life_i > 0:
                                        # Probability of crossing equilibrium within N days
                                        # Simplified: using decay factor
                                        horizons = [5, 10, 20]
                                        
                                        for h in horizons:
                                            # Expected deviation after h days
                                            decay = b_i ** h
                                            expected_dev = decay * (x_current_i - mu_i)
                                            # Probability of being closer to mean
                                            prob_closer = 1 - decay  # Simplified approximation
                                            
                                            if prob_closer > 0.7:
                                                color = "#4ECDC4"
                                            elif prob_closer > 0.4:
                                                color = "#FFE66D"
                                            else:
                                                color = "#FF6B6B"
                                            
                                            st.markdown(f"""
                                            <div style='display: flex; justify-content: space-between; 
                                                        padding: 0.5rem; background: {color}22; 
                                                        border-radius: 4px; margin-bottom: 0.25rem;'>
                                                <span>Closer to μ in {h}d:</span>
                                                <strong style='color: {color};'>{prob_closer*100:.0f}%</strong>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    else:
                                        st.markdown("*No mean-reversion detected*")
                                
                                if n_assets > 1:
                                    st.markdown("---")

                        # ============================================================
                        # SECTION 2: VAR(1) SYSTEM DYNAMICS (Multi-asset only)
                        # ============================================================
                        if n_assets >= 2:
                            
                            st.markdown("## 🔗 VAR(1) System Dynamics")
                            
                            st.markdown("""
                            The **Vector Autoregression** captures how assets influence each other over time.
                            The transition matrix $B$ shows both **own dynamics** (diagonal) and **cross-asset effects** (off-diagonal).
                            """)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Transition Matrix Heatmap
                                st.markdown("##### Transition Matrix B")
                                
                                asset_names = [get_display_name(a) for a in mr['assets']]
                                
                                fig_b = go.Figure(data=go.Heatmap(
                                    z=b_hat,
                                    x=asset_names,
                                    y=asset_names,
                                    colorscale='RdBu_r',
                                    zmid=0,
                                    zmin=-1, zmax=1,
                                    text=np.round(b_hat, 3),
                                    texttemplate='%{text}',
                                    textfont=dict(size=11, color='white'),
                                    hovertemplate='%{y} → %{x}<br>Coefficient: %{z:.4f}<extra></extra>',
                                    colorbar=dict(title='Coefficient')
                                ))
                                
                                fig_b.update_layout(
                                    height=350,
                                    xaxis_title="Today's asset",
                                    yaxis_title="Yesterday's asset",
                                    yaxis=dict(autorange='reversed')
                                )
                                fig_b = apply_plotly_theme(fig_b)
                                st.plotly_chart(fig_b, use_container_width=True)
                                
                                st.markdown("""
                                **How to read this:**
                                - **Diagonal (B_ii):** Own persistence. Values < 1 indicate mean-reversion.
                                - **Off-diagonal (B_ij):** How asset j yesterday affects asset i today.
                                - **Positive off-diagonal:** Assets move together (momentum spillover)
                                - **Negative off-diagonal:** Assets move opposite (mean-reversion spillover)
                                """)
                            
                            with col2:
                                # Eigenvalue Analysis
                                st.markdown("##### Eigenvalue Analysis (Stability)")
                                
                                # Plot eigenvalues in complex plane
                                fig_eig = go.Figure()
                                
                                # Unit circle
                                theta = np.linspace(0, 2*np.pi, 100)
                                fig_eig.add_trace(go.Scatter(
                                    x=np.cos(theta), y=np.sin(theta),
                                    mode='lines', name='Unit Circle',
                                    line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash')
                                ))
                                
                                # Eigenvalues
                                eig_real = np.real(eigenvalues)
                                eig_imag = np.imag(eigenvalues)
                                eig_abs = np.abs(eigenvalues)
                                
                                colors = ['#4ECDC4' if abs(e) < 1 else '#FF6B6B' for e in eigenvalues]
                                
                                fig_eig.add_trace(go.Scatter(
                                    x=eig_real, y=eig_imag,
                                    mode='markers+text',
                                    name='Eigenvalues',
                                    marker=dict(size=15, color=colors, line=dict(width=2, color='white')),
                                    text=[f'λ{i+1}' for i in range(len(eigenvalues))],
                                    textposition='top center',
                                    textfont=dict(color='#E2E8F0', size=10),
                                    hovertemplate='λ = %{x:.3f} + %{y:.3f}i<br>|λ| = ' + 
                                                  '<br>'.join([f'{abs(e):.3f}' for e in eigenvalues]) + '<extra></extra>'
                                ))
                                
                                fig_eig.update_layout(
                                    height=350,
                                    xaxis_title="Real Part",
                                    yaxis_title="Imaginary Part",
                                    xaxis=dict(range=[-1.5, 1.5], scaleanchor="y"),
                                    yaxis=dict(range=[-1.5, 1.5]),
                                    showlegend=False
                                )
                                fig_eig = apply_plotly_theme(fig_eig)
                                st.plotly_chart(fig_eig, use_container_width=True)
                                
                                # Eigenvalue table
                                eig_data = []
                                for idx, (ev, hl) in enumerate(zip(eigenvalues, half_lives)):
                                    eig_data.append({
                                        'Eigenvalue': f"λ{idx+1}",
                                        'Value': f"{np.real(ev):.4f}" + (f" + {np.imag(ev):.4f}i" if np.imag(ev) != 0 else ""),
                                        '|λ|': f"{np.abs(ev):.4f}",
                                        'Half-life': f"{hl:.1f} days" if hl != np.inf else "∞",
                                        'Status': "✅ Stable" if np.abs(ev) < 1 else "⚠️ Unstable"
                                    })
                                
                                st.markdown(create_styled_table(pd.DataFrame(eig_data)), unsafe_allow_html=True)
                            
                            # System stability interpretation
                            st.markdown("##### 🎯 System Interpretation")
                            
                            if mr['is_stable']:
                                st.success(f"""
                                **✅ The system is stable** (all |λ| < 1)
                                
                                This means:
                                - Shocks to any asset will eventually die out
                                - The system has a well-defined long-run equilibrium
                                - Mean-reversion strategies can work on this asset set
                                
                                **Fastest mean-reversion:** {min([hl for hl in half_lives if hl != np.inf]):.1f} days
                                **Slowest mean-reversion:** {max([hl for hl in half_lives if hl != np.inf]):.1f} days
                                """)
                            else:
                                st.error(f"""
                                **⚠️ The system is unstable** (some |λ| ≥ 1)
                                
                                This means:
                                - The assets contain unit roots or explosive dynamics
                                - There is no long-run equilibrium to revert to
                                - Individual mean-reversion strategies are risky
                                
                                **However:** Check the cointegration analysis below - there may still be 
                                stable *combinations* of these assets (spreads) that mean-revert.
                                """)
                            
                            st.markdown("---")
                            
                            # ============================================================
                            # SECTION 3: COINTEGRATION ANALYSIS
                            # ============================================================
                            st.markdown("## 🔄 Cointegration Analysis")
                            
                            st.markdown("""
                            **Cointegration** occurs when non-stationary assets have a linear combination that IS stationary.
                            
                            **Why does this matter?**
                            - Even if SPY and IVV both trend upward (non-stationary), their *spread* might mean-revert
                            - Cointegrated pairs are the foundation of **pairs trading** and **statistical arbitrage**
                            - The cointegration vector tells you the **hedge ratio**
                            """)
                            
                            c_hat = mr['c_hat']
                            beta_hat = mr['beta_hat']
                            
                            if c_hat is not None and c_hat.size > 0:
                                
                                n_coint = c_hat.shape[1] if len(c_hat.shape) > 1 else 1
                                
                                if n_coint > 0:
                                    st.success(f"**✅ Found {n_coint} cointegration relationship(s)!** These represent tradable spreads.")
                                    
                                    # Ensure proper shape
                                    if len(c_hat.shape) == 1:
                                        c_hat = c_hat.reshape(-1, 1)
                                    if np.isscalar(beta_hat):
                                        beta_hat = np.array([beta_hat])
                                    
                                    # Create tabs for each cointegration vector
                                    if n_coint > 1:
                                        coint_tabs = st.tabs([f"Spread {i+1}" for i in range(n_coint)])
                                    else:
                                        coint_tabs = [st.container()]
                                    
                                    for coint_idx, coint_tab in enumerate(coint_tabs):
                                        with coint_tab:
                                            
                                            # Get cointegration vector
                                            c_vec = c_hat[:, coint_idx]
                                            beta_i = beta_hat[coint_idx] if len(beta_hat) > coint_idx else beta_hat[0]
                                            
                                            # Half-life of the spread
                                            spread_half_life = np.log(2) / (-np.log(np.abs(beta_i))) if np.abs(beta_i) < 1 else np.inf
                                            
                                            col1, col2 = st.columns([1.5, 1])
                                            
                                            with col1:
                                                # Compute the spread
                                                spread = x @ c_vec
                                                spread_mean = np.mean(spread)
                                                spread_std = np.std(spread)
                                                spread_zscore = (spread - spread_mean) / spread_std
                                                current_spread_z = spread_zscore[-1]
                                                
                                                # Spread chart
                                                st.markdown(f"##### Spread Time Series (Z-Score)")
                                                
                                                fig_spread = go.Figure()
                                                
                                                # Z-score bands
                                                fig_spread.add_hline(y=2, line_dash="dot", line_color="#FF6B6B", 
                                                                     annotation_text="+2σ (Sell)")
                                                fig_spread.add_hline(y=-2, line_dash="dot", line_color="#4ECDC4",
                                                                     annotation_text="-2σ (Buy)")
                                                fig_spread.add_hline(y=0, line_dash="dash", line_color="#FFE66D")
                                                
                                                # Spread
                                                fig_spread.add_trace(go.Scatter(
                                                    x=dates, y=spread_zscore,
                                                    mode='lines', name='Spread Z-Score',
                                                    line=dict(color=CHART_COLORS[coint_idx % len(CHART_COLORS)], width=2),
                                                    fill='tozeroy',
                                                    fillcolor=f'rgba({",".join(str(int(c*255)) for c in plt.cm.tab10(coint_idx)[:3])}, 0.2)' 
                                                              if 'plt' in dir() else 'rgba(99,102,241,0.2)'
                                                ))
                                                
                                                # Current point
                                                fig_spread.add_trace(go.Scatter(
                                                    x=[dates[-1]], y=[current_spread_z],
                                                    mode='markers', name='Current',
                                                    marker=dict(size=12, color='white', 
                                                               line=dict(width=2, color=CHART_COLORS[coint_idx % len(CHART_COLORS)]))
                                                ))
                                                
                                                fig_spread.update_layout(
                                                    height=300,
                                                    xaxis_title="Date",
                                                    yaxis_title="Z-Score",
                                                    yaxis=dict(range=[-4, 4]),
                                                    showlegend=False,
                                                    hovermode='x unified'
                                                )
                                                fig_spread = apply_plotly_theme(fig_spread)
                                                st.plotly_chart(fig_spread, use_container_width=True)
                                            
                                            with col2:
                                                # Cointegration vector (hedge ratios)
                                                st.markdown("##### Hedge Ratios (Cointegration Vector)")
                                                
                                                # Normalize so first non-zero is 1
                                                c_normalized = c_vec / c_vec[np.argmax(np.abs(c_vec))]
                                                
                                                hedge_data = {
                                                    'Asset': [get_display_name(a) for a in mr['assets']],
                                                    'Raw Weight': [f"{c:.4f}" for c in c_vec],
                                                    'Normalized': [f"{c:.4f}" for c in c_normalized],
                                                    'Position': ['LONG' if c > 0.01 else 'SHORT' if c < -0.01 else '-' for c in c_normalized]
                                                }
                                                st.markdown(create_styled_table(pd.DataFrame(hedge_data)), unsafe_allow_html=True)
                                                
                                                # Spread statistics
                                                st.markdown("##### Spread Statistics")
                                                
                                                spread_stats = {
                                                    'Metric': ['AR(1) β', 'Half-life', 'Current Z', 'Mean', 'Std Dev'],
                                                    'Value': [
                                                        f"{beta_i:.4f}",
                                                        f"{spread_half_life:.1f} days" if spread_half_life != np.inf else "∞",
                                                        f"{current_spread_z:+.2f}σ",
                                                        f"{spread_mean:.4f}",
                                                        f"{spread_std:.4f}"
                                                    ]
                                                }
                                                st.markdown(create_styled_table(pd.DataFrame(spread_stats)), unsafe_allow_html=True)
                                            
                                            # Trading signal for spread
                                            st.markdown("##### 📡 Spread Trading Signal")
                                            
                                            sig_col1, sig_col2, sig_col3 = st.columns(3)
                                            
                                            with sig_col1:
                                                if current_spread_z > 2:
                                                    st.error(f"""
                                                    **🔴 SELL SPREAD** (Z = {current_spread_z:+.2f})
                                                    
                                                    Action: Short the spread
                                                    """)
                                                    # Show positions
                                                    for a, c in zip(mr['assets'], c_normalized):
                                                        pos = "SHORT" if c > 0 else "LONG"
                                                        st.markdown(f"• **{pos}** {abs(c):.2f} units of {get_display_name(a)}")
                                                        
                                                elif current_spread_z < -2:
                                                    st.success(f"""
                                                    **🟢 BUY SPREAD** (Z = {current_spread_z:+.2f})
                                                    
                                                    Action: Long the spread
                                                    """)
                                                    for a, c in zip(mr['assets'], c_normalized):
                                                        pos = "LONG" if c > 0 else "SHORT"
                                                        st.markdown(f"• **{pos}** {abs(c):.2f} units of {get_display_name(a)}")
                                                        
                                                elif abs(current_spread_z) > 1:
                                                    st.warning(f"""
                                                    **🟡 WATCH** (Z = {current_spread_z:+.2f})
                                                    
                                                    Approaching tradable levels. Wait for Z > 2 or Z < -2.
                                                    """)
                                                else:
                                                    st.info(f"""
                                                    **⚪ NEUTRAL** (Z = {current_spread_z:+.2f})
                                                    
                                                    Spread is near equilibrium. No trade signal.
                                                    """)
                                            
                                            with sig_col2:
                                                st.markdown("**Trade Parameters**")
                                                
                                                trade_params = {
                                                    'Parameter': ['Entry (Long)', 'Entry (Short)', 'Target', 'Stop Loss'],
                                                    'Z-Score': ['-2.0σ', '+2.0σ', '0.0σ', '±3.0σ']
                                                }
                                                st.markdown(create_styled_table(pd.DataFrame(trade_params)), unsafe_allow_html=True)
                                            
                                            with sig_col3:
                                                st.markdown("**Expected Holding Period**")
                                                
                                                if spread_half_life != np.inf:
                                                    # Time to go from current Z to 0
                                                    if abs(current_spread_z) > 0.5:
                                                        expected_days = spread_half_life * np.log(abs(current_spread_z)) / np.log(2)
                                                        st.metric("Est. Time to Target", f"{expected_days:.0f} days")
                                                    else:
                                                        st.metric("Est. Time to Target", "At target")
                                                    
                                                    st.metric("Half-Life", f"{spread_half_life:.1f} days")
                                                else:
                                                    st.warning("Spread half-life is infinite")
                                            
                                            if n_coint > 1:
                                                st.markdown("---")
                                
                                else:
                                    st.warning("""
                                    **⚠️ No cointegration relationships found.**
                                    
                                    This means:
                                    - The selected assets don't have stable linear combinations
                                    - Pairs trading on these assets is risky
                                    - Consider selecting different assets or a longer time period
                                    """)
                            
                            else:
                                st.warning("""
                                **⚠️ No cointegration relationships found.**
                                
                                The selected assets appear to move independently without stable long-run relationships.
                                """)
                            
                            st.markdown("---")
                            
                            # ============================================================
                            # SECTION 4: IMPULSE RESPONSE FUNCTIONS
                            # ============================================================
                            st.markdown("## 📊 Impulse Response Analysis")
                            
                            st.markdown("""
                            **Impulse Response Functions (IRF)** show how a shock to one asset propagates through the system over time.
                            
                            This answers: *"If Asset A jumps 1% today, what happens to Assets B, C, D over the next N days?"*
                            """)
                            
                            # Select shock asset
                            shock_asset_idx = st.selectbox(
                                "Select asset to shock",
                                options=list(range(n_assets)),
                                format_func=lambda x: get_display_name(mr['assets'][x]),
                                key="irf_shock_asset"
                            )
                            
                            irf_horizon = st.slider("IRF Horizon (days)", 5, 60, 20, key="irf_horizon")
                            
                            # Compute IRF
                            irf = np.zeros((irf_horizon + 1, n_assets))
                            
                            # Initial shock: 1 unit to selected asset
                            shock = np.zeros(n_assets)
                            shock[shock_asset_idx] = 1.0
                            
                            irf[0, :] = shock
                            
                            # Propagate
                            for t in range(1, irf_horizon + 1):
                                irf[t, :] = b_hat @ irf[t-1, :]
                            
                            # Plot IRF
                            fig_irf = go.Figure()
                            
                            for i, asset in enumerate(mr['assets']):
                                fig_irf.add_trace(go.Scatter(
                                    x=list(range(irf_horizon + 1)),
                                    y=irf[:, i],
                                    mode='lines+markers',
                                    name=get_display_name(asset),
                                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                                    marker=dict(size=6)
                                ))
                            
                            fig_irf.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                            
                            fig_irf.update_layout(
                                height=400,
                                xaxis_title="Days After Shock",
                                yaxis_title="Response",
                                title=f"Response to 1-Unit Shock in {get_display_name(mr['assets'][shock_asset_idx])}",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                hovermode='x unified'
                            )
                            fig_irf = apply_plotly_theme(fig_irf)
                            st.plotly_chart(fig_irf, use_container_width=True)
                            
                            # IRF interpretation
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("##### Cumulative Impact After 20 Days")
                                
                                cumulative_irf = np.sum(irf[:min(21, irf_horizon+1), :], axis=0)
                                
                                impact_data = {
                                    'Asset': [get_display_name(a) for a in mr['assets']],
                                    'Cumulative Response': [f"{c:.4f}" for c in cumulative_irf],
                                    'Interpretation': [
                                        '🔵 Same direction' if c > 0.1 else '🔴 Opposite direction' if c < -0.1 else '⚪ Minimal impact'
                                        for c in cumulative_irf
                                    ]
                                }
                                st.markdown(create_styled_table(pd.DataFrame(impact_data)), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("##### Shock Persistence")
                                
                                # How much of initial shock remains after various horizons
                                shock_remaining = []
                                for h in [5, 10, 20]:
                                    if h <= irf_horizon:
                                        remaining = np.linalg.norm(irf[h, :]) / np.linalg.norm(irf[0, :]) * 100
                                        shock_remaining.append({'Horizon': f'{h} days', 'Shock Remaining': f'{remaining:.1f}%'})
                                
                                if shock_remaining:
                                    st.markdown(create_styled_table(pd.DataFrame(shock_remaining)), unsafe_allow_html=True)
                                
                                if mr['is_stable']:
                                    st.info("✅ Shocks decay over time (stable system)")
                                else:
                                    st.warning("⚠️ Shocks may persist or grow (unstable system)")

                        # ============================================================
                        # SECTION 5: FORECASTING
                        # ============================================================
                        st.markdown("## 🔮 Forecasting")
                        
                        st.markdown(f"""
                        Using the estimated VAR(1) model, we generate **point forecasts** and **confidence intervals** 
                        for the next **{forecast_horizon} days**.
                        
                        The forecast formula is:
                        
                        $$\\hat{{X}}_{{t+h}} = \\mu + B^h (X_t - \\mu)$$
                        
                        Where the forecast error grows with horizon due to accumulated uncertainty.
                        """)
                        
                        # Select asset to forecast
                        forecast_asset_idx = st.selectbox(
                            "Select asset to forecast",
                            options=list(range(n_assets)),
                            format_func=lambda x: get_display_name(mr['assets'][x]),
                            key="forecast_asset"
                        )
                        
                        # Compute forecasts
                        x_current = x[-1, :]
                        deviation = x_current - mu_longrun
                        
                        forecasts = np.zeros((forecast_horizon + 1, n_assets))
                        forecast_std = np.zeros((forecast_horizon + 1, n_assets))
                        
                        forecasts[0, :] = x_current
                        forecast_std[0, :] = 0
                        
                        # Compute forecast variance iteratively
                        # Var(X_{t+h}) = B^h Var(X_t) (B^h)' + sum_{j=0}^{h-1} B^j Sigma (B^j)'
                        # For simplicity, we use the unconditional variance approximation
                        
                        B_power = np.eye(n_assets)
                        cumulative_var = np.zeros((n_assets, n_assets))
                        
                        for h in range(1, forecast_horizon + 1):
                            B_power = B_power @ b_hat
                            forecasts[h, :] = mu_longrun + B_power @ deviation
                            
                            # Accumulate forecast variance
                            cumulative_var = b_hat @ cumulative_var @ b_hat.T + sigma2_hat
                            forecast_std[h, :] = np.sqrt(np.diag(cumulative_var))
                        
                        # Confidence intervals
                        z_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                        
                        forecast_upper = forecasts + z_alpha * forecast_std
                        forecast_lower = forecasts - z_alpha * forecast_std
                        
                        # Convert to prices if using log
                        if use_log:
                            price_forecast = np.exp(forecasts[:, forecast_asset_idx])
                            price_upper = np.exp(forecast_upper[:, forecast_asset_idx])
                            price_lower = np.exp(forecast_lower[:, forecast_asset_idx])
                            historical_prices = np.exp(x[:, forecast_asset_idx])
                        else:
                            price_forecast = forecasts[:, forecast_asset_idx]
                            price_upper = forecast_upper[:, forecast_asset_idx]
                            price_lower = forecast_lower[:, forecast_asset_idx]
                            historical_prices = x[:, forecast_asset_idx]
                        
                        # Create forecast dates
                        last_date = dates[-1]
                        forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='B')
                        
                        # Plot
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig_forecast = go.Figure()
                            
                            # Historical data (last 60 days)
                            hist_start = max(0, len(dates) - 60)
                            
                            fig_forecast.add_trace(go.Scatter(
                                x=dates[hist_start:],
                                y=historical_prices[hist_start:],
                                mode='lines',
                                name='Historical',
                                line=dict(color=CHART_COLORS[forecast_asset_idx % len(CHART_COLORS)], width=2)
                            ))
                            
                            # Confidence interval
                            fig_forecast.add_trace(go.Scatter(
                                x=list(forecast_dates) + list(forecast_dates[::-1]),
                                y=list(price_upper) + list(price_lower[::-1]),
                                fill='toself',
                                fillcolor=f'rgba({99}, {102}, {241}, 0.2)',
                                line=dict(color='rgba(0,0,0,0)'),
                                name=f'{confidence_level*100:.0f}% CI',
                                hoverinfo='skip'
                            ))
                            
                            # Point forecast
                            fig_forecast.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=price_forecast,
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='#FFE66D', width=2, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            # Equilibrium line
                            eq_price = np.exp(mu_longrun[forecast_asset_idx]) if use_log else mu_longrun[forecast_asset_idx]
                            fig_forecast.add_hline(
                                y=eq_price, line_dash="dot", line_color="#4ECDC4",
                                annotation_text=f"Equilibrium: ${eq_price:.2f}"
                            )
                            
                            fig_forecast.update_layout(
                                height=400,
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                title=f"{get_display_name(mr['assets'][forecast_asset_idx])}: {forecast_horizon}-Day Forecast",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                hovermode='x unified'
                            )
                            fig_forecast = apply_plotly_theme(fig_forecast)
                            st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        with col2:
                            st.markdown("##### Forecast Summary")
                            
                            current_price = price_forecast[0]
                            final_forecast = price_forecast[-1]
                            final_upper = price_upper[-1]
                            final_lower = price_lower[-1]
                            
                            expected_return = (final_forecast / current_price - 1) * 100
                            
                            forecast_summary = {
                                'Metric': [
                                    'Current Price',
                                    f'{forecast_horizon}d Forecast',
                                    f'{confidence_level*100:.0f}% Upper',
                                    f'{confidence_level*100:.0f}% Lower',
                                    'Expected Return',
                                    'Equilibrium'
                                ],
                                'Value': [
                                    f"${current_price:.2f}",
                                    f"${final_forecast:.2f}",
                                    f"${final_upper:.2f}",
                                    f"${final_lower:.2f}",
                                    f"{expected_return:+.2f}%",
                                    f"${eq_price:.2f}"
                                ]
                            }
                            st.markdown(create_styled_table(pd.DataFrame(forecast_summary)), unsafe_allow_html=True)
                            
                            # Direction probability
                            st.markdown("##### Direction Probabilities")
                            
                            # Probability of price going up/down
                            # P(X_{t+h} > X_t) using normal approximation
                            forecast_mean_h = forecasts[-1, forecast_asset_idx]
                            forecast_std_h = forecast_std[-1, forecast_asset_idx]
                            current_x = x[-1, forecast_asset_idx]
                            
                            if forecast_std_h > 0:
                                prob_up = 1 - stats.norm.cdf(current_x, forecast_mean_h, forecast_std_h)
                                prob_down = stats.norm.cdf(current_x, forecast_mean_h, forecast_std_h)
                            else:
                                prob_up = 0.5
                                prob_down = 0.5
                            
                            # Probability of reaching equilibrium
                            if forecast_std_h > 0:
                                prob_reach_eq = 1 - abs(stats.norm.cdf(mu_longrun[forecast_asset_idx], forecast_mean_h, forecast_std_h) - 0.5) * 2
                            else:
                                prob_reach_eq = 0
                            
                            prob_col1, prob_col2 = st.columns(2)
                            
                            with prob_col1:
                                up_color = "#4ECDC4" if prob_up > 0.5 else "#94a3b8"
                                st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; 
                                            background: {up_color}22; border-radius: 8px;'>
                                    <div style='font-size: 1.5rem; font-weight: bold; color: {up_color};'>
                                        {prob_up*100:.0f}%
                                    </div>
                                    <div style='color: #94a3b8; font-size: 0.85rem;'>Price UP</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with prob_col2:
                                down_color = "#FF6B6B" if prob_down > 0.5 else "#94a3b8"
                                st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; 
                                            background: {down_color}22; border-radius: 8px;'>
                                    <div style='font-size: 1.5rem; font-weight: bold; color: {down_color};'>
                                        {prob_down*100:.0f}%
                                    </div>
                                    <div style='color: #94a3b8; font-size: 0.85rem;'>Price DOWN</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Forecast table for key horizons
                        st.markdown("##### Forecast at Key Horizons")
                        
                        key_horizons = [5, 10, 20, forecast_horizon] if forecast_horizon > 20 else [5, 10, forecast_horizon]
                        key_horizons = [h for h in key_horizons if h <= forecast_horizon]
                        
                        horizon_data = []
                        for h in key_horizons:
                            h_price = price_forecast[h]
                            h_upper = price_upper[h]
                            h_lower = price_lower[h]
                            h_return = (h_price / current_price - 1) * 100
                            
                            horizon_data.append({
                                'Horizon': f'{h} days',
                                'Forecast': f'${h_price:.2f}',
                                f'{confidence_level*100:.0f}% Range': f'${h_lower:.2f} - ${h_upper:.2f}',
                                'Exp. Return': f'{h_return:+.2f}%',
                                'Uncertainty': f'±${(h_upper - h_lower)/2:.2f}'
                            })
                        
                        st.markdown(create_styled_table(pd.DataFrame(horizon_data)), unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # ============================================================
                        # SECTION 6: SUMMARY & RECOMMENDATIONS
                        # ============================================================
                        st.markdown("## 💼 Summary & Recommendations")
                        
                        # Build comprehensive summary
                        findings = []
                        recommendations = []
                        warnings = []
                        
                        # 1. System stability
                        if mr['is_stable']:
                            findings.append(f"✅ **System is stable** - all eigenvalues inside unit circle (max |λ| = {np.max(np.abs(eigenvalues)):.3f})")
                        else:
                            warnings.append(f"⚠️ **System is unstable** - contains unit roots (max |λ| = {np.max(np.abs(eigenvalues)):.3f})")
                            recommendations.append("Consider differencing the data or focusing on cointegrated spreads only")
                        
                        # 2. Mean-reversion speed
                        tradable_assets = []
                        for i, asset in enumerate(mr['assets']):
                            b_i = b_hat[i, i] if n_assets > 1 else b_hat[0, 0]
                            if np.abs(b_i) < 1:
                                hl = np.log(2) / (-np.log(np.abs(b_i)))
                                if hl < 20:
                                    tradable_assets.append((get_display_name(asset), hl, 'excellent'))
                                elif hl < 60:
                                    tradable_assets.append((get_display_name(asset), hl, 'moderate'))
                        
                        if tradable_assets:
                            excellent = [a for a in tradable_assets if a[2] == 'excellent']
                            if excellent:
                                findings.append(f"✅ **Fast mean-reversion detected** in: {', '.join([a[0] + f' ({a[1]:.0f}d)' for a in excellent])}")
                                recommendations.append(f"Consider mean-reversion strategies on {', '.join([a[0] for a in excellent])}")
                        else:
                            findings.append("ℹ️ No assets show fast mean-reversion (half-life < 20 days)")
                        
                        # 3. Cointegration
                        if n_assets >= 2 and c_hat is not None and c_hat.size > 0:
                            n_coint = c_hat.shape[1] if len(c_hat.shape) > 1 else 1
                            if n_coint > 0:
                                findings.append(f"✅ **{n_coint} cointegration relationship(s) found** - tradable spreads available")
                                recommendations.append("Consider pairs trading / statistical arbitrage on the identified spreads")
                                
                                # Check spread Z-scores
                                for coint_idx in range(n_coint):
                                    c_vec = c_hat[:, coint_idx] if len(c_hat.shape) > 1 else c_hat
                                    spread = x @ c_vec
                                    spread_z = (spread[-1] - np.mean(spread)) / np.std(spread)
                                    
                                    if abs(spread_z) > 2:
                                        signal = "SELL" if spread_z > 0 else "BUY"
                                        recommendations.append(f"🚨 **Active signal on Spread {coint_idx+1}**: {signal} (Z = {spread_z:+.2f})")
                            else:
                                findings.append("ℹ️ No cointegration found between selected assets")
                        
                        # 4. Current positioning
                        extreme_assets = []
                        for i, asset in enumerate(mr['assets']):
                            z = z_scores[i]
                            if abs(z) > 2:
                                extreme_assets.append((get_display_name(asset), z))
                        
                        if extreme_assets:
                            for asset, z in extreme_assets:
                                if z > 2:
                                    warnings.append(f"🔴 **{asset}** is {z:.1f}σ ABOVE equilibrium - potential mean-reversion SHORT")
                                else:
                                    findings.append(f"🟢 **{asset}** is {abs(z):.1f}σ BELOW equilibrium - potential mean-reversion LONG")
                        
                        # 5. Forecast insights
                        if expected_return > 5:
                            findings.append(f"📈 **{forecast_horizon}-day forecast** for {get_display_name(mr['assets'][forecast_asset_idx])}: +{expected_return:.1f}% expected return")
                        elif expected_return < -5:
                            warnings.append(f"📉 **{forecast_horizon}-day forecast** for {get_display_name(mr['assets'][forecast_asset_idx])}: {expected_return:.1f}% expected return")
                        
                        # Display
                        if findings:
                            st.markdown("### 📋 Key Findings")
                            for f in findings:
                                st.markdown(f"- {f}")
                        
                        if recommendations:
                            st.markdown("### 💡 Recommendations")
                            for i, r in enumerate(recommendations, 1):
                                st.markdown(f"{i}. {r}")
                        
                        if warnings:
                            st.markdown("### ⚠️ Warnings")
                            for w in warnings:
                                st.markdown(f"- {w}")
                        
                        # Risk disclaimer
                        st.markdown("---")
                        st.caption("""
                        **Important Disclaimers:**
                        
                        1. **Model limitations:** VAR and O-U models assume linear dynamics and constant parameters. 
                           Real markets exhibit regime changes, fat tails, and time-varying volatility.
                        
                        2. **Estimation uncertainty:** Parameters are estimated from historical data and subject to estimation error.
                           Shorter samples = more uncertainty.
                        
                        3. **Cointegration can break:** Historical cointegration relationships may not persist in the future,
                           especially during market stress or structural changes.
                        
                        4. **Not financial advice:** These tools are for educational and research purposes. 
                           Always conduct your own analysis and consider risk management before trading.
                        
                        📖 *Methodology based on: Hamilton (1994), Engle & Granger (1987), Meucci (2005, 2010)*
                        """)
                        
                        # Academic references
                        with st.expander("📚 Academic References"):
                            st.markdown("""
                            **Core Methodology:**
                            
                            1. **Uhlenbeck, G.E. & Ornstein, L.S. (1930).** "On the Theory of Brownian Motion." 
                               *Physical Review*, 36(5), 823-841.
                            
                            2. **Hamilton, J.D. (1994).** *Time Series Analysis.* Princeton University Press.
                               - Chapters 10-11: VAR models
                               - Chapter 19: Cointegration
                            
                            3. **Engle, R.F. & Granger, C.W.J. (1987).** "Co-Integration and Error Correction: 
                               Representation, Estimation, and Testing." *Econometrica*, 55(2), 251-276.
                            
                            4. **Johansen, S. (1991).** "Estimation and Hypothesis Testing of Cointegration Vectors 
                               in Gaussian Vector Autoregressive Models." *Econometrica*, 59(6), 1551-1580.
                            
                            5. **Meucci, A. (2005).** *Risk and Asset Allocation.* Springer.
                               - Chapter 3: Modeling the market
                               - Chapter 7: Estimating the distribution of the market invariants
                            
                            6. **Meucci, A. (2010).** "Historical Scenarios with Fully Flexible Probabilities." 
                               *GARP Risk Professional*, December 2010.
                            
                            **Trading Applications:**
                            
                            7. **Vidyamurthy, G. (2004).** *Pairs Trading: Quantitative Methods and Analysis.* Wiley.
                            
                            8. **Pole, A. (2007).** *Statistical Arbitrage: Algorithmic Trading Insights and Techniques.* Wiley.
                            
                            9. **Avellaneda, M. & Lee, J.H. (2010).** "Statistical Arbitrage in the US Equities Market." 
                               *Quantitative Finance*, 10(7), 761-782.
                            """)        
        
        
        # Part 7: Backtest Validation Tab
        # TAB 5: BACKTEST VALIDATION - COMPLETE REWRITE
        with tab5:
            st.markdown("### 🧪 Backtest Validation")
            
            st.markdown("""
            This tab helps you answer a critical question that standard analysis ignores:
            
            > **"If I had optimized this strategy in the past, would it have actually worked going forward?"**
            
            We split your data into a **training period** (where we optimize) and a **test period** 
            (where we evaluate with frozen weights). This simulates real-world investing.
            """)
            
            st.markdown("---")
            
            # ================================================================
            # SECTION 1: WALK-FORWARD VISUALIZATION
            # ================================================================
            st.markdown("## 📊 Walk-Forward Analysis")
            
            # Configuration row
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                split_options = {
                    "50/50": 0.50,
                    "60/40": 0.60,
                    "70/30": 0.70,
                    "80/20": 0.80
                }
                split_choice = st.selectbox(
                    "Train/Test Split",
                    options=list(split_options.keys()),
                    index=2,  # Default 70/30
                    key="wf_split",
                    help="Percentage of data used for optimization vs testing"
                )
                train_pct = split_options[split_choice]
            
            with col2:
                # Strategy selector - use strategies from main analysis
                available_strategies = list(analyzer.portfolios.keys())
                strategy_names_map = {k: v['name'] for k, v in analyzer.portfolios.items()}
                
                selected_strategies = st.multiselect(
                    "Strategies to Test",
                    options=available_strategies,
                    default=available_strategies[:4],  # First 4 by default
                    format_func=lambda x: strategy_names_map[x],
                    key="wf_strategies"
                )
            
            with col3:
                # Calculate split dates
                n_days = len(analyzer.returns)
                split_idx = int(n_days * train_pct)
                train_end_date = analyzer.returns.index[split_idx - 1]
                test_start_date = analyzer.returns.index[split_idx]
                
                st.markdown("**Period Split:**")
                st.caption(f"Train: {analyzer.returns.index[0].strftime('%Y-%m-%d')} → {train_end_date.strftime('%Y-%m-%d')}")
                st.caption(f"Test: {test_start_date.strftime('%Y-%m-%d')} → {analyzer.returns.index[-1].strftime('%Y-%m-%d')}")
            
            # Run button
            if len(selected_strategies) < 1:
                st.warning("⚠️ Select at least one strategy to analyze.")
            else:
                if st.button("🚀 Run Walk-Forward Analysis", use_container_width=True, key="run_wf_main"):
                    
                    with st.spinner("Optimizing on training period and evaluating on test period..."):
                        
                        # Split data
                        returns_train = analyzer.returns.iloc[:split_idx]
                        returns_test = analyzer.returns.iloc[split_idx:]
                        
                        # Store results
                        wf_results = {}
                        
                        progress_bar = st.progress(0)
                        
                        for i, strategy_key in enumerate(selected_strategies):
                            progress_bar.progress((i + 1) / len(selected_strategies))
                            
                            # Get optimization method
                            method_map = {
                                'equally_weighted': 'equal',
                                'min_volatility': 'min_vol',
                                'max_return': 'max_return',
                                'max_sharpe': 'max_sharpe',
                                'risk_parity': 'risk_parity',
                                'markowitz': 'markowitz',  # Use max_sharpe as proxy
                                'hrp': 'risk_parity',  # HRP uses different logic, approximate with RP
                                'custom': 'custom'  # Custom portfolio uses fixed weights
                            }
                            
                            method = method_map.get(strategy_key, 'equal')
                            strategy_name = strategy_names_map[strategy_key]
                            
                            # Optimize on TRAINING data only
                            if method == 'custom':
                                # Custom portfolio: use user-defined weights (no optimization)
                                weights_train = analyzer.portfolios['custom']['weights']
                            else:
                                weights_train = optimize_portfolio_weights(
                                    returns_train,
                                    method=method,
                                    rf_rate=rf_rate
                                )                            
                            
                            # Calculate returns for BOTH periods using weights from training
                            train_portfolio_returns = returns_train.dot(weights_train)
                            test_portfolio_returns = returns_test.dot(weights_train)
                            
                            # Calculate metrics for both periods
                            train_metrics = calculate_portfolio_metrics(train_portfolio_returns, rf_rate)
                            test_metrics = calculate_portfolio_metrics(test_portfolio_returns, rf_rate)
                            
                            # Build cumulative series
                            train_cumulative = (1 + train_portfolio_returns).cumprod()
                            test_cumulative = (1 + test_portfolio_returns).cumprod()
                            
                            # Chain test to end of train (continuous growth)
                            test_cumulative_chained = test_cumulative * train_cumulative.iloc[-1]
                            
                            # Full series
                            full_cumulative = pd.concat([train_cumulative, test_cumulative_chained])
                            
                            # Stability ratio
                            if train_metrics['sharpe'] != 0 and not np.isnan(train_metrics['sharpe']):
                                sharpe_stability = test_metrics['sharpe'] / train_metrics['sharpe']
                            else:
                                sharpe_stability = 0
                            
                            wf_results[strategy_key] = {
                                'name': strategy_name,
                                'weights': weights_train,
                                'train_metrics': train_metrics,
                                'test_metrics': test_metrics,
                                'train_cumulative': train_cumulative,
                                'test_cumulative_chained': test_cumulative_chained,
                                'full_cumulative': full_cumulative,
                                'train_returns': train_portfolio_returns,
                                'test_returns': test_portfolio_returns,
                                'sharpe_stability': sharpe_stability
                            }
                        
                        progress_bar.empty()
                    
                    # Store in session state for persistence
                    st.session_state.wf_results = wf_results
                    st.session_state.wf_split_idx = split_idx
                    st.session_state.wf_train_end = train_end_date
                    st.session_state.wf_test_start = test_start_date
                
                # Display results if available
                if 'wf_results' in st.session_state and st.session_state.wf_results:
                    
                    wf_results = st.session_state.wf_results
                    split_idx = st.session_state.wf_split_idx
                    train_end_date = st.session_state.wf_train_end
                    test_start_date = st.session_state.wf_test_start
                    
                    st.markdown("---")
                    
                    # ===== MAIN VISUALIZATION =====
                    st.markdown("#### 📈 Performance: Training vs Test Period")
                    
                    fig = go.Figure()
                    
                    for i, (strategy_key, result) in enumerate(wf_results.items()):
                        color = CHART_COLORS[i % len(CHART_COLORS)]
                        
                        # Training period - solid line
                        fig.add_trace(go.Scatter(
                            x=result['train_cumulative'].index,
                            y=(result['train_cumulative'] - 1) * 100,
                            mode='lines',
                            name=f"{result['name']} (Train)",
                            line=dict(color=color, width=2.5),
                            legendgroup=strategy_key,
                            hovertemplate=f"<b>{result['name']} - Train</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>"
                        ))
                        
                        # Test period - dashed line, same color
                        fig.add_trace(go.Scatter(
                            x=result['test_cumulative_chained'].index,
                            y=(result['test_cumulative_chained'] - 1) * 100,
                            mode='lines',
                            name=f"{result['name']} (Test)",
                            line=dict(color=color, width=3, dash='dot'),
                            legendgroup=strategy_key,
                            hovertemplate=f"<b>{result['name']} - Test</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>"
                        ))
                    
                    # Vertical line at split point
                    fig.add_vline(
                        x=train_end_date,
                        line_dash="dash",
                        line_color="rgba(255, 255, 255, 0.8)",
                        line_width=2
                    )
                    
                    # Annotation for split
                    fig.add_annotation(
                        x=train_end_date,
                        y=1.05,
                        yref="paper",
                        text="◄ OPTIMIZATION | VALIDATION ►",
                        showarrow=False,
                        font=dict(size=11, color="white"),
                        bgcolor="rgba(99, 102, 241, 0.8)",
                        borderpad=4
                    )
                    
                    # Shaded test region
                    fig.add_vrect(
                        x0=test_start_date,
                        x1=analyzer.returns.index[-1],
                        fillcolor="rgba(99, 102, 241, 0.1)",
                        layer="below",
                        line_width=0
                    )
                    
                    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
                    
                    fig.update_layout(
                        height=550,
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return (%)",
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.08,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=10)
                        ),
                        margin=dict(t=80)
                    )
                    
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div style='text-align: center; color: var(--text-secondary); font-size: 0.85rem; margin-top: -10px;'>
                        <strong>Solid lines</strong> = Training period (weights optimized here) | 
                        <strong>Dotted lines</strong> = Test period (weights frozen, performance measured)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # ===== METRICS COMPARISON TABLE =====
                    st.markdown("#### 📊 Train vs Test Metrics")
                    
                    comparison_data = []
                    
                    for strategy_key, result in wf_results.items():
                        train = result['train_metrics']
                        test = result['test_metrics']
                        stability = result['sharpe_stability']
                        
                        # Stability indicator
                        if stability > 0.8:
                            stab_icon = "🟢"
                        elif stability > 0.5:
                            stab_icon = "🟡"
                        elif stability > 0:
                            stab_icon = "🟠"
                        else:
                            stab_icon = "🔴"
                        
                        comparison_data.append({
                            'Strategy': result['name'],
                            'Train Return': f"{train['return']*100:.2f}%",
                            'Test Return': f"{test['return']*100:.2f}%",
                            'Δ Return': f"{(test['return'] - train['return'])*100:+.2f}%",
                            'Train Sharpe': f"{train['sharpe']:.3f}",
                            'Test Sharpe': f"{test['sharpe']:.3f}",
                            'Stability': f"{stab_icon} {stability:.2f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.markdown(create_styled_table(comparison_df, "Performance Comparison"), unsafe_allow_html=True)
                    
                    st.markdown("""
                    **Stability Ratio** = Test Sharpe ÷ Train Sharpe
                    - 🟢 **> 0.8**: Excellent stability
                    - 🟡 **0.5 - 0.8**: Moderate degradation  
                    - 🟠 **0 - 0.5**: Significant degradation
                    - 🔴 **< 0**: Strategy failed out-of-sample
                    """)
                    
                    st.markdown("---")
                    
                    # ===== STABILITY RANKING =====
                    st.markdown("#### 🎯 Stability Ranking")
                    
                    col1, col2 = st.columns([1.5, 1])
                    
                    with col1:
                        # Bar chart
                        sorted_results = sorted(wf_results.items(), key=lambda x: x[1]['sharpe_stability'], reverse=True)
                        
                        strategies_sorted = [r[1]['name'] for r in sorted_results]
                        stabilities_sorted = [r[1]['sharpe_stability'] for r in sorted_results]
                        colors_sorted = [
                            '#4ECDC4' if s > 0.8 else '#FFE66D' if s > 0.5 else '#FF9F43' if s > 0 else '#FF6B6B' 
                            for s in stabilities_sorted
                        ]
                        
                        fig_stab = go.Figure()
                        
                        fig_stab.add_trace(go.Bar(
                            x=strategies_sorted,
                            y=stabilities_sorted,
                            marker_color=colors_sorted,
                            text=[f"{s:.2f}" for s in stabilities_sorted],
                            textposition='outside',
                            textfont=dict(color='#E2E8F0', size=11)
                        ))
                        
                        fig_stab.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.5)",
                                        annotation_text="Perfect", annotation_position="right")
                        fig_stab.add_hline(y=0.8, line_dash="dot", line_color="#4ECDC4")
                        fig_stab.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
                        
                        fig_stab.update_layout(
                            height=350,
                            yaxis_title="Stability Ratio",
                            xaxis_title="",
                            showlegend=False
                        )
                        
                        fig_stab = apply_plotly_theme(fig_stab)
                        st.plotly_chart(fig_stab, use_container_width=True)
                    
                    with col2:
                        st.markdown("##### 🏆 Summary")
                        
                        best_stability = sorted_results[0]
                        worst_stability = sorted_results[-1]
                        best_test_sharpe = max(wf_results.items(), key=lambda x: x[1]['test_metrics']['sharpe'])
                        
                        st.metric(
                            "Most Stable",
                            best_stability[1]['name'].split()[0],
                            f"Ratio: {best_stability[1]['sharpe_stability']:.2f}"
                        )
                        
                        st.metric(
                            "Best Test Sharpe",
                            best_test_sharpe[1]['name'].split()[0],
                            f"{best_test_sharpe[1]['test_metrics']['sharpe']:.3f}"
                        )
                        
                        st.metric(
                            "Least Stable",
                            worst_stability[1]['name'].split()[0],
                            f"Ratio: {worst_stability[1]['sharpe_stability']:.2f}",
                            delta_color="inverse"
                        )
                    
                    st.markdown("---")
                    
                    # ===== INTERPRETATION =====
                    with st.expander("💡 How to Interpret These Results", expanded=False):
                        st.markdown(f"""
                        ### What This Analysis Shows
                        
                        We simulated what would have happened if you had:
                        
                        1. **Optimized** each strategy using data up to **{train_end_date.strftime('%B %d, %Y')}**
                        2. **Invested** with those frozen weights from **{test_start_date.strftime('%B %d, %Y')}** onwards
                        3. **Measured** how well the optimization held up on unseen data
                        
                        ---
                        
                        ### Why Strategies Degrade Out-of-Sample
                        
                        Most optimization strategies show some degradation because:
                        
                        - **Estimation error**: Historical returns/covariances don't perfectly predict the future
                        - **Regime changes**: Market conditions may shift between train and test periods
                        - **Overfitting**: Complex strategies may fit noise rather than signal
                        
                        > DeMiguel et al. (2009) showed that ~250 years of data would be needed for 
                        > mean-variance optimization to reliably outperform simple equal weighting.
                        
                        ---
                        
                        ### What to Look For
                        
                        ✅ **Stability > 0.8**: Strategy is robust—likely to perform similarly in the future
                        
                        ⚠️ **Stability 0.5-0.8**: Some degradation—be cautious with expectations
                        
                        🚨 **Stability < 0.5**: Significant overfitting—the strategy may not be reliable
                        
                        ---
                        
                        ### Key Insight
                        
                        Simpler strategies (Equal Weight, Risk Parity) often show **better stability** 
                        because they have less parameters to overfit. This is the classic 
                        **bias-variance tradeoff** in action.
                        """)
                    
                    st.markdown("---")
            
            # ================================================================
            # SECTION 2: ADVANCED CPCV ANALYSIS (ORDINE OTTIMIZZATO)
            # ================================================================
            st.markdown("### 🧪 Advanced Validation (AFML – Combinatorial Purged CV)")
            
            # ==========================
            # INTRODUCTORY SECTION
            # ==========================
            st.markdown("""
            ## 🎯 What Does This Section Do?
            
            This section implements **Combinatorial Purged Cross-Validation (CPCV)**, a rigorous framework from 
            **Marcos López de Prado's "Advances in Financial Machine Learning" (2018, Chapter 7)**.
            
            ### The Problem with Single Train/Test Splits
            
            The Walk-Forward analysis above uses a single split. While useful, this has limitations:
            
            - **Single regime**: What if the test period had unusual market conditions?
            - **Luck vs skill**: One good result could be random chance
            - **Selection bias**: Choosing the "best" strategy from one split is prone to overfitting
            
            ### The CPCV Solution
            
            **How it works:**
            
            1. **Split time into K folds** (e.g., 5 periods of ~2 years each)
            2. **Create all possible train/test combinations** (e.g., 10 different splits)
            3. **For each split:**
               - Optimize strategy on **training data only**
               - Evaluate performance on **held-out test data**
               - Apply **embargo** to prevent information leakage
            4. **Aggregate results** across all splits to get a **distribution** of out-of-sample performance
            5. **Calculate PBO** to measure overfitting risk

            ### What You Learn
            
            ✅ **Robustness:** Does the strategy work across different market regimes?  
            ✅ **Stability:** How consistent is performance over time?  
            ✅ **Overfitting Risk:** Is the high Sharpe due to skill or luck?  
            ✅ **Tail Risk:** What's the worst-case scenario in unseen data?
            """)
            
            with st.expander("📚 Mathematical Framework"):
                st.markdown(r"""
                ### Combinatorial Purged Cross-Validation
                
                **Notation:**
                - $T = \{t_1, t_2, ..., t_N\}$ = time series of dates
                - $K$ = number of folds
                - $N^*$ = number of test folds per split
                
                **Algorithm:**
                
                1. **Partition** $T$ into $K$ consecutive folds: $T = F_1 \cup F_2 \cup ... \cup F_K$
                
                2. **Generate** all $\binom{K}{N^*}$ combinations of test folds
                
                3. **For each combination** $\mathcal{C} = \{i_1, ..., i_{N^*}\}$:
                
                - **Test set:** $T_{test} = F_{i_1} \cup ... \cup F_{i_{N^*}}$
                
                - **Embargo set:** $E = \{t \in T : t > \max(T_{test}), t \leq \max(T_{test}) + \delta\}$  
                    where $\delta$ is the embargo period
                
                - **Train set:** $T_{train} = T \setminus (T_{test} \cup E)$
                
                4. **Optimize** portfolio weights $w^*$ on $T_{train}$:
                
                $$w^* = \arg\max_w \text{Objective}(R_{train}, w)$$
                
                5. **Evaluate** on $T_{test}$:
                
                $$\text{Performance}_{OOS} = f(R_{test} \cdot w^*)$$
                
                ---
                
                ### Risk-Adjusted Performance Metrics
                
                Let $r_t$ be the portfolio return at time $t$, and $r_f$ the risk-free rate.
                
                **Sharpe Ratio:**
                $$\text{Sharpe} = \frac{\mathbb{E}[r_t - r_f]}{\sigma[r_t]} \times \sqrt{252}$$
                
                - Measures return per unit of **total volatility**
                - Assumes normally distributed returns
                - **Limitation:** Penalizes upside volatility equally with downside
                
                **Sortino Ratio** (recommended):
                $$\text{Sortino} = \frac{\mathbb{E}[r_t - r_f]}{\sigma^-[r_t]} \times \sqrt{252}$$
                
                where $\sigma^-[r_t] = \sqrt{\mathbb{E}[\min(r_t - r_f, 0)^2]}$ is the **downside deviation**.
                
                - Only penalizes **downside volatility**
                - More aligned with investor preferences
                - Better for asymmetric return distributions
                
                **Calmar Ratio** (recommended):
                $$\text{Calmar} = \frac{\text{Annualized Return}}{\text{Maximum Drawdown}}$$
                
                where Maximum Drawdown is:
                $$\text{MDD} = \max_{t \in [0,T]} \left[ \frac{\max_{s \leq t} V_s - V_t}{\max_{s \leq t} V_s} \right]$$
                
                - Measures return per unit of **worst-case loss**
                - Directly quantifies tail risk
                - Intuitive for risk management
                
                **Conditional Value at Risk (CVaR)**:
                $$\text{CVaR}_\alpha = \mathbb{E}[r_t \mid r_t \leq \text{VaR}_\alpha]$$
                
                - Average of the worst $\alpha$% of returns
                - Captures tail risk beyond VaR
                - Used in Basel III regulations
                
                ---
                
                ### Probability of Backtest Overfitting (PBO)
                
                **Definition:** The probability that the strategy with the best in-sample performance 
                actually underperforms out-of-sample.
                
                **Algorithm:**
                
                1. For each split $s$, rank all strategies by **in-sample** metric → get best strategy $\pi_s^{IS}$
                
                2. Find the **out-of-sample rank** $R_s^{OOS}$ of strategy $\pi_s^{IS}$
                
                3. Compute the probability:
                
                $$\text{PBO} = P\left(R^{OOS} > \frac{M+1}{2}\right)$$
                
                where $M$ is the number of strategies.
                
                **Interpretation:**
                - **PBO < 10%**: Excellent - strategy selection is robust
                - **PBO < 30%**: Acceptable - results are reliable
                - **PBO > 50%**: Warning - severe overfitting detected
                
                **Reference:** Bailey, D. H., & López de Prado, M. (2014). "The Probability of Backtest Overfitting." 
                *Journal of Computational Finance*, 20(4).
                
                """)
            
            st.markdown("---")
            st.markdown("## ⚙️ Configuration")

            # ==========================
            # CONFIGURATION
            # ==========================
            col1, col2 = st.columns(2)
            with col1:
                n_splits = st.selectbox("Number of folds (K)", [5, 6, 8], index=0, key="cpcv_n_splits")
            with col2:
                n_test_splits = st.selectbox("Test folds per split", [1, 2], index=1, key="cpcv_n_test")

            embargo_pct = st.slider(
                "Embargo (% of dataset)",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.25,
                key="cpcv_embargo"
            ) / 100
            st.caption(
                "🛑 **Embargo** removes a buffer of data *after* the test period to prevent "
                "information leakage due to autocorrelation. \n\n"
                "An embargo of **1%–5%** means that data immediately following the test window "
                "is excluded from training. Higher values make validation more conservative."
            )

            # Build available methods list
            available_cpcv_methods = ["Equally Weighted", "Minimum Volatility", "Maximum Sharpe", "Risk Parity", "Hierarchical Risk Parity", "CVaR (95%)"]
            if 'custom' in analyzer.portfolios:
                available_cpcv_methods.append("Your Portfolio")

            methods_to_test = st.multiselect(
                "Strategies to validate",
                available_cpcv_methods,
                default=["Equally Weighted", "Minimum Volatility", "Maximum Sharpe", "Risk Parity", "CVaR (95%)"],
                key="cpcv_methods"
            )

            # Primary metric for PBO calculation
            primary_metric_display = st.selectbox(
                "Primary metric for PBO",
                ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
                index=1,
                help="The metric used to calculate Probability of Backtest Overfitting. This determines which metric ranks strategies for the PBO calculation.",
                key="cpcv_primary_metric"
            )

            metric_key_map = {
                "Sharpe Ratio": "sharpe",
                "Sortino Ratio": "sortino",
                "Calmar Ratio": "calmar"
            }
            primary_metric = metric_key_map[primary_metric_display]

            # ==========================
            # RUN BACKTEST
            # ==========================
            if st.button("🚀 Run Multi-Metric Validation", use_container_width=True, key="run_cpcv_btn"):
                if not methods_to_test:
                    st.error("⚠️ Select at least one strategy.")
                elif len(methods_to_test) < 2:
                    st.error("⚠️ Select at least 2 strategies for meaningful comparison.")
                else:
                    with st.spinner("Running Combinatorial Purged Cross-Validation..."):
                        try:
                            is_metrics_all, oos_metrics_all, oos_returns = run_cpcv_backtest(
                                analyzer.returns,
                                methods_to_test,
                                rf_rate,
                                n_splits,
                                n_test_splits,
                                embargo_pct
                            )
                            
                            pbo = compute_pbo(is_metrics_all, oos_metrics_all, primary_metric)
                            
                            n_valid_splits = len([m for m in oos_metrics_all[methods_to_test[0]] 
                                                if not np.isnan(m['sharpe'])])
                            st.success(f"✅ Validation completed ({n_valid_splits} valid splits)")
                            
                        except ValueError as e:
                            st.error(f"❌ Configuration error: {str(e)}")
                            st.stop()
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {str(e)}")
                            st.stop()

                    st.markdown("---")

                    # ==========================
                    # 1. PBO METRIC (PRIMA - METRICA RIASSUNTIVA)
                    # ==========================
                    st.markdown(f"## 📉 Probability of Backtest Overfitting")
                    
                    st.markdown(f"""
                    The PBO measures how often the strategy that looks best on **{primary_metric_display}** 
                    during training actually underperforms out-of-sample. This is the single most important 
                    metric to assess overfitting risk.
                    """)
                    
                    if np.isnan(pbo):
                        st.warning("⚠️ PBO could not be calculated (insufficient valid splits)")
                    else:
                        # Visual display of PBO
                        if pbo < 0.1:
                            delta_color = "normal"
                            delta_text = "Excellent"
                            status_color = "#4ECDC4"
                        elif pbo < 0.3:
                            delta_color = "normal"
                            delta_text = "Good"
                            status_color = "#FFE66D"
                        else:
                            delta_color = "inverse"
                            delta_text = "Warning"
                            status_color = "#FF6B6B"
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("PBO", f"{pbo:.1%}", delta_text, delta_color=delta_color)
                        with col2:
                            st.metric("Primary Metric", primary_metric_display)
                        with col3:
                            st.metric("Valid Splits", n_valid_splits)
                        
                        # Interpretation box
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {status_color}22, {status_color}11); 
                                    border-left: 4px solid {status_color}; padding: 16px; border-radius: 8px; margin: 16px 0;'>
                            <strong style='color: {status_color}; font-size: 1.1em;'>Interpretation:</strong><br>
                            <span style='color: #E2E8F0; font-size: 0.95em;'>
                                {'🟢 <strong>Excellent robustness.</strong> Your strategies show consistent performance across different market conditions with minimal overfitting.' if pbo < 0.1 
                                 else '🟡 <strong>Acceptable reliability.</strong> Some overfitting detected, but results are still meaningful. Monitor live performance carefully.' if pbo < 0.3
                                 else '🔴 <strong>High overfitting risk.</strong> The strategy that looks best in training often fails out-of-sample. Consider simpler strategies or ensemble approaches.'}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption("""
                        **📖 What is PBO?**  
                        PBO answers: "If I pick the strategy with the best training performance, what's the probability it will underperform out-of-sample?"  
                        • **< 10%** = Excellent (strategy selection is robust)  
                        • **< 30%** = Acceptable (results are reliable)  
                        • **> 50%** = Warning (severe overfitting detected)
                        """)
                    
                    st.markdown("---")

                    # ==========================
                    # 2. RANKING TABLE (SECONDA - CHI HA VINTO)
                    # ==========================
                    st.markdown("## 🏆 Strategy Ranking (Out-of-Sample)")
                    
                    st.markdown(f"""
                    Strategies ranked by **{primary_metric_display}** (median across all {n_valid_splits} splits). 
                    This ranking reflects real-world performance on unseen data.
                    """)

                    # Method names mapping
                    method_names = {
                        "Equally Weighted": "Equally Weighted",
                        "Minimum Volatility": "Minimum Volatility",
                        "Maximum Sharpe": "Maximum Sharpe",
                        "Risk Parity": "Risk Parity",
                        "Maximum Return": "Maximum Return",
                        "Hierarchical Risk Parity": "Hierarchical Risk Parity",
                        "CVaR (95%)": "CVaR (95%)",
                        "Your Portfolio": "Your Portfolio"
                    }

                    ranking_data = []
                    ranking_data_raw = []
                    
                    for method in methods_to_test:
                        metrics_list = oos_metrics_all[method]
                        
                        # Extract each metric
                        sharpes = [m['sharpe'] for m in metrics_list if not np.isnan(m['sharpe'])]
                        sortinos = [m['sortino'] for m in metrics_list if not np.isnan(m['sortino']) and not np.isinf(m['sortino'])]
                        calmars = [m['calmar'] for m in metrics_list if not np.isnan(m['calmar']) and not np.isinf(m['calmar'])]
                        max_dds = [m['max_drawdown'] for m in metrics_list if not np.isnan(m['max_drawdown'])]
                        cvars = [m['cvar_95'] for m in metrics_list if not np.isnan(m['cvar_95'])]
                        win_rates = [m['win_rate'] for m in metrics_list if not np.isnan(m['win_rate'])]
                        
                        if len(sharpes) > 0:
                            # Raw data for sorting
                            raw_sharpe = np.median(sharpes)
                            raw_sortino = np.median(sortinos) if sortinos else np.nan
                            raw_calmar = np.median(calmars) if calmars else np.nan
                            
                            ranking_data_raw.append({
                                "method": method,
                                "sharpe": raw_sharpe,
                                "sortino": raw_sortino,
                                "calmar": raw_calmar
                            })
                            
                            # Formatted data for display
                            ranking_data.append({
                                "method": method,
                                "Strategy": method_names.get(method, method),
                                "Sharpe": f"{raw_sharpe:.3f}",
                                "Sortino": f"{raw_sortino:.3f}" if not np.isnan(raw_sortino) else "N/A",
                                "Calmar": f"{raw_calmar:.3f}" if not np.isnan(raw_calmar) else "N/A",
                                "Max DD": f"{np.max(max_dds)*100:.2f}%" if max_dds else "N/A",
                                "CVaR (5%)": f"{np.median(cvars)*100:.2f}%" if cvars else "N/A",
                                "Win Rate": f"{np.mean(win_rates)*100:.1f}%" if win_rates else "N/A"
                            })

                    # Sort by primary metric
                    sort_key = primary_metric
                    ranking_data_raw.sort(key=lambda x: x[sort_key] if not np.isnan(x[sort_key]) else -999, reverse=True)
                    
                    # Reorder display data based on sorted raw data
                    method_order = [item['method'] for item in ranking_data_raw]
                    ranking_data = sorted(ranking_data, key=lambda x: method_order.index(x['method']))

                    # Add rank icons with highlighting for primary metric
                    metric_col_map = {
                        'sharpe': 'Sharpe',
                        'sortino': 'Sortino',
                        'calmar': 'Calmar'
                    }
                    primary_col_name = metric_col_map[primary_metric]
                    
                    for i, row in enumerate(ranking_data):
                        icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
                        row['#'] = f"{icon} {i+1}"
                        
                        # Add star to primary metric value
                        if primary_col_name in row:
                            current_val = row[primary_col_name]
                            if current_val != "N/A":
                                row[primary_col_name] = f"⭐ {current_val}"

                    # Create DataFrame
                    ranking_df = pd.DataFrame(ranking_data)
                    cols = ['#', 'Strategy', 'Sharpe', 'Sortino', 'Calmar', 'Max DD', 'CVaR (5%)', 'Win Rate']
                    ranking_df = ranking_df[cols]

                    # Display table
                    st.markdown(
                        create_styled_table(ranking_df, f"Ranked by {primary_metric_display} (Median OOS) — ⭐ indicates primary metric"),
                        unsafe_allow_html=True
                    )

                    st.caption(f"""
                    **📊 Reading this table:**  
                    • Strategies are sorted by **{primary_metric_display}** (marked with ⭐)  
                    • All values represent **median** performance across {n_valid_splits} out-of-sample tests  
                    • Lower Max DD and CVaR are better (less risk)  
                    • Higher Win Rate is better (more consistent gains)
                    """)

                    st.markdown("---")

                    # ==========================
                    # 3. MULTI-METRIC DISTRIBUTIONS (TERZA - DETTAGLI)
                    # ==========================
                    st.markdown("## 📊 Performance Distributions Across All Splits")
                    
                    st.markdown(f"""
                    These box plots show the **distribution** of out-of-sample performance across all {n_valid_splits} 
                    train/test splits. Wide boxes indicate high variability (less robust), narrow boxes indicate 
                    consistency (more robust).
                    
                    📌 **Note:** While the ranking above uses **{primary_metric_display}**, we show distributions 
                    for all metrics to give you a complete risk profile.
                    """)
                    
                    metric_display = {
                        'sharpe': f"{'⭐ ' if primary_metric == 'sharpe' else ''}Sharpe Ratio",
                        'sortino': f"{'⭐ ' if primary_metric == 'sortino' else ''}Sortino Ratio",
                        'calmar': f"{'⭐ ' if primary_metric == 'calmar' else ''}Calmar Ratio",
                        'max_drawdown': 'Max Drawdown (%)'
                    }
                    
                    # Create tabs for each metric (star on primary)
                    metric_tabs = st.tabs(list(metric_display.values()))
                    
                    for tab_idx, (metric_key, metric_name) in enumerate(metric_display.items()):
                        with metric_tabs[tab_idx]:
                            fig = go.Figure()
                            
                            for i, method in enumerate(methods_to_test):
                                values = [m[metric_key] for m in oos_metrics_all[method] 
                                        if not np.isnan(m[metric_key])]
                                
                                # Handle special case for max_drawdown
                                if metric_key == 'max_drawdown':
                                    values = [v * 100 for v in values]
                                
                                if len(values) > 0:
                                    fig.add_trace(go.Box(
                                        y=values,
                                        name=method_names.get(method, method),
                                        boxmean='sd',
                                        marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                                        hovertemplate='<b>%{fullData.name}</b><br>Value: %{y:.3f}<extra></extra>'
                                    ))
                            
                            y_title = metric_name.replace('⭐ ', '')
                            if metric_key == 'max_drawdown':
                                y_title += ' (lower is better)'
                            
                            fig.update_layout(
                                height=400,
                                yaxis_title=y_title,
                                showlegend=False,
                                title=dict(
                                    text=f"{'📌 Primary Metric for PBO' if metric_key == primary_metric else ''}",
                                    font=dict(size=11, color='#FFE66D'),
                                    x=0.5,
                                    xanchor='center'
                                )
                            )
                            fig = apply_plotly_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add interpretation for each metric
                            if metric_key == primary_metric:
                                st.info(f"⭐ This is the **primary metric** used to calculate PBO ({pbo:.1%})")
                    
                    # Interpretation guide
                    with st.expander("📖 How to Interpret Box Plots"):
                        st.markdown("""
                        ### Understanding Box Plots
                        
                        Each box plot shows the distribution of a metric across all out-of-sample tests:
                        
                        - **Box**: Contains 50% of results (25th to 75th percentile)
                        - **Line inside box**: Median value (the ranking table uses this)
                        - **Whiskers**: Extend to min/max (excluding outliers)
                        - **Diamond**: Mean value
                        - **Dots**: Outliers (if any)
                        
                        ### What to Look For
                        
                        **Narrow boxes** = Consistent performance (good!)  
                        **Wide boxes** = High variability across market regimes (risky!)  
                        **High median** = Good average performance  
                        **No extreme outliers** = Strategy doesn't break in edge cases
                        
                        ### Ideal Strategy
                        
                        Look for strategies with:
                        1. High median (good average performance)
                        2. Narrow box (consistency across regimes)
                        3. Few/no negative outliers (robust in worst cases)
                        """)

                    st.markdown("---")

                    # ==========================
                    # 4. STRATEGY-SPECIFIC ANALYSIS (QUARTA - CONFRONTO)
                    # ==========================
                    st.markdown("## 🔍 Best vs Worst: Deep Comparison")
                    
                    if len(ranking_data_raw) > 0:
                        best_method = ranking_data_raw[0]['method']
                        best_name = method_names.get(best_method, best_method)
                        
                        # Find formatted data for best strategy
                        best_data = next((item for item in ranking_data if item['method'] == best_method), None)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"#### 🥇 Top Performer: {best_name}")
                            
                            if best_data:
                                top_strategy_data = {
                                    'Metric': ['Sharpe', 'Sortino', 'Calmar', 'Max DD', 'CVaR (5%)', 'Win Rate'],
                                    'Value': [
                                        best_data['Sharpe'].replace('⭐ ', ''),
                                        best_data['Sortino'].replace('⭐ ', ''),
                                        best_data['Calmar'].replace('⭐ ', ''),
                                        best_data['Max DD'],
                                        best_data['CVaR (5%)'],
                                        best_data['Win Rate']
                                    ]
                                }
                                st.markdown(create_styled_table(pd.DataFrame(top_strategy_data)), unsafe_allow_html=True)
                        
                        with col2:
                            if len(ranking_data_raw) > 1:
                                worst_method = ranking_data_raw[-1]['method']
                                worst_name = method_names.get(worst_method, worst_method)
                                
                                st.markdown(f"#### 📊 Bottom Performer: {worst_name}")
                                
                                best_sortino = ranking_data_raw[0]['sortino']
                                worst_sortino = ranking_data_raw[-1]['sortino']
                                
                                # Get max_dd from original data
                                best_max_dd = None
                                worst_max_dd = None
                                
                                for method in methods_to_test:
                                    metrics_list = oos_metrics_all[method]
                                    max_dds = [m['max_drawdown'] for m in metrics_list if not np.isnan(m['max_drawdown'])]
                                    
                                    if method == best_method and max_dds:
                                        best_max_dd = np.max(max_dds) * 100
                                    elif method == worst_method and max_dds:
                                        worst_max_dd = np.max(max_dds) * 100
                                
                                if not np.isnan(best_sortino) and not np.isnan(worst_sortino) and best_max_dd is not None and worst_max_dd is not None:
                                    sortino_diff = best_sortino - worst_sortino
                                    dd_diff = worst_max_dd - best_max_dd
                                    
                                    comparison_data = {
                                        'Advantage': ['Sortino', 'Drawdown'],
                                        'Difference': [
                                            f"+{sortino_diff:.2f} ({sortino_diff/worst_sortino*100:+.1f}%)" if worst_sortino != 0 else f"+{sortino_diff:.2f}",
                                            f"{dd_diff:.1f}pp better" if dd_diff > 0 else f"{abs(dd_diff):.1f}pp worse"
                                        ]
                                    }
                                    st.markdown(create_styled_table(pd.DataFrame(comparison_data)), unsafe_allow_html=True)
                                    
                                    # Verdict
                                    st.markdown("---")
                                    if sortino_diff > 0.3 and dd_diff > 10:
                                        st.success("✅ **Clear winner:** Superior risk-adjusted returns with significantly lower tail risk")
                                    elif sortino_diff > 0.3:
                                        st.info("📊 **Better returns:** Superior risk-adjusted performance, similar drawdown profile")
                                    elif dd_diff > 10:
                                        st.info("📊 **Lower risk:** Significantly lower tail risk, similar return profile")
                                    else:
                                        st.warning("⚠️ **Marginal difference:** Performance profiles are quite similar")
                    
                    st.markdown("---")

                    # ==========================
                    # METRIC INTERPRETATION GUIDE (SEMPRE DISPONIBILE)
                    # ==========================
                    with st.expander("📖 Understanding the Metrics"):
                        st.markdown("""
                        ### Risk-Adjusted Performance Metrics
                        
                        **Sharpe Ratio**
                        - Measures return per unit of total volatility
                        - Higher is better (> 1.0 is good, > 2.0 is excellent)
                        - ⚠️ Limitation: Penalizes upside volatility equally with downside
                        
                        **Sortino Ratio** ⭐ Recommended 
                        - Measures return per unit of downside volatility only
                        - Higher is better
                        - ✅ Advantage: Only penalizes losses, not gains
                        - Better for strategies with asymmetric returns
                        **Calmar Ratio** ⭐ *Recommended*
                        - Annual return divided by maximum drawdown
                        - Higher is better
                        - ✅ Advantage: Directly measures worst-case scenario risk
                        - Intuitive for investors: "return per unit of pain"
                    
                        **Maximum Drawdown**
                        - Largest peak-to-trough decline
                        - Lower is better (more negative = worse)
                        - Critical for psychological sustainability
                        - Hedge funds typically target < 20%
                    
                        **CVaR (Conditional Value at Risk)**
                        - Average of the worst 5% of returns
                        - More negative = higher tail risk
                        - Captures "black swan" events better than volatility
                        - More conservative than VaR
                    
                        **Win Rate**
                        - Percentage of positive return periods
                        - Higher is better
                        - ⚠️ Not sufficient alone: high win rate with big losses can still lose money
                        - Best used in combination with other metrics
                    
                        ---
                        
                        ### Best Practice: Multi-Metric Approach
                        
                        Don't rely on a single metric. Look for strategies that show:
                        
                        ✅ **High Sortino/Calmar** = Robust risk-adjusted returns  
                        ✅ **Low Max Drawdown** = Psychologically sustainable  
                        ✅ **Moderate CVaR** = Controlled tail risk  
                        ✅ **Decent Win Rate** = Consistent gains  
                        
                        A strategy that ranks #1 on one metric but #5 on others might not be your best choice.
                        """)

                    st.markdown("---")

                    # ==========================
                    # 5. FINAL TAKEAWAY (QUINTA - RACCOMANDAZIONI)
                    # ==========================
                    st.markdown("## 🎯 Final Assessment & Recommendations")

                    # Overall assessment based on PBO
                    if not np.isnan(pbo):
                        if pbo < 0.1:
                            assessment_icon = "🟢"
                            assessment_text = "ROBUST"
                            assessment_detail = "Low overfitting risk. Your strategies show consistent out-of-sample performance."
                            assessment_color = "#4ECDC4"
                        elif pbo < 0.3:
                            assessment_icon = "🟡"
                            assessment_text = "ACCEPTABLE"
                            assessment_detail = "Moderate reliability. Results are meaningful but monitor live performance carefully."
                            assessment_color = "#FFE66D"
                        elif pbo < 0.5:
                            assessment_icon = "🟠"
                            assessment_text = "CAUTION"
                            assessment_detail = "High overfitting risk. Consider simpler strategies or ensemble approaches."
                            assessment_color = "#FF9F43"
                        else:
                            assessment_icon = "🔴"
                            assessment_text = "WARNING"
                            assessment_detail = "Severe overfitting detected. Strategy selection is unreliable."
                            assessment_color = "#FF6B6B"
                    else:
                        assessment_icon = "⚠️"
                        assessment_text = "INSUFFICIENT DATA"
                        assessment_detail = "Unable to assess overfitting risk due to insufficient valid splits."
                        assessment_color = "#94a3b8"

                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {assessment_color}33, {assessment_color}11); 
                                border: 2px solid {assessment_color}; padding: 24px; border-radius: 16px; margin: 20px 0;'>
                        <div style='text-align: center;'>
                            <span style='font-size: 3em;'>{assessment_icon}</span>
                            <h2 style='color: {assessment_color}; margin: 12px 0;'>{assessment_text}</h2>
                            <p style='color: #E2E8F0; font-size: 1.1em; margin: 0;'>{assessment_detail}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Key insights summary
                    st.markdown("### 📌 Key Findings")
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    with insight_col1:
                        st.metric("Primary Metric Used", primary_metric_display)
                    
                    with insight_col2:
                        if not np.isnan(pbo):
                            pbo_color_metric = "normal" if pbo < 0.3 else "inverse"
                            st.metric(
                                "PBO Score", 
                                f"{pbo:.1%}",
                                delta="Robust" if pbo < 0.3 else "Overfitting Risk",
                                delta_color=pbo_color_metric
                            )
                        else:
                            st.metric("PBO Score", "N/A")
                    
                    with insight_col3:
                        if len(ranking_data) > 0:
                            winner_name = ranking_data[0]['Strategy']
                            st.metric("Best Strategy (OOS)", winner_name if len(winner_name) <= 20 else winner_name[:17] + "...")
                    
                    st.markdown("---")
                    
                    # Actionable recommendations
                    st.markdown("### 💡 What Should You Do?")
                    
                    rec_col1, rec_col2 = st.columns(2)
                    
                    with rec_col1:
                        st.success("### ✅ Green Flags")
                        st.markdown("""
                        Look for strategies with:
                        
                        - **High Sortino/Calmar** ratios (> 1.0)
                        - **Max DD < 40%** for sustainability
                        - **Low CVaR** for controlled tail risk
                        - **Narrow distribution boxes** (consistency)
                        - **PBO < 30%** for the overall ensemble
                        """)
                    
                    with rec_col2:
                        st.error("### 🚨 Red Flags")
                        st.markdown("""
                        Be cautious of:
                        
                        - **PBO > 30%** indicates overfitting
                        - **Wide distribution boxes** (regime-dependent)
                        - **Extreme outliers** in box plots
                        - **Very high Sharpe (> 3.0)** might be luck
                        - **Strategy works only in 1-2 regimes**
                        """)
                    
                    st.markdown("---")
                    
                    # Strategic recommendations based on results
                    st.markdown("### 🎓 Strategic Recommendations")
                    
                    st.info("""
                    **Remember These Key Principles:**
                    
                    1. **Out-of-sample > In-sample:** The metrics here are closer to real-world performance than the main analysis
                    
                    2. **Robustness > Performance:** A strategy with 0.8 Sharpe but consistent across all splits is better than one with 1.5 Sharpe in only half the splits
                    
                    3. **Multiple metrics matter:** Don't pick based on a single metric. Consider the complete risk-return profile
                    
                    4. **Simpler is often better:** If Equal Weight or Risk Parity rank high, they're likely more robust than complex optimizations
                    
                    5. **Ensemble if uncertain:** If multiple strategies show robustness (low PBO), consider allocating across them rather than picking one
                    
                    ---
                    
                    **Next Steps:**
                    
                    - If PBO < 30%: Proceed with confidence, but start with smaller position sizes
                    - If PBO > 30%: Consider simpler strategies or wait for more data
                    - Monitor live performance: Even the best backtest can't predict the future perfectly
                    """)
                    
                    st.markdown("---")
                    
                    # Academic reference
                    st.caption("""
                    📚 **Methodology:** This analysis implements Combinatorial Purged Cross-Validation 
                    from López de Prado (2018) "Advances in Financial Machine Learning", Chapter 7, 
                    combined with the Probability of Backtest Overfitting framework from Bailey & López de Prado (2014).
                    """)
# Part 8: Frontier, Benchmark, Export, Footer

        # TAB 6: FRONTIER
        with tab6:
            st.markdown("### 📐 Efficient Frontier")
            
            st.markdown("""
            The **Efficient Frontier** represents the set of optimal portfolios that offer the highest 
            expected return for a given level of risk (or the lowest risk for a given return level).
            
            This visualization helps you understand where your optimized strategies stand relative 
            to the universe of all possible portfolio combinations.
            """)
            
            # ================================================================
            # FRONTIER TYPE SELECTOR
            # ================================================================
            frontier_type = st.radio(
                "Select Frontier Type",
                ["📊 Mean-Variance (Traditional)", "📉 Mean-CVaR (Tail Risk)"],
                horizontal=True,
                key="frontier_type_selector"
            )
            
            # ================================================================
            # UNIFIED EDUCATIONAL SECTION
            # ================================================================
            with st.expander("📚 Understanding the Efficient Frontier", expanded=False):
                st.markdown("""
                ### What is the Efficient Frontier?
                
                The Efficient Frontier is a cornerstone concept of **Modern Portfolio Theory (MPT)**, 
                introduced by Harry Markowitz in 1952. It answers a fundamental question:
                
                > *"Given a set of assets, what is the best possible combination of risk and return I can achieve?"*
                
                ---
                
                ### Key Concepts
                
                **1. The Risk-Return Trade-off**
                
                In finance, higher returns typically come with higher risk. The Efficient Frontier 
                shows the **optimal trade-off**: portfolios on the frontier give you the maximum 
                return for each level of risk you're willing to accept.
                
                **2. Diversification Benefit**
                
                The "bullet" shape of the portfolio cloud demonstrates the power of diversification:
                - By combining assets that don't move perfectly together (correlation < 1), 
                you can achieve **lower portfolio volatility** than holding any single asset
                - The leftmost point of the frontier represents the **Global Minimum Variance Portfolio**
                
                **3. Dominated vs Efficient Portfolios**
                
                - **Efficient portfolios** (on the frontier): No other portfolio offers higher return 
                for the same risk, or lower risk for the same return
                - **Dominated portfolios** (inside the cloud): Suboptimal—you could do better by 
                moving to the frontier
                
                ---
                
                ### Two Types of Efficient Frontiers
                
                #### 📊 Mean-Variance Frontier (Markowitz, 1952)
                
                The traditional efficient frontier uses **standard deviation (volatility)** as the risk measure.
                
                **Optimization problem:**
                $$\\min_w \\sigma_p = \\sqrt{w^T \\Sigma w}$$
                $$\\text{subject to: } w^T \\mu = \\mu_{target}, \\quad \\sum w_i = 1$$
                
                **Advantages:**
                - Well-established theory with decades of academic research
                - Computationally efficient (quadratic programming)
                - Intuitive interpretation
                - Captures overall portfolio variability
                
                **Limitations:**
                - Assumes returns are normally distributed
                - Penalizes upside and downside volatility equally
                - Underestimates tail risk and extreme events
                - Does not distinguish between good volatility (gains) and bad volatility (losses)
                
                ---
                
                #### 📉 Mean-CVaR Frontier (Rockafellar & Uryasev, 2000)
                
                This frontier uses **Conditional Value-at-Risk (CVaR)** as the risk measure.
                
                **What is CVaR?**
                $$\\text{CVaR}_{\\alpha} = \\mathbb{E}[R | R \\leq \\text{VaR}_{\\alpha}]$$
                
                CVaR₉₅ answers: *"What is the average return in the worst 5% of scenarios?"*
                
                **Optimization problem:**
                $$\\min_w \\text{CVaR}_{\\alpha}(w)$$
                $$\\text{subject to: } w^T \\mu = \\mu_{target}, \\quad \\sum w_i = 1$$
                
                **Advantages:**
                - Coherent risk measure (unlike VaR)
                - Focuses specifically on tail risk (what really hurts portfolios)
                - Better for non-normal distributions (captures fat tails and skewness)
                - Captures "black swan" events more effectively
                - Aligns with investor psychology (we fear large losses more than we enjoy equivalent gains)
                
                **Key Differences from Mean-Variance:**
                - Portfolios on the CVaR frontier may differ significantly from the Variance frontier
                - Often more conservative in extreme tail scenarios
                - May suggest different asset allocations, especially for assets with asymmetric returns
                - The shape of the CVaR frontier is typically more linear due to CVaR being a coherent risk measure
                
                **Important Note on CVaR:**
                CVaR has an additional estimation challenge: it's calculated from only the **worst α% of observations**. 
                With 1000 days of data and α=0.95, that's only ~50 data points—leading to higher estimation error 
                than volatility-based measures.
                
                ---
                
                ### How to Read the Chart
                
                | Element | What it represents |
                |---------|-------------------|
                | **Cloud of dots** | Random portfolio combinations (the "feasible region") |
                | **Upper-left boundary** | The Efficient Frontier itself |
                | **Colored markers** | Your optimized portfolio strategies |
                | **X-axis** | Risk measure (Volatility for Mean-Variance, CVaR for Mean-CVaR) |
                | **Y-axis** | Expected annualized return |
                
                ---
                
                ### Interpreting Your Strategies
                
                - **On the frontier**: Your optimization is working well—the strategy is efficient
                - **Below the frontier**: The strategy may have constraints (e.g., transaction costs, 
                rebalancing rules, estimation error) that prevent it from reaching theoretical optimality
                - **Similar positions**: Multiple strategies clustering together suggests they 
                produce similar risk-return profiles despite different methodologies
                
                ---
                
                ### Limitations to Keep in Mind
                
                1. **Based on historical data**: Past correlations and returns may not persist
                2. **Estimation error**: Both expected returns and risk measures are estimated with noise
                3. **Static view**: The frontier shifts as market conditions change
                4. **Transaction costs**: Theoretical frontier doesn't account for trading frictions
                5. **Model risk**: Both approaches make simplifying assumptions about return distributions
                
                ---
                
                📖 **Key References**: 
                - Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77-91.
                - Rockafellar & Uryasev (2000). "Optimization of CVaR." *Journal of Risk*
                - Krokhmal et al. (2002). "Portfolio Optimization with CVaR." *Journal of Risk*
                """)
            
            st.markdown("---")
            
            # ================================================================
            # CONFIGURATION
            # ================================================================
            col1, col2, col3 = st.columns(3)
            with col1:
                n_portfolios = st.slider(
                    "Number of random portfolios", 
                    2000, 20000, 10000, 
                    step=1000, 
                    key="n_port",
                    help="More portfolios = smoother frontier visualization"
                )
            with col2:
                allow_short = st.checkbox(
                    "Allow short selling", 
                    value=False, 
                    key="allow_short",
                    help="If enabled, portfolio weights can be negative (betting against assets)"
                )
            with col3:
                if frontier_type == "📉 Mean-CVaR (Tail Risk)":
                    cvar_alpha = st.selectbox(
                        "CVaR Confidence Level",
                        options=[0.90, 0.95, 0.99],
                        index=1,
                        format_func=lambda x: f"{x:.0%} (worst {(1-x)*100:.0f}%)",
                        key="cvar_alpha"
                    )
                else:
                    cvar_alpha = 0.95  # Default, not used for Mean-Variance
            
            # ================================================================
            # GENERATE PORTFOLIOS
            # ================================================================
            with st.spinner(f"Generating {n_portfolios:,} portfolios..."):
                
                # ===== SETUP =====
                returns_aligned = analyzer.returns.reindex(columns=analyzer.symbols)
                returns_matrix = returns_aligned.values
                n_assets = len(analyzer.symbols)
                
                # Annualized covariance and returns
                cov_matrix = returns_aligned.cov() * 252
                mean_returns = returns_aligned.mean() * 252
                
                # ===== GENERATE RANDOM PORTFOLIOS =====
                np.random.seed(42)
                
                portfolio_returns = []
                portfolio_risk_list = []  # Will contain volatility OR CVaR depending on frontier type
                portfolio_volatilities = []  # Always store volatility for Sharpe calculation
                portfolio_sharpes = []
                portfolio_weights = []
                
                dirichlet_alpha = 2.0
                
                for _ in range(n_portfolios):
                    if allow_short:
                        w = np.random.normal(1, 0.3, n_assets)
                        w = w / np.sum(w)
                    else:
                        w = np.random.dirichlet(np.ones(n_assets) * dirichlet_alpha)
                    
                    # Calculate return (same for both frontiers)
                    port_return = np.dot(mean_returns, w)
                    
                    # Calculate volatility (always needed for Sharpe)
                    port_variance = np.dot(w, np.dot(cov_matrix, w))
                    port_volatility = np.sqrt(port_variance)
                    
                    # Calculate risk metric based on frontier type
                    if frontier_type == "📊 Mean-Variance (Traditional)":
                        port_risk = port_volatility  # Annualized volatility
                    else:
                        # CVaR calculation
                        daily_portfolio_returns = returns_matrix @ w
                        var_threshold = np.percentile(daily_portfolio_returns, 100 * (1 - cvar_alpha))
                        tail_returns = daily_portfolio_returns[daily_portfolio_returns <= var_threshold]
                        if len(tail_returns) > 0:
                            port_risk = -tail_returns.mean()  # Daily CVaR (positive = tail loss)
                        else:
                            port_risk = -var_threshold
                    
                    # Sharpe ratio (always using volatility)
                    port_sharpe = (port_return - rf_rate) / port_volatility if port_volatility > 0 else 0
                    
                    portfolio_returns.append(port_return * 100)
                    portfolio_risk_list.append(port_risk * 100)
                    portfolio_volatilities.append(port_volatility * 100)
                    portfolio_sharpes.append(port_sharpe)
                    portfolio_weights.append(w)
                
                # ===== YOUR OPTIMIZED PORTFOLIOS =====
                strategy_returns = []
                strategy_risk = []
                strategy_volatilities = []  # Always store volatility
                strategy_names = []
                strategy_sharpes = []
                
                for p_name, p in analyzer.portfolios.items():
                    strat_return = p['annualized_return'] * 100
                    strat_volatility = p['annualized_volatility'] * 100
                    strat_sharpe = p['sharpe_ratio']
                    
                    strategy_returns.append(strat_return)
                    strategy_volatilities.append(strat_volatility)
                    strategy_names.append(p['name'])
                    strategy_sharpes.append(strat_sharpe)
                    
                    # Risk metric based on frontier type
                    if frontier_type == "📊 Mean-Variance (Traditional)":
                        strat_risk = strat_volatility
                    else:
                        # Calculate CVaR for this portfolio
                        port_daily_returns = p['returns'].values
                        var_threshold = np.percentile(port_daily_returns, 100 * (1 - cvar_alpha))
                        tail = port_daily_returns[port_daily_returns <= var_threshold]
                        if len(tail) > 0:
                            strat_cvar = -tail.mean() * 100  # Daily CVaR %
                        else:
                            strat_cvar = -var_threshold * 100
                        strat_risk = strat_cvar
                    
                    strategy_risk.append(strat_risk)
                
                # ===== CREATE CHART =====
                fig = go.Figure()
                
                # Labels based on frontier type
                if frontier_type == "📊 Mean-Variance (Traditional)":
                    risk_label = "Annualized Volatility (Standard Deviation) %"
                    cloud_color = 'rgba(255, 107, 107, 0.4)'
                else:
                    risk_label = f"Daily CVaR{int(cvar_alpha*100)} (Tail Loss) %"
                    cloud_color = 'rgba(147, 112, 219, 0.4)'
                
                # 1. Random portfolios (cloud)
                fig.add_trace(go.Scatter(
                    x=portfolio_risk_list,
                    y=portfolio_returns,
                    mode='markers',
                    name=f'Random Portfolios ({n_portfolios:,})',
                    marker=dict(
                        size=3,
                        color=cloud_color,
                        symbol='circle'
                    ),
                    hovertemplate=f'Return: %{{y:.2f}}%<br>{risk_label.split(" %")[0]}: %{{x:.2f}}%<extra></extra>'
                ))
                
                # 2. Your optimized portfolios (colored markers)
                portfolio_colors = ['#FFE66D', '#A855F7', '#6366F1', '#FF9F43', '#EC4899', '#10B981', '#F59E0B']
                portfolio_symbols = ['star', 'diamond', 'hexagon', 'pentagon', 'circle', 'square', 'triangle-up']
                
                for i, (x_val, y_val, name, sharpe) in enumerate(zip(strategy_risk, strategy_returns, strategy_names, strategy_sharpes)):
                    fig.add_trace(go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode='markers',
                        name=name,
                        marker=dict(
                            size=18,
                            color=portfolio_colors[i % len(portfolio_colors)],
                            symbol=portfolio_symbols[i % len(portfolio_symbols)],
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=f'<b>{name}</b><br>Return: %{{y:.2f}}%<br>{risk_label.split(" %")[0]}: %{{x:.2f}}%<br>Sharpe: {sharpe:.3f}<extra></extra>'
                    ))
                
                # Layout
                fig.update_layout(
                    height=650,
                    xaxis_title=risk_label,
                    yaxis_title="Annualized Return %",
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.02,
                        bgcolor='rgba(26,26,36,0.95)',
                        bordercolor='rgba(99,102,241,0.5)',
                        borderwidth=1,
                        font=dict(size=10)
                    ),
                    hovermode='closest',
                    margin=dict(r=200)
                )
                
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                # ===== QUICK STATS =====
                st.markdown("---")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎲 Simulated Portfolios", f"{n_portfolios:,}")
                with col2:
                    best_sharpe_idx = np.argmax(portfolio_sharpes)
                    st.metric("⭐ Best Random Sharpe", f"{portfolio_sharpes[best_sharpe_idx]:.3f}")
                with col3:
                    min_risk_idx = np.argmin(portfolio_risk_list)
                    risk_type = "Volatility" if frontier_type == "📊 Mean-Variance (Traditional)" else "CVaR"
                    st.metric(f"📉 Min {risk_type}", f"{portfolio_risk_list[min_risk_idx]:.2f}%")
                with col4:
                    max_ret_idx = np.argmax(portfolio_returns)
                    st.metric("📈 Max Return Found", f"{portfolio_returns[max_ret_idx]:.2f}%")
                
                # ===== WHY SOME STRATEGIES ARE INSIDE THE CLOUD =====
                st.markdown("---")
        
                with st.expander("📚 Why Are Some Strategies Inside the Cloud?", expanded=False):
                    st.markdown("""
                    You might wonder: *"If I optimized these portfolios, why aren't they all on the efficient frontier?"*
                    
                    This is one of the most important insights in modern portfolio theory.
                    
                    ---
                    
                    ### The Efficient Frontier is "Optimal" Only in Hindsight
                    
                    The frontier you see is calculated using **historical data**. It shows the best you 
                    *could have done* if you had known the future perfectly. But when these strategies 
                    were "optimized," they only had access to past data—just like in real investing.
                    
                    ---
                    
                    ### The Problem: Estimation Error
                    
                    Markowitz optimization requires estimating **expected returns** and **covariances**. 
                    These estimates are extremely noisy:
                    
                    > Merton (1980) showed that estimating expected returns with the same precision as 
                    > volatility would require **~500 years of data**.
                    
                    Worse, mean-variance optimization **amplifies these errors**—it overweights assets 
                    with overestimated returns and underweights those with underestimated returns.
                    
                    > Michaud (1989) called this phenomenon **"error maximization."**
                    
                    ---
                    
                    ### The Surprising Truth: Simple Often Beats "Optimal"
                    
                    In a landmark study, **DeMiguel, Garlappi & Uppal (2009)** compared 14 optimization 
                    strategies against the naive 1/N (Equal Weight) portfolio across 7 datasets.
                    
                    **Result:** No optimized strategy consistently beat Equal Weight out-of-sample!
                    
                    They estimated that **~250 years of data** would be needed for mean-variance 
                    optimization to reliably outperform 1/N.
                    
                    ---
                    
                    ### An Additional Problem: Volatility (Standard Deviation) Isn't "Risk"
                    
                    The entire framework assumes returns are **normally distributed**. In reality, 
                    financial returns exhibit:
                    
                    - **Fat tails**: Extreme events happen far more often than the normal distribution predicts
                    - **Skewness**: Crashes are sharper than rallies
                    - **Volatility clustering**: Calm and turbulent periods cluster together
                    
                    > Mandelbrot (1963) and Cont (2001) documented these "stylized facts" that violate 
                    > the normality assumption underlying mean-variance optimization.
                    
                    Volatility also **penalizes gains equally with losses**—but investors don't mind 
                    upside volatility! This is why metrics like **Sortino** and **CVaR** are often 
                    more meaningful than Sharpe.
                    
                    ---
                    
                    ### For Mean-CVaR Specifically
                    
                    The same estimation issues apply to CVaR, with an additional challenge: CVaR is 
                    estimated from only the **worst α% of observations**. With 1000 days of data and 
                    α=0.95, that's only ~50 data points for CVaR estimation—leading to high estimation error!
                    
                    ---
                    
                    ### The Bottom Line
                    
                    > **The efficient frontier shows the best you could have done with perfect foresight. 
                    > Strategies like Equal Weight, Risk Parity, and HRP accept being "suboptimal" 
                    > in-sample to be more robust when the future is uncertain—which is always.**
                    
                    ---
                    
                    #### 📖 Key References
                    
                    - Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*
                    - Merton, R.C. (1980). "On Estimating the Expected Return on the Market." *JFE*
                    - Michaud, R.O. (1989). "The Markowitz Optimization Enigma." *FAJ*
                    - DeMiguel, V. et al. (2009). "Optimal Versus Naive Diversification." *RFS*
                    - López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample." *JPM*
                    - Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices." *Journal of Business*
                    - Cont, R. (2001). "Empirical Properties of Asset Returns." *Quantitative Finance*
                    """)
                
                st.markdown("---")
                
                # ================================================================
                # FRONTIER EXPLORER
                # ================================================================
                st.markdown("#### 🔍 Explore the Frontier")
                
                st.markdown("""
                Use the slider below to explore portfolios along the efficient frontier. 
                Select different risk levels and compare with your optimized strategies.
                """)
                
                # Find approximate frontier by bucketing risk and finding max return
                n_buckets = 50
                risk_min, risk_max = min(portfolio_risk_list), max(portfolio_risk_list)
                bucket_size = (risk_max - risk_min) / n_buckets
                
                frontier_points = []
                for i in range(n_buckets):
                    bucket_start = risk_min + i * bucket_size
                    bucket_end = bucket_start + bucket_size
                    
                    bucket_indices = [
                        j for j, v in enumerate(portfolio_risk_list) 
                        if bucket_start <= v < bucket_end
                    ]
                    
                    if bucket_indices:
                        best_idx = max(bucket_indices, key=lambda j: portfolio_returns[j])
                        frontier_points.append({
                            'risk': portfolio_risk_list[best_idx],
                            'return': portfolio_returns[best_idx],
                            'volatility': portfolio_volatilities[best_idx],  # Always store volatility
                            'sharpe': portfolio_sharpes[best_idx],
                            'weights': portfolio_weights[best_idx]
                        })
                
                if frontier_points:
                    # Controls row
                    ctrl_col1, ctrl_col2 = st.columns([2, 1])
                    
                    with ctrl_col1:
                        frontier_position = st.slider(
                            "Risk Level (0% = Min Risk, 100% = Max Risk)",
                            min_value=0,
                            max_value=100,
                            value=30,
                            step=5,
                            key="frontier_slider"
                        )
                    
                    with ctrl_col2:
                        compare_strategy = st.selectbox(
                            "Compare with strategy",
                            options=list(analyzer.portfolios.keys()),
                            format_func=lambda x: analyzer.portfolios[x]['name'],
                            key="compare_strategy"
                        )
                    
                    frontier_idx = int((frontier_position / 100) * (len(frontier_points) - 1))
                    selected = frontier_points[frontier_idx]
                    
                    # ===== METRICS AND WEIGHTS =====
                    col1, col2 = st.columns([1, 1.5])
                    
                    with col1:
                        st.markdown("##### 📊 Selected Frontier Portfolio")
                        
                        # Risk profile indicator
                        if frontier_position < 25:
                            risk_profile = "🛡️ Conservative"
                            risk_color = "#4ECDC4"
                        elif frontier_position < 50:
                            risk_profile = "⚖️ Moderate"
                            risk_color = "#FFE66D"
                        elif frontier_position < 75:
                            risk_profile = "📈 Growth"
                            risk_color = "#FF9F43"
                        else:
                            risk_profile = "🚀 Aggressive"
                            risk_color = "#FF6B6B"
                        
                        st.markdown(f"**Profile:** <span style='color:{risk_color}; font-size:1.1em;'>{risk_profile}</span>", unsafe_allow_html=True)
                        
                        mcols = st.columns(2)
                        mcols[0].metric("Return", f"{selected['return']:.2f}%")
                        mcols[1].metric("Volatility", f"{selected['volatility']:.2f}%")
                        
                        mcols2 = st.columns(2)
                        mcols2[0].metric("Sharpe", f"{selected['sharpe']:.3f}")
                        mcols2[1].metric("Risk Level", f"{frontier_position}%")
                    
                    with col2:
                        st.markdown("##### ⚖️ Asset Allocation")
                        
                        weights_data = []
                        for ticker, weight in zip(analyzer.symbols, selected['weights']):
                            if abs(weight) > 0.01:
                                weights_data.append({
                                    'Asset': get_display_name(ticker),
                                    'Ticker': ticker,
                                    'Weight': weight * 100
                                })
                        
                        if weights_data:
                            weights_df = pd.DataFrame(weights_data).sort_values('Weight', ascending=False)
                            
                            # Dynamic y-axis range based on data
                            max_weight = weights_df['Weight'].max()
                            min_weight = weights_df['Weight'].min()
                            y_max = max(max_weight * 1.15, 10)
                            y_min = min(min_weight * 1.15, 0) if min_weight < 0 else 0
                            
                            colors = ['#4ECDC4' if w >= 0 else '#FF6B6B' for w in weights_df['Weight']]
                            
                            fig_w = go.Figure(data=[go.Bar(
                                x=weights_df['Asset'],
                                y=weights_df['Weight'],
                                marker_color=colors,
                                text=[f"{w:.1f}%" for w in weights_df['Weight']],
                                textposition='outside',
                                textfont=dict(color='#E2E8F0', size=10)
                            )])
                            
                            fig_w.update_layout(
                                height=280,
                                yaxis_title="Weight (%)",
                                yaxis=dict(range=[y_min, y_max]),
                                xaxis_title="",
                                showlegend=False,
                                margin=dict(t=20, b=30, l=50, r=20)
                            )
                            fig_w.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
                            fig_w = apply_plotly_theme(fig_w)
                            st.plotly_chart(fig_w, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # ================================================================
                    # HISTORICAL PERFORMANCE COMPARISON
                    # ================================================================
                    st.markdown("##### 📈 Historical Performance Comparison")
                    
                    st.markdown(f"""
                    Compare the **selected frontier portfolio** (risk level {frontier_position}%) 
                    with **{analyzer.portfolios[compare_strategy]['name']}** over the analysis period.
                    """)
                    
                    # Calculate cumulative returns for frontier portfolio
                    frontier_weights = selected['weights']
                    frontier_daily_returns = returns_aligned.dot(frontier_weights)
                    frontier_cumulative = (1 + frontier_daily_returns).cumprod() * 100
                    
                    # Get comparison strategy cumulative returns
                    compare_portfolio = analyzer.portfolios[compare_strategy]
                    compare_cumulative = (1 + compare_portfolio['returns']).cumprod() * 100
                    
                    # Create performance chart
                    fig_perf = go.Figure()
                    
                    # Frontier portfolio
                    fig_perf.add_trace(go.Scatter(
                        x=frontier_cumulative.index,
                        y=frontier_cumulative.values,
                        mode='lines',
                        name=f'Frontier Portfolio ({frontier_position}% risk)',
                        line=dict(color=risk_color, width=2.5),
                        hovertemplate='<b>Frontier Portfolio</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Comparison strategy
                    strategy_color = portfolio_colors[list(analyzer.portfolios.keys()).index(compare_strategy) % len(portfolio_colors)]
                    fig_perf.add_trace(go.Scatter(
                        x=compare_cumulative.index,
                        y=compare_cumulative.values,
                        mode='lines',
                        name=compare_portfolio['name'],
                        line=dict(color=strategy_color, width=2.5, dash='dash'),
                        hovertemplate=f'<b>{compare_portfolio["name"]}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ))
                    
                    # Base line
                    fig_perf.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                                    annotation_text="Initial Investment", annotation_position="right")
                    
                    fig_perf.update_layout(
                        height=400,
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value (Base 100)",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        hovermode='x unified'
                    )
                    
                    fig_perf = apply_plotly_theme(fig_perf)
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    # Performance comparison metrics
                    st.markdown("##### 📊 Performance Summary")
                    
                    # Calculate metrics for frontier portfolio
                    frontier_ann_return = frontier_daily_returns.mean() * 252
                    frontier_ann_vol = frontier_daily_returns.std() * np.sqrt(252)
                    frontier_sharpe_calc = (frontier_ann_return - rf_rate) / frontier_ann_vol if frontier_ann_vol > 0 else 0
                    frontier_cum_return = (frontier_cumulative.iloc[-1] / 100) - 1
                    
                    # Drawdown for frontier
                    frontier_rolling_max = frontier_cumulative.expanding().max()
                    frontier_drawdown = (frontier_cumulative - frontier_rolling_max) / frontier_rolling_max
                    frontier_max_dd = frontier_drawdown.min()
                    
                    # Comparison metrics
                    compare_cum_return = compare_portfolio['cumulative_return']
                    compare_max_dd = compare_portfolio['max_drawdown']
                    
                    comparison_data = {
                        'Metric': ['Cumulative Return', 'Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                        f'Frontier ({frontier_position}%)': [
                            f"{frontier_cum_return*100:.2f}%",
                            f"{frontier_ann_return*100:.2f}%",
                            f"{frontier_ann_vol*100:.2f}%",
                            f"{frontier_sharpe_calc:.3f}",
                            f"{frontier_max_dd*100:.2f}%"
                        ],
                        compare_portfolio['name']: [
                            f"{compare_cum_return*100:.2f}%",
                            f"{compare_portfolio['annualized_return']*100:.2f}%",
                            f"{compare_portfolio['annualized_volatility']*100:.2f}%",
                            f"{compare_portfolio['sharpe_ratio']:.3f}",
                            f"{compare_max_dd*100:.2f}%"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.markdown(
                        create_styled_table(comparison_df, "Side-by-Side Comparison"),
                        unsafe_allow_html=True
                    )
                    
                    # ================================================================
                    # QUICK INSIGHT
                    # ================================================================
                    frontier_better = frontier_sharpe_calc > compare_portfolio['sharpe_ratio']
                    sharpe_diff = frontier_sharpe_calc - compare_portfolio['sharpe_ratio']
                    
                    st.markdown("---")
                    
                    if frontier_better:
                        st.success(
                            f"✅ **Frontier portfolio dominates on a risk-adjusted basis** "
                            f"(Sharpe +{sharpe_diff:.3f} vs {compare_portfolio['sharpe_ratio']:.3f})."
                        )
                    else:
                        st.info(
                            f"ℹ️ **{compare_portfolio['name']} remains superior on a risk-adjusted basis** "
                            f"(Sharpe {compare_portfolio['sharpe_ratio']:.3f} vs {frontier_sharpe_calc:.3f})."
                        )
        
        # TAB 7: BENCHMARK (if available)
        if tab8 is not None:
            with tab7:
                st.markdown("### 🎯 Benchmark Comparison")
                st.markdown(f"Comparing against **{get_display_name(st.session_state.benchmark)}**")
                
                if benchmark_returns is not None and len(benchmark_returns) > 0:
                    bench_cumret = float((1 + benchmark_returns).prod() - 1)
                    n_yrs = len(benchmark_returns) / 252
                    bench_annret = float((1 + bench_cumret)**(1/n_yrs) - 1) if n_yrs > 0 else 0
                    bench_vol = float(benchmark_returns.std() * np.sqrt(252))
                    bench_sharpe = (bench_annret - rf_rate) / bench_vol if bench_vol > 0 else 0
                    
                    bcols = st.columns(4)
                    bcols[0].metric("📈 Return", f"{bench_annret*100:.2f}%")
                    bcols[1].metric("📊 Vol", f"{bench_vol*100:.2f}%")
                    bcols[2].metric("⭐ Sharpe", f"{bench_sharpe:.3f}")
                    bcols[3].metric("💰 Cum", f"{bench_cumret*100:.2f}%")
                    
                    st.markdown("---")
                    
                    portfolio_keys = list(analyzer.portfolios.keys())
                    sel_bench = st.multiselect("Compare with", portfolio_keys, default=portfolio_keys[:3], format_func=lambda x: analyzer.portfolios[x]['name'], key="bench_sel")
                    
                    if sel_bench:
                        st.markdown("#### 📊 Performance vs Benchmark")
                        fig = go.Figure()
                        
                        bench_cum = (1 + benchmark_returns).cumprod() * 100
                        bench_vals = bench_cum.values.flatten() if hasattr(bench_cum.values, 'flatten') else bench_cum.values
                        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_vals, name=f"{get_display_name(st.session_state.benchmark)} (Benchmark)",
                            mode='lines', line=dict(color='#FFFFFF', width=3, dash='dash'),
                            hovertemplate=f'<b>{get_display_name(st.session_state.benchmark)}</b><br>Value: %{{y:.2f}}<extra></extra>'))
                        
                        for i, p_name in enumerate(sel_bench):
                            p = analyzer.portfolios[p_name]
                            cum_val = (1 + p['returns']).cumprod() * 100
                            fig.add_trace(go.Scatter(x=cum_val.index, y=cum_val.values, name=p['name'], mode='lines',
                                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                                hovertemplate=f'<b>{p["name"]}</b><br>Value: %{{y:.2f}}<extra></extra>'))
                        
                        fig.update_layout(height=450, hovermode='x unified', xaxis_title="Date", yaxis_title="Value (Base 100)",
                                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                        fig = apply_plotly_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("#### 📋 Comparison Table")
                        bench_comp = [{'Strategy': f"{get_display_name(st.session_state.benchmark)} (Benchmark)", 'Return': f"{bench_annret*100:.2f}%",
                                      'Vol': f"{bench_vol*100:.2f}%", 'Sharpe': f"{bench_sharpe:.3f}", 'Excess': "-"}]
                        for p_name in sel_bench:
                            p = analyzer.portfolios[p_name]
                            excess = (p['annualized_return'] - bench_annret) * 100
                            bench_comp.append({'Strategy': p['name'], 'Return': f"{p['annualized_return']*100:.2f}%",
                                'Vol': f"{p['annualized_volatility']*100:.2f}%", 'Sharpe': f"{p['sharpe_ratio']:.3f}",
                                'Excess': f"{excess:+.2f}%"})
                        st.markdown(create_styled_table(pd.DataFrame(bench_comp)), unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Benchmark data not available")
        
        # EXPORT TAB
        export_tab = tab8 if tab8 is not None else tab7
        
        with export_tab:
            st.markdown("### 📥 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Excel Report")
                if OPENPYXL_AVAILABLE:
                    if st.button("Generate Excel", use_container_width=True, key="exp_xlsx"):
                        with st.spinner("Creating..."):
                            try:
                                filename = analyzer.export_to_excel()
                                with open(filename, 'rb') as f:
                                    st.download_button("⬇️ Download Excel", f, filename, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
                                st.success("✅ Ready!")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                else:
                    st.warning("openpyxl not installed")
            
            with col2:
                st.markdown("#### 🔗 JSON Export")
                if st.button("Generate JSON", use_container_width=True, key="exp_json"):
                    export_data = {
                        'metadata': {'generated': datetime.now().isoformat(), 'assets': [{'ticker': t, 'name': get_display_name(t)} for t in symbols]},
                        'portfolios': {name: {'name': p['name'], 'weights': {get_display_name(s): round(float(w)*100, 2) for s, w in zip(symbols, p['weights']) if w > 0.001},
                            'metrics': {'return': round(p['annualized_return']*100, 2), 'volatility': round(p['annualized_volatility']*100, 2),
                                'sharpe': round(p['sharpe_ratio'], 3)}} for name, p in analyzer.portfolios.items()}
                    }
                    st.download_button("⬇️ Download JSON", json.dumps(export_data, indent=2), f"portfolio_{datetime.now().strftime('%Y%m%d')}.json", "application/json", key="dl_json")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-section">
    <h3>📊 Portfolio Analyzer Pro</h3>
    <p><strong>Features:</strong> 7 Strategies • Walk-Forward Backtest • Yahoo Search • Deep Statistics • GARCH</p>
    <p style="opacity:0.6;font-size:0.8rem;">⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
