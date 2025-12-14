"""Portfolio Analyzer Pro - Streamlit Cloud Ready - v3.0 (Mobile Responsive)"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import warnings

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

st.set_page_config(page_title="Portfolio Analyzer Pro", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# CSS Styling with Mobile Responsive
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
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --border-color: rgba(99,102,241,0.2);
}

* { font-family: 'Inter', sans-serif; }
.main, .stApp { background: var(--bg-primary); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border-right: 1px solid var(--border-color);
}

/* Button Styling - Better contrast */
.stButton > button {
    width: 100%;
    background: var(--accent-gradient);
    color: #000000 !important;
    padding: 0.875rem 1.5rem;
    font-weight: 700;
    border-radius: 12px;
    border: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    transition: all 0.3s;
    text-shadow: none;
}

.stButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.1);
    color: #000000 !important;
}

/* Form submit button */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 1rem 1.5rem;
}

.stFormSubmitButton > button:hover {
    filter: brightness(1.15);
    color: #ffffff !important;
}

.dashboard-card {
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    padding: 1.75rem;
    border-radius: 16px;
    margin: 0.75rem 0;
    border: 1px solid var(--border-color);
}

.dashboard-card h3 {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

.dashboard-card ul, .dashboard-card ol, .dashboard-card p {
    color: var(--text-secondary);
    line-height: 1.7;
}

.ticker-pill {
    display: inline-block;
    background: var(--accent-gradient);
    color: white;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    margin: 0.25rem;
    font-size: 0.8rem;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem !important;
}

h1 {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

h2 {
    color: var(--text-primary);
    font-weight: 700;
    border-bottom: 2px solid transparent;
    border-image: var(--accent-gradient);
    border-image-slice: 1;
    padding-bottom: 0.75rem;
}

h3 { color: var(--accent-primary); font-weight: 600; }
p, li, span { color: var(--text-secondary); }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 0.5rem;
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary);
    font-weight: 600;
    border-radius: 8px;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-gradient) !important;
    color: white !important;
}

.hero-section { text-align: center; padding: 2rem 0; }

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-divider {
    width: 120px;
    height: 3px;
    background: var(--accent-gradient);
    margin: 1.5rem auto;
    border-radius: 2px;
}

.footer-section {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
    border-radius: 20px;
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
    font-size: 0.9rem;
}

.styled-table th {
    background: var(--accent-gradient);
    color: white;
    padding: 12px 10px;
    text-align: left;
    font-weight: 600;
    font-size: 0.8rem;
}

.styled-table td {
    padding: 10px;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-secondary);
    font-size: 0.85rem;
}

.styled-table tr:hover { background: rgba(99,102,241,0.1); }
.styled-table tr:last-child td { border-bottom: none; }

/* Mobile Responsive Styles */
@media (max-width: 768px) {
    .hero-title { font-size: 1.8rem; }
    .hero-subtitle { font-size: 0.8rem; letter-spacing: 1px; }
    .hero-section { padding: 1rem 0; }
    
    .dashboard-card { padding: 1rem; margin: 0.5rem 0; }
    .dashboard-card h3 { font-size: 1rem; }
    
    .styled-table { font-size: 0.75rem; }
    .styled-table th, .styled-table td { padding: 8px 6px; }
    
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.65rem !important; }
    
    .stTabs [data-baseweb="tab"] { font-size: 0.8rem; padding: 0.4rem 0.8rem; }
    
    h2 { font-size: 1.3rem; }
    h3 { font-size: 1.1rem; }
    
    .ticker-pill { font-size: 0.7rem; padding: 0.25rem 0.6rem; }
    
    .footer-section { padding: 1.5rem 1rem; }
    .footer-section p { font-size: 0.8rem; }
}

@media (max-width: 480px) {
    .hero-title { font-size: 1.5rem; }
    .styled-table th, .styled-table td { padding: 6px 4px; font-size: 0.7rem; }
    [data-testid="column"] { padding: 0 0.25rem !important; }
}

/* Fix for checkbox labels */
.stCheckbox label { color: var(--text-secondary) !important; }
.stCheckbox label:hover { color: var(--text-primary) !important; }

/* Expander styling */
.streamlit-expanderHeader {
    background: var(--bg-card);
    border-radius: 8px;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)

# Ticker Database with Company Names
TICKER_INFO = {
    # US Tech
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet (Google)", "AMZN": "Amazon",
    "META": "Meta (Facebook)", "NVDA": "NVIDIA", "AMD": "AMD", "INTC": "Intel",
    "CRM": "Salesforce", "ADBE": "Adobe", "NFLX": "Netflix", "PYPL": "PayPal",
    "QCOM": "Qualcomm", "AVGO": "Broadcom", "PLTR": "Palantir", "CRWD": "CrowdStrike",
    "SNOW": "Snowflake", "NET": "Cloudflare", "CSCO": "Cisco", "ORCL": "Oracle",
    "IBM": "IBM", "SHOP": "Shopify", "SQ": "Block (Square)", "UBER": "Uber",
    # US Finance
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo", "C": "Citigroup",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "BLK": "BlackRock", "SCHW": "Charles Schwab",
    "AXP": "American Express", "V": "Visa", "MA": "Mastercard", "COF": "Capital One",
    # US Healthcare
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "PFE": "Pfizer", "ABBV": "AbbVie",
    "TMO": "Thermo Fisher", "ABT": "Abbott Labs", "MRK": "Merck", "LLY": "Eli Lilly",
    "AMGN": "Amgen", "BMY": "Bristol-Myers Squibb", "GILD": "Gilead Sciences", "ISRG": "Intuitive Surgical",
    # US Consumer
    "TSLA": "Tesla", "HD": "Home Depot", "MCD": "McDonald's", "NKE": "Nike",
    "SBUX": "Starbucks", "LOW": "Lowe's", "TGT": "Target", "COST": "Costco",
    "WMT": "Walmart", "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "DIS": "Disney", "ABNB": "Airbnb", "BKNG": "Booking Holdings",
    # US Energy
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips", "SLB": "Schlumberger",
    "EOG": "EOG Resources", "MPC": "Marathon Petroleum", "OXY": "Occidental Petroleum",
    "DVN": "Devon Energy", "HAL": "Halliburton", "KMI": "Kinder Morgan",
    # Indices
    "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ Composite",
    "^RUT": "Russell 2000", "^FTSE": "FTSE 100", "^GDAXI": "DAX (Germany)",
    "^N225": "Nikkei 225", "^STOXX50E": "Euro Stoxx 50",
    # ETFs
    "SPY": "SPDR S&P 500 ETF", "VOO": "Vanguard S&P 500", "VTI": "Vanguard Total Market",
    "QQQ": "Invesco NASDAQ 100", "IWM": "iShares Russell 2000", "VEA": "Vanguard FTSE Developed",
    "VWO": "Vanguard FTSE Emerging", "EEM": "iShares MSCI Emerging", "EFA": "iShares MSCI EAFE",
    "ACWI": "iShares MSCI ACWI", "XLK": "Technology Select SPDR", "XLV": "Health Care Select SPDR",
    "XLF": "Financial Select SPDR", "XLE": "Energy Select SPDR", "XLI": "Industrial Select SPDR",
    "XLY": "Consumer Discret. SPDR", "XLP": "Consumer Staples SPDR", "XLU": "Utilities Select SPDR",
    "VNQ": "Vanguard Real Estate", "ARKK": "ARK Innovation ETF",
    # Crypto
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana", "ADA-USD": "Cardano", "XRP-USD": "Ripple (XRP)",
    "DOGE-USD": "Dogecoin", "DOT-USD": "Polkadot", "AVAX-USD": "Avalanche",
    # Bonds
    "TLT": "iShares 20+ Year Treasury", "IEF": "iShares 7-10 Year Treasury",
    "SHY": "iShares 1-3 Year Treasury", "AGG": "iShares Core US Aggregate",
    "BND": "Vanguard Total Bond", "LQD": "iShares Investment Grade Corp",
    "HYG": "iShares High Yield Corp", "TIP": "iShares TIPS Bond",
    # Gold & Commodities
    "GLD": "SPDR Gold Shares", "IAU": "iShares Gold Trust", "SLV": "iShares Silver Trust",
    "GDX": "VanEck Gold Miners", "GDXJ": "VanEck Junior Gold Miners",
}

TICKER_DATABASE = {
    "🇺🇸 US Tech": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AMD","INTC","CRM","ADBE","NFLX","PYPL","PLTR","CRWD","SNOW","NET"],
    "🇺🇸 US Finance": ["JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP","V","MA","COF"],
    "🇺🇸 US Healthcare": ["JNJ","UNH","PFE","ABBV","TMO","ABT","MRK","LLY","AMGN","BMY","GILD","ISRG"],
    "🇺🇸 US Consumer": ["TSLA","HD","MCD","NKE","SBUX","LOW","TGT","COST","WMT","PG","KO","PEP"],
    "🇺🇸 US Energy": ["XOM","CVX","COP","SLB","EOG","MPC","OXY","DVN","HAL","KMI"],
    "📊 Indices": ["^GSPC","^DJI","^IXIC","^RUT","^FTSE","^GDAXI","^N225","^STOXX50E"],
    "📈 ETF Broad": ["SPY","VOO","VTI","QQQ","IWM","VEA","VWO","EEM","EFA","ACWI"],
    "📈 ETF Sector": ["XLK","XLV","XLF","XLE","XLI","XLY","XLP","XLU","VNQ"],
    "💎 Crypto": ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","ADA-USD","XRP-USD","DOGE-USD"],
    "🏛️ Bonds": ["TLT","IEF","SHY","AGG","BND","LQD","HYG","TIP"],
    "💰 Gold": ["GLD","IAU","SLV","GDX","GDXJ"]
}

def get_display_name(ticker):
    return TICKER_INFO.get(ticker, ticker)

def get_ticker_from_name(name):
    for ticker, display_name in TICKER_INFO.items():
        if display_name == name:
            return ticker
    return name

# Distinct Colors
CHART_COLORS = ['#FF6B6B','#4ECDC4','#FFE66D','#95E1D3','#F38181','#AA96DA','#FCBAD3','#A8D8EA','#FF9F43','#6C5CE7']

def apply_plotly_theme(fig):
    fig.update_layout(
        title=None,  # Remove title to avoid "undefined"
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', family='Inter', size=12),
        legend=dict(
            bgcolor='rgba(26,26,36,0.95)',
            bordercolor='rgba(99,102,241,0.5)',
            borderwidth=1,
            font=dict(color='#F8FAFC', size=11)
        ),
        xaxis=dict(
            gridcolor='rgba(99,102,241,0.15)',
            linecolor='rgba(99,102,241,0.3)',
            tickfont=dict(color='#E2E8F0', size=11),
            title_font=dict(color='#E2E8F0', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(99,102,241,0.15)',
            linecolor='rgba(99,102,241,0.3)',
            tickfont=dict(color='#E2E8F0', size=11),
            title_font=dict(color='#E2E8F0', size=12)
        ),
        hoverlabel=dict(
            bgcolor='#1E1E2E',
            bordercolor='#6366F1',
            font=dict(color='#F8FAFC', size=12)
        ),
        margin=dict(t=30, b=50, l=50, r=50)
    )
    return fig

def create_styled_table(df, title=""):
    html = f'<div style="overflow-x:auto;"><table class="styled-table">'
    if title:
        html += f'<caption style="color:#94a3b8;padding:10px;font-size:1rem;font-weight:600">{title}</caption>'
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

# Session State Initialization
default_states = {
    'analyzer': None,
    'analysis_complete': False,
    'selected_tickers': [],
    'saved_portfolios': {},
    'alerts': [],
    'benchmark': '^GSPC',
    'use_benchmark': True,
    'benchmark_returns': None,
    'run_analysis': False,
    'form_tickers': [],
    'form_start_date': datetime(2020, 1, 1),
    'form_end_date': datetime.now(),
    'form_risk_free': 2.0,
    'form_window': 3,
    'form_benchmark': '^GSPC',
    'form_use_benchmark': True,
    'form_enable_alerts': True,
    'form_max_dd': 20,
    'form_min_sharpe': 0.5
}

for key, default in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default
# Part 2: Header and Sidebar with Form

# Header
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">📊 PORTFOLIO ANALYZER PRO</h1>
    <p class="hero-subtitle">Advanced Multi-Asset Optimization System</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Sidebar with Form to prevent auto-refresh
with st.sidebar:
    st.markdown("## ⚙️ CONFIGURATION")
    
    # Use a form to prevent auto-refresh
    with st.form(key="analysis_form"):
        selection_method = st.radio("Selection Method", ["📋 Database", "✍️ Manual"], horizontal=True)
        
        selected_symbols = []
        
        if selection_method == "📋 Database":
            st.markdown("#### 📚 Select Assets")
            search_query = st.text_input("🔍 Search", placeholder="e.g., Apple, Tesla...")
            
            for category, tickers in TICKER_DATABASE.items():
                with st.expander(category, expanded=False):
                    if search_query:
                        filtered = [t for t in tickers if search_query.upper() in t.upper() or search_query.upper() in get_display_name(t).upper()]
                    else:
                        filtered = tickers
                    
                    if filtered:
                        # Select All checkbox
                        select_all_key = f"selall_{category}"
                        select_all = st.checkbox(f"✅ Select All ({len(filtered)})", key=select_all_key)
                        
                        # Individual checkboxes
                        for ticker in filtered:
                            default_val = select_all
                            if st.checkbox(get_display_name(ticker), value=default_val, key=f"cb_{category}_{ticker}"):
                                if ticker not in selected_symbols:
                                    selected_symbols.append(ticker)
        else:
            st.markdown("#### ✍️ Manual Entry")
            manual_input = st.text_area(
                "Enter tickers (comma separated)",
                height=100,
                placeholder="AAPL, MSFT, GOOGL, AMZN",
                help="Use Yahoo Finance ticker symbols"
            )
            if manual_input:
                import re
                selected_symbols = [s.strip().upper() for s in re.split('[,\n]', manual_input) if s.strip()]
        
        selected_symbols = list(dict.fromkeys(selected_symbols))
        
        st.markdown("---")
        st.markdown("#### 📅 Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1), min_value=datetime(2000, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1)
        window_years = st.slider("Rolling Window (years)", 1, 5, 3)
        
        st.markdown("#### 📊 Benchmark")
        use_benchmark = st.checkbox("Compare with benchmark", value=True)
        benchmark_options = ["^GSPC", "^DJI", "^IXIC", "SPY", "QQQ", "VTI"]
        benchmark_ticker = st.selectbox(
            "Select Benchmark",
            benchmark_options,
            format_func=lambda x: get_display_name(x),
            disabled=not use_benchmark
        )
        
        st.markdown("#### 🔔 Alerts")
        enable_alerts = st.checkbox("Enable risk alerts", value=True)
        
        col1, col2 = st.columns(2)
        with col1:
            max_dd_threshold = st.number_input("Max DD Alert (%)", 5, 50, 20, disabled=not enable_alerts)
        with col2:
            min_sharpe_threshold = st.number_input("Min Sharpe Alert", 0.0, 2.0, 0.5, 0.1, disabled=not enable_alerts)
        
        st.markdown("---")
        
        # Show selected count
        st.info(f"📦 **{len(selected_symbols)}** assets selected")
        
        # Submit button
        submitted = st.form_submit_button("🚀 START ANALYSIS", use_container_width=True)
        
        if submitted:
            if len(selected_symbols) < 2:
                st.error("⚠️ Select at least 2 assets!")
            else:
                # Store form values
                st.session_state.form_tickers = selected_symbols
                st.session_state.form_start_date = start_date
                st.session_state.form_end_date = end_date
                st.session_state.form_risk_free = risk_free_rate
                st.session_state.form_window = window_years
                st.session_state.form_benchmark = benchmark_ticker if use_benchmark else None
                st.session_state.form_use_benchmark = use_benchmark
                st.session_state.form_enable_alerts = enable_alerts
                st.session_state.form_max_dd = max_dd_threshold
                st.session_state.form_min_sharpe = min_sharpe_threshold
                st.session_state.run_analysis = True
                st.session_state.selected_tickers = selected_symbols
    
    # Reset button (outside form)
    if st.session_state.analyzer is not None:
        if st.button("🔄 Reset Analysis", use_container_width=True, key="reset_btn"):
            st.session_state.analyzer = None
            st.session_state.analysis_complete = False
            st.session_state.selected_tickers = []
            st.session_state.benchmark_returns = None
            st.session_state.run_analysis = False
            st.rerun()

# Main Content
if not st.session_state.run_analysis and st.session_state.analyzer is None:
    # Landing Page
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='dashboard-card'>
            <h3>🎯 QUICK START</h3>
            <ol>
                <li>Select assets from sidebar</li>
                <li>Configure parameters</li>
                <li>Click Start Analysis</li>
                <li>Explore results</li>
            </ol>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='dashboard-card'>
            <h3>📚 DATABASE</h3>
            <p>Access to <strong>100+</strong> assets:</p>
            <ul>
                <li>US Stocks by sector</li>
                <li>ETFs & Indices</li>
                <li>Crypto & Bonds</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='dashboard-card'>
            <h3>⚡ FEATURES</h3>
            <ul>
                <li>8 optimization strategies</li>
                <li>Rolling analysis</li>
                <li>Efficient frontier</li>
                <li>Benchmark comparison</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🎲 QUICK EXAMPLES")
    st.markdown("*Click to load a pre-configured portfolio:*")
    
    examples = {
        "🔥 Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        "💰 Dividend Stars": ["JNJ", "PG", "KO", "PEP", "MCD", "WMT"],
        "🌍 Global ETFs": ["SPY", "EFA", "EEM", "GLD", "TLT", "VNQ"],
        "🏦 Financials": ["JPM", "BAC", "GS", "MS", "V", "MA"]
    }
    
    cols = st.columns(4)
    for idx, (name, tickers) in enumerate(examples.items()):
        with cols[idx]:
            # Create a mini-form for each example button
            with st.form(key=f"example_form_{idx}"):
                if st.form_submit_button(name, use_container_width=True):
                    st.session_state.form_tickers = tickers
                    st.session_state.selected_tickers = tickers
                    st.session_state.form_start_date = datetime(2020, 1, 1)
                    st.session_state.form_end_date = datetime.now()
                    st.session_state.form_risk_free = 2.0
                    st.session_state.form_window = 3
                    st.session_state.form_benchmark = "^GSPC"
                    st.session_state.form_use_benchmark = True
                    st.session_state.form_enable_alerts = True
                    st.session_state.form_max_dd = 20
                    st.session_state.form_min_sharpe = 0.5
                    st.session_state.run_analysis = True
                    st.rerun()
# Part 3: Analysis Logic

if st.session_state.run_analysis or st.session_state.analyzer is not None:
    
    # Run analysis only if triggered
    if st.session_state.run_analysis and st.session_state.analyzer is None:
        symbols = st.session_state.form_tickers
        start_date = st.session_state.form_start_date
        end_date = st.session_state.form_end_date
        risk_free_rate = st.session_state.form_risk_free / 100
        window_years = st.session_state.form_window
        use_benchmark = st.session_state.form_use_benchmark
        benchmark_ticker = st.session_state.form_benchmark
        enable_alerts = st.session_state.form_enable_alerts
        max_dd_threshold = st.session_state.form_max_dd
        min_sharpe_threshold = st.session_state.form_min_sharpe
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing...")
            progress_bar.progress(10)
            
            from portfolio_analyzer import AdvancedPortfolioAnalyzer
            
            status_text.text("📥 Downloading market data from Yahoo Finance...")
            progress_bar.progress(20)
            
            analyzer = AdvancedPortfolioAnalyzer(
                symbols,
                start_date=start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
                end_date=end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
                risk_free_rate=risk_free_rate
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
            
            status_text.text("💼 Optimizing portfolios (8 strategies)...")
            analyzer.build_all_portfolios()
            progress_bar.progress(85)
            
            # Benchmark
            benchmark_returns = None
            if use_benchmark and benchmark_ticker and YF_AVAILABLE:
                try:
                    status_text.text(f"📈 Loading benchmark ({get_display_name(benchmark_ticker)})...")
                    benchmark_df = yf.download(
                        benchmark_ticker,
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    if not benchmark_df.empty:
                        if isinstance(benchmark_df.columns, pd.MultiIndex):
                            benchmark_prices = benchmark_df['Close'][benchmark_ticker]
                        else:
                            benchmark_prices = benchmark_df['Close']
                        
                        benchmark_prices = pd.Series(
                            benchmark_prices.values.flatten(),
                            index=benchmark_df.index
                        )
                        benchmark_returns = benchmark_prices.pct_change().dropna()
                        
                        if len(benchmark_returns) == 0:
                            benchmark_returns = None
                except Exception as e:
                    st.warning(f"⚠️ Benchmark loading issue: {str(e)}")
                    benchmark_returns = None
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store results
            st.session_state.analyzer = analyzer
            st.session_state.benchmark_returns = benchmark_returns
            st.session_state.benchmark = benchmark_ticker
            st.session_state.use_benchmark = use_benchmark
            st.session_state.analysis_complete = True
            st.session_state.run_analysis = False  # Reset trigger
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Generate alerts
            if enable_alerts:
                st.session_state.alerts = []
                for name, portfolio in analyzer.portfolios.items():
                    if abs(portfolio['max_drawdown']) > (max_dd_threshold / 100):
                        st.session_state.alerts.append({
                            'type': 'warning',
                            'portfolio': portfolio['name'],
                            'message': f"High Drawdown: {portfolio['max_drawdown']*100:.1f}%"
                        })
                    if portfolio['sharpe_ratio'] < min_sharpe_threshold:
                        st.session_state.alerts.append({
                            'type': 'info',
                            'portfolio': portfolio['name'],
                            'message': f"Low Sharpe: {portfolio['sharpe_ratio']:.2f}"
                        })
            
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            st.session_state.run_analysis = False
            st.stop()
    
    # Display results if analysis is complete
    if st.session_state.analyzer is not None:
        analyzer = st.session_state.analyzer
        benchmark_returns = st.session_state.benchmark_returns
        symbols = analyzer.symbols
        risk_free_rate = st.session_state.form_risk_free / 100
        window_years = st.session_state.form_window
        
        # Alert banner
        if st.session_state.alerts:
            with st.expander(f"🔔 {len(st.session_state.alerts)} Risk Alerts", expanded=False):
                for alert in st.session_state.alerts:
                    if alert['type'] == 'warning':
                        st.warning(f"**{alert['portfolio']}**: {alert['message']}")
                    else:
                        st.info(f"**{alert['portfolio']}**: {alert['message']}")
        
        # KPI Dashboard
        st.markdown("## 📊 KPI DASHBOARD")
        portfolios_list = list(analyzer.portfolios.values())
        
        num_cols = 6 if (st.session_state.use_benchmark and benchmark_returns is not None) else 5
        kpi_cols = st.columns(num_cols)
        
        with kpi_cols[0]:
            best_return = max(portfolios_list, key=lambda x: x['annualized_return'])
            st.metric("🏆 Best Return", f"{best_return['annualized_return']*100:.1f}%", delta=best_return['name'].split()[0])
        with kpi_cols[1]:
            best_sharpe = max(portfolios_list, key=lambda x: x['sharpe_ratio'])
            st.metric("⭐ Best Sharpe", f"{best_sharpe['sharpe_ratio']:.2f}", delta=best_sharpe['name'].split()[0])
        with kpi_cols[2]:
            min_vol = min(portfolios_list, key=lambda x: x['annualized_volatility'])
            st.metric("🛡️ Min Volatility", f"{min_vol['annualized_volatility']*100:.1f}%", delta=min_vol['name'].split()[0])
        with kpi_cols[3]:
            best_dd = max(portfolios_list, key=lambda x: x['max_drawdown'])
            st.metric("📉 Min Drawdown", f"{best_dd['max_drawdown']*100:.1f}%", delta=best_dd['name'].split()[0])
        with kpi_cols[4]:
            avg_ret = np.mean([p['annualized_return'] for p in portfolios_list])
            st.metric("📊 Avg Return", f"{avg_ret*100:.1f}%")
        
        if st.session_state.use_benchmark and benchmark_returns is not None and len(benchmark_returns) > 0:
            with kpi_cols[5]:
                try:
                    bench_cumret = float((1 + benchmark_returns).prod() - 1)
                    n_yrs = len(benchmark_returns) / 252
                    bench_annret = float((1 + bench_cumret)**(1/n_yrs) - 1) if n_yrs > 0 else 0
                    st.metric(f"📈 {get_display_name(st.session_state.benchmark)}", f"{bench_annret*100:.1f}%")
                except:
                    st.metric("📈 Benchmark", "N/A")
        
        st.markdown("---")
        
        # Tabs
        if st.session_state.use_benchmark and benchmark_returns is not None:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Overview", "💼 Portfolios", "📈 Performance", "🔄 Rolling",
                "🔗 Correlations", "📐 Frontier", "🎯 Benchmark", "📥 Export"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Overview", "💼 Portfolios", "📈 Performance", "🔄 Rolling",
                "🔗 Correlations", "📐 Frontier", "📥 Export"
            ])
            tab8 = None
# Part 4: Overview and Portfolio Tabs

        # TAB 1: OVERVIEW
        with tab1:
            st.markdown("### 📊 Overview")
            col1, col2 = st.columns([2.5, 1])
            
            with col1:
                st.markdown("#### 📈 Asset Performance (Base 100)")
                normalized = (analyzer.data / analyzer.data.iloc[0]) * 100
                fig = go.Figure()
                for i, ticker in enumerate(normalized.columns):
                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized[ticker],
                        name=get_display_name(ticker),
                        mode='lines',
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                        hovertemplate=f'<b>{get_display_name(ticker)}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ))
                fig.update_layout(
                    height=500,
                    hovermode='x unified',
                    xaxis_title="Date",
                    yaxis_title="Value (Base 100)",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
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
                st.metric("📅 Period", f"{len(analyzer.data)} days")
                st.metric("📦 Assets", len(symbols))
            
            # Correlation Heatmap
            st.markdown("#### 🔥 Correlation Heatmap")
            corr_matrix = analyzer.returns.corr()
            corr_display = corr_matrix.copy()
            corr_display.index = [get_display_name(t) for t in corr_display.index]
            corr_display.columns = [get_display_name(t) for t in corr_display.columns]
            
            fig = px.imshow(
                corr_display,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            fig.update_layout(height=450)
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: PORTFOLIOS
        with tab2:
            st.markdown("### 💼 Portfolio Analysis")
            
            st.markdown("#### 📊 Strategy Comparison")
            
            # Sorted comparison
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
            st.markdown(create_styled_table(comparison_df, "Ranked by Sharpe Ratio"), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Charts
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
                    textfont=dict(color='#E2E8F0', size=10)
                )])
                fig.update_layout(height=400, xaxis_tickangle=-45, yaxis_title="Annual Return (%)")
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
                        marker=dict(size=18, color=CHART_COLORS[i % len(CHART_COLORS)]),
                        text=[p['name'].split()[0]],
                        textposition='top center',
                        textfont=dict(color='#E2E8F0', size=9),
                        hovertemplate=f"<b>{p['name']}</b><br>Return: %{{y:.2f}}%<br>Vol: %{{x:.2f}}%<extra></extra>"
                    ))
                fig.update_layout(height=400, xaxis_title="Volatility (%)", yaxis_title="Return (%)", showlegend=False)
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Portfolio Details
            st.markdown("#### 🔍 Portfolio Details")
            
            portfolio_keys = list(analyzer.portfolios.keys())
            selected_p = st.selectbox(
                "Select strategy",
                portfolio_keys,
                format_func=lambda x: analyzer.portfolios[x]['name'],
                key="portfolio_detail_select"
            )
            portfolio = analyzer.portfolios[selected_p]
            
            # Metrics
            st.markdown("##### 📈 Performance Metrics")
            mcols = st.columns(6)
            metrics_list = [
                ("Return", f"{portfolio['annualized_return']*100:.2f}%"),
                ("Volatility", f"{portfolio['annualized_volatility']*100:.2f}%"),
                ("Sharpe", f"{portfolio['sharpe_ratio']:.3f}"),
                ("Sortino", f"{portfolio['sortino_ratio']:.3f}"),
                ("Max DD", f"{portfolio['max_drawdown']*100:.2f}%"),
                ("Calmar", f"{portfolio['calmar_ratio']:.3f}")
            ]
            for col, (label, value) in zip(mcols, metrics_list):
                col.metric(label, value)
            
            st.markdown("##### ⚖️ Asset Allocation")
            
            # Weights data
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
                table_data = [{'Asset': row['Asset'], 'Weight': f"{row['Weight']:.2f}%"} for _, row in weights_df.iterrows()]
                st.markdown(create_styled_table(pd.DataFrame(table_data)), unsafe_allow_html=True)
            
            with col2:
                fig = go.Figure(data=[go.Pie(
                    labels=weights_df['Asset'],
                    values=weights_df['Weight'],
                    hole=0.4,
                    marker_colors=CHART_COLORS[:len(weights_df)],
                    textinfo='percent',
                    textposition='outside',
                    textfont=dict(color='#E2E8F0', size=10),
                    hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>'
                )])
                fig.update_layout(
                    height=350,
                    showlegend=False,
                    annotations=[dict(text=portfolio['name'].split()[0], x=0.5, y=0.5, font_size=12, font_color='#E2E8F0', showarrow=False)]
                )
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
# Part 5: Performance and Rolling Tabs

        # TAB 3: PERFORMANCE
        with tab3:
            st.markdown("### 📈 Performance Comparison")
            
            portfolio_keys = list(analyzer.portfolios.keys())
            sel_perf = st.multiselect(
                "Select strategies to compare",
                portfolio_keys,
                default=portfolio_keys[:4],
                format_func=lambda x: analyzer.portfolios[x]['name'],
                key="perf_multiselect"
            )
            
            if sel_perf:
                # Cumulative Performance
                st.markdown("#### 📊 Cumulative Performance")
                fig = go.Figure()
                for i, p_name in enumerate(sel_perf):
                    p = analyzer.portfolios[p_name]
                    cum_val = (1 + p['returns']).cumprod() * 100
                    fig.add_trace(go.Scatter(
                        x=cum_val.index,
                        y=cum_val.values,
                        name=p['name'],
                        mode='lines',
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=3),
                        hovertemplate=f'<b>{p["name"]}</b><br>Value: %{{y:.2f}}<extra></extra>'
                    ))
                fig.update_layout(
                    height=450,
                    hovermode='x unified',
                    xaxis_title="Date",
                    yaxis_title="Value (Base 100)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown
                st.markdown("#### 📉 Drawdown Analysis")
                fig = go.Figure()
                for i, p_name in enumerate(sel_perf):
                    p = analyzer.portfolios[p_name]
                    cum_val = (1 + p['returns']).cumprod()
                    roll_max = cum_val.expanding().max()
                    dd = (cum_val - roll_max) / roll_max * 100
                    
                    color = CHART_COLORS[i % len(CHART_COLORS)]
                    # Convert hex to rgba for fill
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    
                    fig.add_trace(go.Scatter(
                        x=dd.index,
                        y=dd.values,
                        name=p['name'],
                        mode='lines',
                        line=dict(color=color, width=2),
                        fill='tozeroy',
                        fillcolor=f'rgba({r},{g},{b},0.2)',
                        hovertemplate=f'<b>{p["name"]}</b><br>Drawdown: %{{y:.2f}}%<extra></extra>'
                    ))
                fig.update_layout(
                    height=350,
                    hovermode='x unified',
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison table
                st.markdown("#### 📋 Detailed Comparison")
                stats_data = []
                for p_name in sel_perf:
                    p = analyzer.portfolios[p_name]
                    stats_data.append({
                        'Strategy': p['name'],
                        'Return': f"{p['annualized_return']*100:.2f}%",
                        'Volatility': f"{p['annualized_volatility']*100:.2f}%",
                        'Sharpe': f"{p['sharpe_ratio']:.3f}",
                        'Sortino': f"{p['sortino_ratio']:.3f}",
                        'Max DD': f"{p['max_drawdown']*100:.2f}%",
                        'Cumulative': f"{p['cumulative_return']*100:.2f}%"
                    })
                st.markdown(create_styled_table(pd.DataFrame(stats_data)), unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please select at least one strategy")
        
        # TAB 4: ROLLING
        with tab4:
            st.markdown("### 🔄 Rolling Analysis")
            st.info(f"📊 Rolling window: **{window_years} years** ({window_years * 252} trading days)")
            
            portfolio_keys = list(analyzer.portfolios.keys())
            sel_roll = st.multiselect(
                "Select strategies",
                portfolio_keys,
                default=portfolio_keys[:3],
                format_func=lambda x: analyzer.portfolios[x]['name'],
                key="roll_multiselect"
            )
            
            if sel_roll:
                rolling_data = {}
                window = window_years * 252
                
                for p_name in sel_roll:
                    p = analyzer.portfolios[p_name]
                    rets = p['returns']
                    if len(rets) >= window:
                        roll_ret = rets.rolling(window=window).apply(
                            lambda x: (1+x).prod()**(252/len(x))-1 if len(x)==window else np.nan
                        )
                        roll_vol = rets.rolling(window=window).std() * np.sqrt(252)
                        roll_sharpe = (roll_ret - risk_free_rate) / roll_vol
                        rolling_data[p['name']] = pd.DataFrame({
                            'Return': roll_ret,
                            'Volatility': roll_vol,
                            'Sharpe': roll_sharpe
                        }).dropna()
                
                if rolling_data:
                    # Rolling Returns
                    st.markdown("#### 📈 Rolling Returns")
                    fig = go.Figure()
                    for i, (name, data) in enumerate(rolling_data.items()):
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Return']*100,
                            name=name,
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                            hovertemplate=f'<b>{name}</b><br>Return: %{{y:.2f}}%<extra></extra>'
                        ))
                    fig.update_layout(
                        height=380,
                        hovermode='x unified',
                        yaxis_title="Return (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Rolling Volatility
                    st.markdown("#### 📊 Rolling Volatility")
                    fig = go.Figure()
                    for i, (name, data) in enumerate(rolling_data.items()):
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Volatility']*100,
                            name=name,
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                            hovertemplate=f'<b>{name}</b><br>Volatility: %{{y:.2f}}%<extra></extra>'
                        ))
                    fig.update_layout(
                        height=380,
                        hovermode='x unified',
                        yaxis_title="Volatility (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Rolling Sharpe
                    st.markdown("#### ⭐ Rolling Sharpe Ratio")
                    fig = go.Figure()
                    for i, (name, data) in enumerate(rolling_data.items()):
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Sharpe'],
                            name=name,
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                            hovertemplate=f'<b>{name}</b><br>Sharpe: %{{y:.3f}}<extra></extra>'
                        ))
                    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    fig.update_layout(
                        height=380,
                        hovermode='x unified',
                        yaxis_title="Sharpe Ratio",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary table
                    st.markdown("#### 📋 Rolling Statistics Summary")
                    roll_stats = []
                    for name, data in rolling_data.items():
                        roll_stats.append({
                            'Strategy': name,
                            'Avg Return': f"{data['Return'].mean()*100:.2f}%",
                            'Avg Vol': f"{data['Volatility'].mean()*100:.2f}%",
                            'Avg Sharpe': f"{data['Sharpe'].mean():.3f}",
                            'Min Sharpe': f"{data['Sharpe'].min():.3f}",
                            'Max Sharpe': f"{data['Sharpe'].max():.3f}"
                        })
                    st.markdown(create_styled_table(pd.DataFrame(roll_stats)), unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Not enough data for the selected rolling window")
            else:
                st.warning("⚠️ Please select at least one strategy")
# Part 6: Correlations, Frontier, Benchmark, Export, Footer

        # TAB 5: CORRELATIONS
        with tab5:
            st.markdown("### 🔗 Correlation Analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 📊 Asset Correlation Matrix")
                corr = analyzer.returns.corr()
                corr_display = corr.copy()
                corr_display.index = [get_display_name(t) for t in corr_display.index]
                corr_display.columns = [get_display_name(t) for t in corr_display.columns]
                
                fig = px.imshow(
                    corr_display,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1
                )
                fig.update_layout(height=500)
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Correlation Insights")
                pairs = []
                cols_list = corr.columns.tolist()
                for i in range(len(cols_list)):
                    for j in range(i+1, len(cols_list)):
                        pairs.append({
                            'A1': cols_list[i],
                            'A2': cols_list[j],
                            'Corr': corr.iloc[i, j]
                        })
                pairs_df = pd.DataFrame(pairs).sort_values('Corr', ascending=False)
                
                st.markdown("**🔥 Most Correlated**")
                for _, r in pairs_df.head(5).iterrows():
                    st.success(f"{get_display_name(r['A1'])} ↔ {get_display_name(r['A2'])}: **{r['Corr']:.2f}**")
                
                st.markdown("**❄️ Least Correlated**")
                for _, r in pairs_df.tail(5).iterrows():
                    st.info(f"{get_display_name(r['A1'])} ↔ {get_display_name(r['A2'])}: **{r['Corr']:.2f}**")
        
        # TAB 6: EFFICIENT FRONTIER
        with tab6:
            st.markdown("### 📐 Efficient Frontier")
            st.markdown("The efficient frontier shows optimal portfolios offering the highest return for each risk level.")
            
            with st.spinner("Calculating efficient frontier..."):
                frontier_df = analyzer.get_efficient_frontier(n_points=50)
                
                if not frontier_df.empty:
                    fig = go.Figure()
                    
                    # Frontier line
                    fig.add_trace(go.Scatter(
                        x=frontier_df['Volatility']*100,
                        y=frontier_df['Return']*100,
                        mode='lines',
                        name='Efficient Frontier',
                        line=dict(color='#FFE66D', width=4),
                        hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                    ))
                    
                    # Portfolio points
                    for i, (p_name, p) in enumerate(analyzer.portfolios.items()):
                        fig.add_trace(go.Scatter(
                            x=[p['annualized_volatility']*100],
                            y=[p['annualized_return']*100],
                            mode='markers+text',
                            name=p['name'],
                            marker=dict(size=16, color=CHART_COLORS[i % len(CHART_COLORS)], symbol='diamond',
                                       line=dict(width=2, color='white')),
                            text=[p['name'].split()[0]],
                            textposition='top center',
                            textfont=dict(color='#E2E8F0', size=10),
                            hovertemplate=f"<b>{p['name']}</b><br>Return: %{{y:.2f}}%<br>Vol: %{{x:.2f}}%<extra></extra>"
                        ))
                    
                    # Capital Market Line
                    max_sharpe_p = max(analyzer.portfolios.values(), key=lambda x: x['sharpe_ratio'])
                    cml_x = [0, max_sharpe_p['annualized_volatility']*100*2.5]
                    cml_y = [risk_free_rate*100, risk_free_rate*100 + max_sharpe_p['sharpe_ratio']*cml_x[1]]
                    fig.add_trace(go.Scatter(
                        x=cml_x, y=cml_y,
                        mode='lines',
                        name='Capital Market Line',
                        line=dict(color='#4ECDC4', width=2, dash='dash'),
                        hoverinfo='skip'
                    ))
                    
                    fig.update_layout(
                        height=550,
                        xaxis_title="Volatility (%)",
                        yaxis_title="Return (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stats
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📉 Min Volatility", f"{frontier_df['Volatility'].min()*100:.2f}%")
                    col2.metric("📈 Max Return", f"{frontier_df['Return'].max()*100:.2f}%")
                    col3.metric("⭐ Max Sharpe", f"{frontier_df['Sharpe'].max():.3f}")
                else:
                    st.warning("Could not calculate efficient frontier")
        
        # TAB 7: BENCHMARK (if enabled)
        if tab8 is not None:
            with tab7:
                st.markdown("### 🎯 Benchmark Comparison")
                st.markdown(f"Comparing portfolios against **{get_display_name(st.session_state.benchmark)}**")
                
                if benchmark_returns is not None and len(benchmark_returns) > 0:
                    # Benchmark metrics
                    bench_cumret = float((1 + benchmark_returns).prod() - 1)
                    n_yrs = len(benchmark_returns) / 252
                    bench_annret = float((1 + bench_cumret)**(1/n_yrs) - 1) if n_yrs > 0 else 0
                    bench_vol = float(benchmark_returns.std() * np.sqrt(252))
                    bench_sharpe = (bench_annret - risk_free_rate) / bench_vol if bench_vol > 0 else 0
                    
                    bcols = st.columns(4)
                    bcols[0].metric("📈 Return", f"{bench_annret*100:.2f}%")
                    bcols[1].metric("📊 Volatility", f"{bench_vol*100:.2f}%")
                    bcols[2].metric("⭐ Sharpe", f"{bench_sharpe:.3f}")
                    bcols[3].metric("💰 Cumulative", f"{bench_cumret*100:.2f}%")
                    
                    st.markdown("---")
                    
                    portfolio_keys = list(analyzer.portfolios.keys())
                    sel_bench = st.multiselect(
                        "Select strategies to compare",
                        portfolio_keys,
                        default=portfolio_keys[:3],
                        format_func=lambda x: analyzer.portfolios[x]['name'],
                        key="bench_multiselect"
                    )
                    
                    if sel_bench:
                        st.markdown("#### 📊 Performance vs Benchmark")
                        
                        fig = go.Figure()
                        
                        # Benchmark line
                        bench_cum = (1 + benchmark_returns).cumprod() * 100
                        bench_values = bench_cum.values.flatten() if hasattr(bench_cum.values, 'flatten') else bench_cum.values
                        
                        fig.add_trace(go.Scatter(
                            x=bench_cum.index,
                            y=bench_values,
                            name=f"{get_display_name(st.session_state.benchmark)} (Benchmark)",
                            mode='lines',
                            line=dict(color='#FFFFFF', width=3, dash='dash'),
                            hovertemplate=f'<b>{get_display_name(st.session_state.benchmark)}</b><br>Value: %{{y:.2f}}<extra></extra>'
                        ))
                        
                        # Portfolio lines
                        for i, p_name in enumerate(sel_bench):
                            p = analyzer.portfolios[p_name]
                            cum_val = (1 + p['returns']).cumprod() * 100
                            fig.add_trace(go.Scatter(
                                x=cum_val.index,
                                y=cum_val.values,
                                name=p['name'],
                                mode='lines',
                                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                                hovertemplate=f'<b>{p["name"]}</b><br>Value: %{{y:.2f}}<extra></extra>'
                            ))
                        
                        fig.update_layout(
                            height=500,
                            hovermode='x unified',
                            xaxis_title="Date",
                            yaxis_title="Value (Base 100)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                        )
                        fig = apply_plotly_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Comparison table
                        st.markdown("#### 📋 Performance vs Benchmark")
                        bench_comp = [{
                            'Strategy': f"{get_display_name(st.session_state.benchmark)} (Benchmark)",
                            'Return': f"{bench_annret*100:.2f}%",
                            'Volatility': f"{bench_vol*100:.2f}%",
                            'Sharpe': f"{bench_sharpe:.3f}",
                            'Excess Return': "-",
                            'Outperformed': "-"
                        }]
                        
                        for p_name in sel_bench:
                            p = analyzer.portfolios[p_name]
                            excess = (p['annualized_return'] - bench_annret) * 100
                            bench_comp.append({
                                'Strategy': p['name'],
                                'Return': f"{p['annualized_return']*100:.2f}%",
                                'Volatility': f"{p['annualized_volatility']*100:.2f}%",
                                'Sharpe': f"{p['sharpe_ratio']:.3f}",
                                'Excess Return': f"{excess:+.2f}%",
                                'Outperformed': "✅ Yes" if excess > 0 else "❌ No"
                            })
                        
                        st.markdown(create_styled_table(pd.DataFrame(bench_comp)), unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Benchmark data not available. Try selecting a different benchmark or check the date range.")
        
        # EXPORT TAB
        export_tab = tab8 if tab8 is not None else tab7
        
        with export_tab:
            st.markdown("### 📥 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Excel Report")
                st.markdown("Complete analysis with prices, returns, weights, and statistics.")
                if OPENPYXL_AVAILABLE:
                    if st.button("📥 Generate Excel Report", use_container_width=True, key="export_excel"):
                        with st.spinner("Creating report..."):
                            try:
                                filename = analyzer.export_to_excel()
                                with open(filename, 'rb') as f:
                                    st.download_button(
                                        "⬇️ Download Excel",
                                        f,
                                        filename,
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="dl_excel"
                                    )
                                st.success(f"✅ Report ready!")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                else:
                    st.warning("openpyxl not installed")
            
            with col2:
                st.markdown("#### 🔗 JSON Export")
                st.markdown("Portfolio configuration and metrics in JSON format.")
                if st.button("📥 Generate JSON", use_container_width=True, key="export_json"):
                    export_data = {
                        'metadata': {
                            'generated': datetime.now().isoformat(),
                            'assets': [{'ticker': t, 'name': get_display_name(t)} for t in symbols],
                            'period': {
                                'start': str(st.session_state.form_start_date),
                                'end': str(st.session_state.form_end_date)
                            },
                            'risk_free_rate': risk_free_rate
                        },
                        'portfolios': {}
                    }
                    for name, p in analyzer.portfolios.items():
                        export_data['portfolios'][name] = {
                            'name': p['name'],
                            'weights': {get_display_name(s): round(float(w)*100, 2) for s, w in zip(symbols, p['weights']) if w > 0.001},
                            'metrics': {
                                'annual_return': round(p['annualized_return']*100, 2),
                                'volatility': round(p['annualized_volatility']*100, 2),
                                'sharpe_ratio': round(p['sharpe_ratio'], 3),
                                'max_drawdown': round(p['max_drawdown']*100, 2)
                            }
                        }
                    st.download_button(
                        "⬇️ Download JSON",
                        json.dumps(export_data, indent=2),
                        f"portfolio_{datetime.now().strftime('%Y%m%d')}.json",
                        "application/json",
                        key="dl_json"
                    )

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-section">
    <h3>📊 PORTFOLIO ANALYZER PRO</h3>
    <p><strong>Powered by:</strong> Python • Streamlit • Plotly • yfinance • NumPy • SciPy</p>
    <p style="margin-top:1rem;opacity:0.6;">⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
