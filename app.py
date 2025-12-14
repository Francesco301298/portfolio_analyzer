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
import requests

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
    "🛢️ Commodities": ["GLD","IAU","SLV","GDX","GDXJ","USO","UNG","DBA","DBC","PDBC","CPER","WEAT","CORN","SOYB"]
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

# ============ BACKTEST VALIDATION FUNCTIONS ============

def calculate_portfolio_metrics(returns, rf_rate=0.02):
    """Calculate comprehensive portfolio metrics from returns series."""
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else 0
    
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = (ann_return - rf_rate) / downside_std if downside_std > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    var_5 = np.percentile(returns, 5)
    cvar_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else var_5
    
    return {
        'return': ann_return, 'volatility': ann_vol, 'sharpe': sharpe, 'sortino': sortino,
        'max_drawdown': max_dd, 'calmar': calmar, 'var_5': var_5, 'cvar_5': cvar_5,
        'cumulative': (cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0
    }

def optimize_portfolio_weights(returns_df, method='max_sharpe', rf_rate=0.02):
    """Optimize portfolio weights using specified method."""
    expected_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    n = len(returns_df.columns)
    
    eigenvalues = np.linalg.eigvals(cov_matrix)
    if np.any(eigenvalues <= 0):
        cov_matrix = cov_matrix + np.eye(n) * 1e-8
    
    if method == 'equal':
        return np.array([1/n] * n)
    elif method == 'min_vol':
        def vol(w): return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(vol, [1/n]*n, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else np.array([1/n]*n)
    elif method == 'max_sharpe':
        def neg_sharpe(w):
            ret = np.dot(w, expected_returns)
            vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(ret - rf_rate) / vol if vol > 0 else 0
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(neg_sharpe, [1/n]*n, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else np.array([1/n]*n)
    elif method == 'risk_parity':
        def risk_contrib_error(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if port_vol == 0: return 1e10
            marginal = np.dot(cov_matrix, w)
            risk_contrib = w * marginal / port_vol
            target = port_vol / n
            return np.sum((risk_contrib - target) ** 2)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.001, 1) for _ in range(n))
        result = minimize(risk_contrib_error, [1/n]*n, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else np.array([1/n]*n)
    else:
        return np.array([1/n] * n)

def run_walk_forward_analysis(returns_df, train_ratio=0.7, methods=None, rf_rate=0.02):
    """Perform Walk-Forward Analysis with train/test split."""
    if methods is None:
        methods = ['equal', 'min_vol', 'max_sharpe', 'risk_parity']
    
    n_obs = len(returns_df)
    train_end = int(n_obs * train_ratio)
    
    train_returns = returns_df.iloc[:train_end]
    test_returns = returns_df.iloc[train_end:]
    
    results = {}
    for method in methods:
        weights = optimize_portfolio_weights(train_returns, method=method, rf_rate=rf_rate)
        train_port_returns = train_returns.dot(weights)
        test_port_returns = test_returns.dot(weights)
        train_metrics = calculate_portfolio_metrics(train_port_returns, rf_rate)
        test_metrics = calculate_portfolio_metrics(test_port_returns, rf_rate)
        stability_ratio = test_metrics['sharpe'] / train_metrics['sharpe'] if train_metrics['sharpe'] != 0 else 0
        
        results[method] = {
            'weights': weights, 'train_metrics': train_metrics, 'test_metrics': test_metrics,
            'train_returns': train_port_returns, 'test_returns': test_port_returns, 'stability_ratio': stability_ratio
        }
    return results, train_end

# Deep-dive Statistics Functions
def compute_autocorrelation(x, max_lag):
    n = len(x)
    x_centered = x - np.mean(x)
    var_x = np.var(x)
    acf = []
    for lag in range(1, max_lag + 1):
        if lag < n:
            acf_val = np.sum(x_centered[lag:] * x_centered[:-lag]) / ((n - lag) * var_x)
            acf.append(acf_val)
        else:
            acf.append(0)
    return np.array(acf)

def invariance_test_ellipsoid(eps, l_bar, conf_lev=0.95):
    t_bar = len(eps)
    acf = compute_autocorrelation(eps, l_bar)
    conf_int = stats.norm.ppf((1 + conf_lev) / 2) / np.sqrt(t_bar)
    test_passed = np.all(np.abs(acf) < conf_int)
    return acf, conf_int, test_passed

def ks_test(eps):
    eps_std = (eps - np.mean(eps)) / np.std(eps)
    ks_stat, p_value = stats.kstest(eps_std, 'norm')
    return ks_stat, p_value

def fit_garch(returns):
    if not ARCH_AVAILABLE:
        return None, None, None
    try:
        model = arch_model(returns, vol='garch', p=1, o=0, q=1, rescale=False)
        result = model.fit(disp='off')
        return result.params, result.std_resid, result.conditional_volatility
    except:
        return None, None, None

# Session State
defaults = {
    'analyzer': None, 'analysis_complete': False, 'selected_tickers': [],
    'benchmark': '^GSPC', 'use_benchmark': True, 'benchmark_returns': None,
    'run_analysis': False, 'alerts': [], 'yf_search_results': [], 'yf_selected': []
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
    
    st.markdown("---")
    st.markdown("#### 📅 Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2018, 1, 1), min_value=datetime(2000, 1, 1), key="start")
    with col2:
        end_date = st.date_input("End Date", value=datetime.now(), key="end")
    
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1, key="rf")
    window_years = st.slider("Rolling Window (years)", 1, 10, 3, key="window")  # Extended to 10 years
    
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
            <h3>🎯 Quick Start</h3>
            <ol><li>Select assets</li><li>Configure parameters</li><li>Click Start Analysis</li><li>Explore results</li></ol>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='dashboard-card'>
            <h3>🔍 Yahoo Search</h3>
            <p>Search <strong>any ticker</strong> worldwide: stocks, ETFs, crypto, indices!</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='dashboard-card'>
            <h3>🧪 NEW: Backtest</h3>
            <p>Walk-Forward Analysis with Train/Test validation to detect overfitting!</p>
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
# Part 3: Analysis Logic

if st.session_state.run_analysis or st.session_state.analyzer is not None:
    
    if st.session_state.run_analysis and st.session_state.analyzer is None:
        symbols = st.session_state.selected_tickers
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing...")
            progress_bar.progress(10)
            
            from portfolio_analyzer import AdvancedPortfolioAnalyzer
            
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
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
                "📊 Overview", "💼 Portfolios", "📈 Performance", "🔄 Rolling",
                "🔬 Deep-dive", "🧪 Backtest", "📐 Frontier", "🎯 Benchmark", "📥 Export"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Overview", "💼 Portfolios", "📈 Performance", "🔄 Rolling",
                "🔬 Deep-dive", "🧪 Backtest", "📐 Frontier", "📥 Export"
            ])
            tab9 = None
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
            
            st.markdown("#### 📊 Strategy Comparison")
            
            comparison_data = []
            for idx, (key, p) in enumerate(sorted(analyzer.portfolios.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True), 1):
                comparison_data.append({
                    '#': idx, 'Strategy': p['name'], 'Return': f"{p['annualized_return']*100:.2f}%",
                    'Vol': f"{p['annualized_volatility']*100:.2f}%", 'Sharpe': f"{p['sharpe_ratio']:.3f}",
                    'Sortino': f"{p['sortino_ratio']:.3f}", 'Max DD': f"{p['max_drawdown']*100:.2f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.markdown(create_styled_table(comparison_df, "Ranked by Sharpe Ratio"), unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Returns by Strategy")
                port_data = sorted(analyzer.portfolios.values(), key=lambda x: x['annualized_return'], reverse=True)
                fig = go.Figure(data=[go.Bar(
                    x=[p['name'] for p in port_data], y=[p['annualized_return'] * 100 for p in port_data],
                    marker_color=CHART_COLORS[:len(port_data)],
                    text=[f"{p['annualized_return']*100:.1f}%" for p in port_data],
                    textposition='outside', textfont=dict(color='#E2E8F0', size=9)
                )])
                fig.update_layout(height=380, xaxis_tickangle=-45, yaxis_title="Return (%)")
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### ⚖️ Risk vs Return")
                fig = go.Figure()
                for i, p in enumerate(analyzer.portfolios.values()):
                    fig.add_trace(go.Scatter(
                        x=[p['annualized_volatility'] * 100], y=[p['annualized_return'] * 100],
                        mode='markers+text', name=p['name'],
                        marker=dict(size=16, color=CHART_COLORS[i % len(CHART_COLORS)]),
                        text=[p['name'].split()[0]], textposition='top center', textfont=dict(color='#E2E8F0', size=8),
                        hovertemplate=f"<b>{p['name']}</b><br>Return: %{{y:.2f}}%<br>Vol: %{{x:.2f}}%<extra></extra>"
                    ))
                fig.update_layout(height=380, xaxis_title="Volatility (%)", yaxis_title="Return (%)", showlegend=False)
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown("#### 🔍 Portfolio Details")
            
            portfolio_keys = list(analyzer.portfolios.keys())
            selected_p = st.selectbox("Select strategy", portfolio_keys, format_func=lambda x: analyzer.portfolios[x]['name'], key="port_detail")
            portfolio = analyzer.portfolios[selected_p]
            
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
            
            weights_data = []
            for ticker, weight in zip(symbols, portfolio['weights']):
                if weight > 0.001:
                    weights_data.append({'Asset': get_display_name(ticker), 'Ticker': ticker, 'Weight': weight * 100})
            weights_df = pd.DataFrame(weights_data).sort_values('Weight', ascending=False)
            
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                table_data = [{'Asset': r['Asset'], 'Weight': f"{r['Weight']:.2f}%"} for _, r in weights_df.iterrows()]
                st.markdown(create_styled_table(pd.DataFrame(table_data)), unsafe_allow_html=True)
            
            with col2:
                fig = go.Figure(data=[go.Pie(
                    labels=weights_df['Asset'], values=weights_df['Weight'], hole=0.4,
                    marker_colors=CHART_COLORS[:len(weights_df)], textinfo='percent', textposition='outside',
                    textfont=dict(color='#E2E8F0', size=10),
                    hovertemplate='<b>%{label}</b><br>%{value:.2f}%<extra></extra>'
                )])
                fig.update_layout(height=320, showlegend=False,
                    annotations=[dict(text=portfolio['name'].split()[0], x=0.5, y=0.5, font_size=11, font_color='#E2E8F0', showarrow=False)])
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
# Part 5: Performance, Rolling, and Deep-dive Statistics Tabs

        # TAB 3: PERFORMANCE
        with tab3:
            st.markdown("### 📈 Performance Comparison")
            
            portfolio_keys = list(analyzer.portfolios.keys())
            sel_perf = st.multiselect("Select strategies", portfolio_keys, default=portfolio_keys[:4], format_func=lambda x: analyzer.portfolios[x]['name'], key="perf_sel")
            
            if sel_perf:
                st.markdown("#### 📊 Cumulative Performance")
                fig = go.Figure()
                for i, p_name in enumerate(sel_perf):
                    p = analyzer.portfolios[p_name]
                    cum_val = (1 + p['returns']).cumprod() * 100
                    fig.add_trace(go.Scatter(x=cum_val.index, y=cum_val.values, name=p['name'], mode='lines',
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
                        hovertemplate=f'<b>{p["name"]}</b><br>Value: %{{y:.2f}}<extra></extra>'))
                fig.update_layout(height=420, hovermode='x unified', xaxis_title="Date", yaxis_title="Value (Base 100)",
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 📉 Drawdown Analysis")
                fig = go.Figure()
                for i, p_name in enumerate(sel_perf):
                    p = analyzer.portfolios[p_name]
                    cum_val = (1 + p['returns']).cumprod()
                    roll_max = cum_val.expanding().max()
                    dd = (cum_val - roll_max) / roll_max * 100
                    color = CHART_COLORS[i % len(CHART_COLORS)]
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=p['name'], mode='lines',
                        line=dict(color=color, width=2), fill='tozeroy', fillcolor=f'rgba({r},{g},{b},0.2)',
                        hovertemplate=f'<b>{p["name"]}</b><br>DD: %{{y:.2f}}%<extra></extra>'))
                fig.update_layout(height=320, hovermode='x unified', xaxis_title="Date", yaxis_title="Drawdown (%)",
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Select at least one strategy")
        
        # TAB 4: ROLLING
        with tab4:
            st.markdown("### 🔄 Rolling Analysis")
            st.info(f"📊 Window: **{window_years} years** ({window_years * 252} days)")
            
            portfolio_keys = list(analyzer.portfolios.keys())
            sel_roll = st.multiselect("Select strategies", portfolio_keys, default=portfolio_keys[:3], format_func=lambda x: analyzer.portfolios[x]['name'], key="roll_sel")
            
            if sel_roll:
                rolling_data = {}
                window = window_years * 252
                for p_name in sel_roll:
                    p = analyzer.portfolios[p_name]
                    rets = p['returns']
                    if len(rets) >= window:
                        roll_ret = rets.rolling(window=window).apply(lambda x: (1+x).prod()**(252/len(x))-1 if len(x)==window else np.nan)
                        roll_vol = rets.rolling(window=window).std() * np.sqrt(252)
                        roll_sharpe = (roll_ret - rf_rate) / roll_vol
                        rolling_data[p['name']] = pd.DataFrame({'Return': roll_ret, 'Volatility': roll_vol, 'Sharpe': roll_sharpe}).dropna()
                
                if rolling_data:
                    st.markdown("#### 📈 Rolling Returns")
                    fig = go.Figure()
                    for i, (name, data) in enumerate(rolling_data.items()):
                        fig.add_trace(go.Scatter(x=data.index, y=data['Return']*100, name=name,
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5)))
                    fig.update_layout(height=350, hovermode='x unified', yaxis_title="Return (%)",
                                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### ⭐ Rolling Sharpe")
                    fig = go.Figure()
                    for i, (name, data) in enumerate(rolling_data.items()):
                        fig.add_trace(go.Scatter(x=data.index, y=data['Sharpe'], name=name,
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5)))
                    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    fig.update_layout(height=350, hovermode='x unified', yaxis_title="Sharpe Ratio",
                                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Not enough data for rolling window")
            else:
                st.warning("⚠️ Select at least one strategy")
        
        # TAB 5: DEEP-DIVE STATISTICS (NEW)
        with tab5:
            st.markdown("### 🔬 Deep-dive Statistics")
            st.markdown("Statistical analysis of individual asset risk drivers using log-returns and invariance tests.")
            
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
                
                # Section 1: Risk Driver Time Series
                st.markdown(f"#### 📈 Risk Driver: Log-values of {get_display_name(selected_asset)}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates_idx, y=x_stock, mode='lines',
                        line=dict(color=CHART_COLORS[0], width=2),
                        hovertemplate='Date: %{x}<br>Log-value: %{y:.4f}<extra></extra>'))
                    fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Log-values")
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Basic statistics
                    st.markdown("##### 📊 Summary Statistics")
                    stats_data = {
                        'Metric': ['Mean Log-Return', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max'],
                        'Value': [
                            f"{np.mean(delta_x):.6f}",
                            f"{np.std(delta_x):.6f}",
                            f"{stats.skew(delta_x):.4f}",
                            f"{stats.kurtosis(delta_x):.4f}",
                            f"{np.min(delta_x):.4f}",
                            f"{np.max(delta_x):.4f}"
                        ]
                    }
                    st.markdown(create_styled_table(pd.DataFrame(stats_data)), unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Section 2: Log-Returns (Compounded Returns)
                st.markdown("#### 📉 Compounded Returns (Log-Returns)")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates_idx[1:], y=delta_x, mode='markers',
                    marker=dict(size=3, color=CHART_COLORS[1], opacity=0.7),
                    hovertemplate='Date: %{x}<br>Return: %{y:.4f}<extra></extra>'))
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Log-Return")
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Section 3: Invariance Tests
                st.markdown("#### 🧪 Invariance Tests")
                st.markdown("Testing if log-returns are IID (independent and identically distributed).")
                
                l_bar = st.slider("Maximum lag for autocorrelation", 5, 50, 25, key="lag_slider")
                conf_lev = 0.95
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Ellipsoid test on returns
                    st.markdown("##### Autocorrelation of Returns")
                    acf, conf_int, test_passed = invariance_test_ellipsoid(delta_x, l_bar, conf_lev)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(range(1, l_bar+1)), y=acf,
                        marker_color=CHART_COLORS[2], name='ACF'))
                    fig.add_hline(y=conf_int, line_dash="dash", line_color="#FF6B6B", 
                                 annotation_text=f"+{conf_lev*100:.0f}% CI")
                    fig.add_hline(y=-conf_int, line_dash="dash", line_color="#FF6B6B",
                                 annotation_text=f"-{conf_lev*100:.0f}% CI")
                    fig.add_hline(y=0, line_color="rgba(255,255,255,0.5)")
                    fig.update_layout(height=300, xaxis_title="Lag", yaxis_title="Autocorrelation",
                                     yaxis_range=[-0.3, 0.3])
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if test_passed:
                        st.success("✅ Returns appear IID (all ACF within confidence bands)")
                    else:
                        st.warning("⚠️ Some autocorrelations exceed confidence bands")
                
                with col2:
                    # Ellipsoid test on absolute returns (volatility clustering)
                    st.markdown("##### Autocorrelation of Absolute Returns (Volatility Clustering)")
                    acf_abs, conf_int_abs, test_passed_abs = invariance_test_ellipsoid(delta_x_abs, l_bar, conf_lev)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(range(1, l_bar+1)), y=acf_abs,
                        marker_color=CHART_COLORS[3], name='ACF |r|'))
                    fig.add_hline(y=conf_int_abs, line_dash="dash", line_color="#FF6B6B")
                    fig.add_hline(y=-conf_int_abs, line_dash="dash", line_color="#FF6B6B")
                    fig.add_hline(y=0, line_color="rgba(255,255,255,0.5)")
                    fig.update_layout(height=300, xaxis_title="Lag", yaxis_title="Autocorrelation",
                                     yaxis_range=[-0.1, 0.4])
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if test_passed_abs:
                        st.success("✅ No volatility clustering detected")
                    else:
                        st.info("ℹ️ Volatility clustering detected")
                
                st.markdown("---")
                
                # Section 4: Distribution Analysis
                st.markdown("#### 📊 Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram with normal overlay
                    st.markdown("##### Return Distribution vs Normal")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=delta_x, nbinsx=50, name='Returns',
                        marker_color=CHART_COLORS[4], opacity=0.7, histnorm='probability density'))
                    
                    # Normal overlay
                    x_range = np.linspace(delta_x.min(), delta_x.max(), 100)
                    normal_pdf = stats.norm.pdf(x_range, np.mean(delta_x), np.std(delta_x))
                    fig.add_trace(go.Scatter(x=x_range, y=normal_pdf, mode='lines',
                        name='Normal', line=dict(color='#FFFFFF', width=2)))
                    
                    fig.update_layout(height=300, xaxis_title="Log-Return", yaxis_title="Density",
                                     legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Kolmogorov-Smirnov test
                    st.markdown("##### Normality Tests")
                    
                    ks_stat, ks_pval = ks_test(delta_x)
                    
                    # Jarque-Bera test
                    jb_stat, jb_pval = stats.jarque_bera(delta_x)
                    
                    # Shapiro-Wilk (on subsample if too large)
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
                    
                    st.markdown("*p-value > 0.05 suggests normality cannot be rejected*")
                
                st.markdown("---")
                
                # Section 5: GARCH Analysis (if available)
                if ARCH_AVAILABLE:
                    st.markdown("#### ⚡ GARCH(1,1) Analysis")
                    st.markdown("Modeling conditional volatility with GARCH(1,1)")
                    
                    with st.spinner("Fitting GARCH model..."):
                        params, std_resid, cond_vol = fit_garch(delta_x)
                    
                    if params is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Model Parameters")
                            param_data = {
                                'Parameter': ['μ (mean)', 'ω (constant)', 'α (ARCH)', 'β (GARCH)', 'Persistence (α+β)'],
                                'Value': [
                                    f"{params['mu']:.6f}",
                                    f"{params['omega']:.8f}",
                                    f"{params['alpha[1]']:.4f}",
                                    f"{params['beta[1]']:.4f}",
                                    f"{params['alpha[1]'] + params['beta[1]']:.4f}"
                                ]
                            }
                            st.markdown(create_styled_table(pd.DataFrame(param_data)), unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("##### Conditional Volatility")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=dates_idx[1:], y=cond_vol*np.sqrt(252)*100,
                                mode='lines', line=dict(color=CHART_COLORS[5], width=1.5),
                                hovertemplate='Date: %{x}<br>Ann. Vol: %{y:.2f}%<extra></extra>'))
                            fig.update_layout(height=250, xaxis_title="Date", yaxis_title="Annualized Vol (%)")
                            fig = apply_plotly_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Standardized residuals
                        st.markdown("##### Standardized Residuals (Invariants)")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=dates_idx[1:], y=std_resid, mode='markers',
                                marker=dict(size=2, color=CHART_COLORS[6], opacity=0.6)))
                            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                            fig.update_layout(height=250, xaxis_title="Date", yaxis_title="Std. Residual")
                            fig = apply_plotly_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # ACF of squared standardized residuals
                            acf_resid2, conf_resid, _ = invariance_test_ellipsoid(std_resid**2, 20, 0.95)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=list(range(1, 21)), y=acf_resid2,
                                marker_color=CHART_COLORS[7], name='ACF ε²'))
                            fig.add_hline(y=conf_resid, line_dash="dash", line_color="#FF6B6B")
                            fig.add_hline(y=-conf_resid, line_dash="dash", line_color="#FF6B6B")
                            fig.update_layout(height=250, xaxis_title="Lag", yaxis_title="ACF of ε²",
                                             yaxis_range=[-0.15, 0.15])
                            fig = apply_plotly_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Could not fit GARCH model to this data")
                else:
                    st.info("ℹ️ Install `arch` package for GARCH analysis: `pip install arch`")
# Part 7: Backtest Validation Tab

        # TAB 6: BACKTEST VALIDATION
        with tab6:
            st.markdown("### 🧪 Backtest Validation (AFML – Combinatorial Purged CV)")
            st.markdown("""
            This validation framework follows **López de Prado (2018, Ch.7)**:
            
            - **Combinatorial Purged Cross-Validation**
            - **Embargo to prevent information leakage**
            - **Out-of-Sample performance distributions**
            - **Probability of Backtest Overfitting (PBO)**

            This tab is designed for **research-grade strategy validation**, not point-estimate backtests.
            """)
            st.markdown("---")

            # ==========================
            # CONFIGURATION
            # ==========================
            col1, col2 = st.columns(2)
            with col1:
                n_splits = st.selectbox("Number of folds (K)", [5, 6, 8], index=0)
            with col2:
                n_test_splits = st.selectbox("Test folds per split", [1, 2], index=1)

            embargo_pct = st.slider(
                "Embargo (% of dataset)",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.25
            ) / 100

            methods_to_test = st.multiselect(
                "Strategies",
                ["equal", "min_vol", "max_sharpe", "risk_parity"],
                default=["equal", "min_vol", "max_sharpe", "risk_parity"]
            )

            method_names = {
                "equal": "Equally Weighted",
                "min_vol": "Minimum Volatility",
                "max_sharpe": "Maximum Sharpe",
                "risk_parity": "Risk Parity"
            }

            # ==========================
            # CORE FUNCTIONS (AFML)
            # ==========================
            from itertools import combinations

            def combinatorial_purged_cv(dates, n_splits, n_test_splits, embargo_pct):
                fold_size = len(dates) // n_splits
                folds = [dates[i*fold_size:(i+1)*fold_size] for i in range(n_splits)]

                splits = []
                for test_folds in combinations(range(n_splits), n_test_splits):
                    test_idx = pd.Index([])
                    for f in test_folds:
                        test_idx = test_idx.append(folds[f])

                    train_idx = dates.difference(test_idx)

                    # Embargo
                    embargo = int(len(dates) * embargo_pct)
                    if embargo > 0:
                        test_start = test_idx.min()
                        embargo_dates = dates[(dates >= test_start)][:embargo]
                        train_idx = train_idx.difference(embargo_dates)

                    splits.append((train_idx.sort_values(), test_idx.sort_values()))

                return splits


            def run_cpcv_backtest(
                returns_df,
                methods,
                rf_rate,
                n_splits=5,
                n_test_splits=2,
                embargo_pct=0.01
            ):
                splits = combinatorial_purged_cv(
                    returns_df.index,
                    n_splits,
                    n_test_splits,
                    embargo_pct
                )

                oos_sharpes = {m: [] for m in methods}
                oos_returns = {m: [] for m in methods}

                for train_idx, test_idx in splits:
                    train_returns = returns_df.loc[train_idx]
                    test_returns = returns_df.loc[test_idx]

                    for method in methods:
                        # 🔑 USE EXISTING OPTIMIZER
                        weights = optimize_portfolio_weights(
                            train_returns,
                            method=method,
                            rf_rate=rf_rate
                        )

                        test_portfolio_returns = test_returns.dot(weights)

                        # 🔑 USE EXISTING METRICS FUNCTION
                        metrics = calculate_portfolio_metrics(
                            test_portfolio_returns,
                            rf_rate=rf_rate
                        )

                        oos_returns[method].append(test_portfolio_returns)
                        oos_sharpes[method].append(metrics['sharpe'])

                return oos_sharpes, oos_returns


            def compute_pbo(oos_sharpes):
                sharpe_df = pd.DataFrame(oos_sharpes)

                # Rank strategies per split
                ranks = sharpe_df.rank(axis=1, ascending=False)
                best_is = ranks.idxmin(axis=1)

                oos_rank = np.array([
                    ranks.loc[i, best_is[i]] for i in ranks.index
                ])

                logit = np.log(oos_rank / (len(oos_sharpes) - oos_rank))
                pbo = np.mean(logit < 0)

                return pbo

            # ==========================
            # RUN BACKTEST
            # ==========================
            if st.button("🚀 Run AFML Validation", use_container_width=True):
                if not methods_to_test:
                    st.error("Select at least one strategy.")
                else:
                    with st.spinner("Running Combinatorial Purged Cross-Validation..."):
                        oos_sharpes, oos_returns = run_cpcv_backtest(
                            analyzer.returns,
                            methods_to_test,
                            rf_rate,
                            n_splits,
                            n_test_splits,
                            embargo_pct
                        )

                        pbo = compute_pbo(oos_sharpes)

                    st.success("✅ Validation completed")
                    st.markdown("---")

                    # ==========================
                    # PBO METRIC
                    # ==========================
                    st.markdown("#### 📉 Probability of Backtest Overfitting")
                    st.metric(
                        "PBO",
                        f"{pbo:.2%}",
                        help="Probability that the best in-sample strategy underperforms out-of-sample."
                    )
                    st.markdown("---")

                    # ==========================
                    # OOS SHARPE DISTRIBUTIONS
                    # ==========================
                    st.markdown("#### 📊 Out-of-Sample Sharpe Distributions")

                    fig = go.Figure()
                    for i, (method, sharpes) in enumerate(oos_sharpes.items()):
                        fig.add_trace(go.Box(
                            y=sharpes,
                            name=method_names.get(method, method),
                            boxmean=True,
                            marker_color=CHART_COLORS[i % len(CHART_COLORS)]
                        ))

                    fig.update_layout(
                        height=400,
                        yaxis_title="Sharpe Ratio",
                        showlegend=False
                    )
                    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

                    st.markdown("---")

                    # ==========================
                    # OOS RETURN DISTRIBUTIONS
                    # ==========================
                    st.markdown("#### 📈 Out-of-Sample Return Distributions")

                    fig = go.Figure()
                    for i, (method, ret_list) in enumerate(oos_returns.items()):
                        flat_returns = pd.concat(ret_list)
                        fig.add_trace(go.Histogram(
                            x=flat_returns.values * 100,
                            name=method_names.get(method, method),
                            opacity=0.65,
                            marker_color=CHART_COLORS[i % len(CHART_COLORS)]
                        ))

                    fig.update_layout(
                        height=350,
                        barmode="overlay",
                        xaxis_title="Daily Return (%)"
                    )
                    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

                    st.markdown("---")

                    # ==========================
                    # ROBUST RANKING (MEDIAN OOS)
                    # ==========================
                    st.markdown("#### 🏆 Robust Out-of-Sample Ranking")

                    ranking_data = []
                    for method, sharpes in oos_sharpes.items():
                        ranking_data.append({
                            "Strategy": method_names.get(method, method),
                            "Median OOS Sharpe": np.median(sharpes),
                            "Mean OOS Sharpe": np.mean(sharpes),
                            "Std OOS Sharpe": np.std(sharpes)
                        })

                    ranking_df = pd.DataFrame(ranking_data).sort_values(
                        "Median OOS Sharpe", ascending=False
                    )

                    st.markdown(
                        create_styled_table(ranking_df.round(3)),
                        unsafe_allow_html=True
                    )

                    st.info(
                        "Ranking is based on **median OOS Sharpe**, not on a single backtest.\n\n"
                        "This reduces selection bias and aligns with AFML best practices."
                    )
                    # ============================================================
                    # 🧠 INTERPRETATION & FINAL RECAP (AFML STYLE)
                    # ============================================================

                    st.markdown("---")
                    st.markdown("## 🧠 Interpretation & Final Recap")

                    # ==========================
                    # BUILD SUMMARY STATISTICS
                    # ==========================
                    summary_rows = []
                    for method, sharpes in oos_sharpes.items():
                        summary_rows.append({
                            "Strategy": method_names.get(method, method),
                            "Median OOS Sharpe": np.median(sharpes),
                            "Mean OOS Sharpe": np.mean(sharpes),
                            "Std OOS Sharpe": np.std(sharpes),
                            "Min OOS Sharpe": np.min(sharpes),
                            "Max OOS Sharpe": np.max(sharpes)
                        })

                    summary_df = (
                        pd.DataFrame(summary_rows)
                        .sort_values("Median OOS Sharpe", ascending=False)
                    )

                    st.markdown(
                        create_styled_table(summary_df.round(3)),
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                    **How to read this table:**
                    - **Median OOS Sharpe** → typical performance across time regimes (most important)
                    - **Std OOS Sharpe** → stability (lower = more robust)
                    - **Min / Max** → tail risk and lucky runs
                    """
                    )

                    # ==========================
                    # PBO INTERPRETATION
                    # ==========================
                    st.markdown("### 📉 Overfitting Assessment")

                    if pbo < 0.1:
                        pbo_msg = "🟢 **Very low probability of overfitting**. The validation results are highly reliable."
                    elif pbo < 0.3:
                        pbo_msg = "🟡 **Moderate overfitting risk**. Results are acceptable but should be monitored."
                    elif pbo < 0.5:
                        pbo_msg = "🟠 **High risk of overfitting**. Strategy selection may not be stable."
                    else:
                        pbo_msg = "🔴 **Severe overfitting detected**. Apparent performance is likely due to chance."

                    st.markdown(f"""
                    **Probability of Backtest Overfitting (PBO): `{pbo:.2%}`**

                    {pbo_msg}

                    **Interpretation**  
                    PBO answers the question:

                    > *"How often does the strategy that looks best in-sample fail out-of-sample?"*

                    Lower is better. Values above **30%** should be treated with caution.
                    """)

                    # ==========================
                    # ROBUST STRATEGY ANALYSIS
                    # ==========================
                    best_strategy = summary_df.iloc[0]
                    worst_strategy = summary_df.iloc[-1]

                    st.markdown("### 🏆 Robustness Analysis")

                    st.markdown(f"""
                    - **Most robust strategy**: **{best_strategy['Strategy']}**
                    - Median OOS Sharpe: **{best_strategy['Median OOS Sharpe']:.2f}**
                    - Dispersion (Std): **{best_strategy['Std OOS Sharpe']:.2f}**
                    - Indicates **consistent performance across multiple market regimes**.

                    - **Least robust strategy**: **{worst_strategy['Strategy']}**
                    - Median OOS Sharpe: **{worst_strategy['Median OOS Sharpe']:.2f}**
                    - High variability suggests **sensitivity to regime and parameter instability**.
                    """)

                    # ==========================
                    # FINAL TAKEAWAY
                    # ==========================
                    st.markdown("### 🎯 Final Takeaway")

                    st.info("""
                    This validation **does not aim to identify the single best strategy**.

                    Instead, it evaluates:
                    - **Robustness to time variation**
                    - **Stability of performance**
                    - **Reliability of the selection process**

                    **Best practice (AFML):**
                    - Prefer strategies with **high median OOS Sharpe**
                    - Penalize high dispersion
                    - Avoid strategy selection if **PBO > 30%**
                    - Treat extreme Sharpe values as **potential overfitting**
                    """)
# Part 8: Frontier, Benchmark, Export, Footer

        # TAB 7: FRONTIER
        with tab7:
            st.markdown("### 📐 Efficient Frontier")
            
            with st.spinner("Calculating..."):
                frontier_df = analyzer.get_efficient_frontier(n_points=50)
                
                if not frontier_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=frontier_df['Volatility']*100, y=frontier_df['Return']*100, mode='lines', name='Frontier',
                        line=dict(color='#FFE66D', width=4), hovertemplate='Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'))
                    
                    for i, (p_name, p) in enumerate(analyzer.portfolios.items()):
                        fig.add_trace(go.Scatter(x=[p['annualized_volatility']*100], y=[p['annualized_return']*100], mode='markers+text', name=p['name'],
                            marker=dict(size=14, color=CHART_COLORS[i % len(CHART_COLORS)], symbol='diamond', line=dict(width=2, color='white')),
                            text=[p['name'].split()[0]], textposition='top center', textfont=dict(color='#E2E8F0', size=9),
                            hovertemplate=f"<b>{p['name']}</b><br>Return: %{{y:.2f}}%<br>Vol: %{{x:.2f}}%<extra></extra>"))
                    
                    max_sharpe_p = max(analyzer.portfolios.values(), key=lambda x: x['sharpe_ratio'])
                    cml_x = [0, max_sharpe_p['annualized_volatility']*100*2.5]
                    cml_y = [rf_rate*100, rf_rate*100 + max_sharpe_p['sharpe_ratio']*cml_x[1]]
                    fig.add_trace(go.Scatter(x=cml_x, y=cml_y, mode='lines', name='CML', line=dict(color='#4ECDC4', width=2, dash='dash'), hoverinfo='skip'))
                    
                    fig.update_layout(height=500, xaxis_title="Volatility (%)", yaxis_title="Return (%)",
                                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📉 Min Vol", f"{frontier_df['Volatility'].min()*100:.2f}%")
                    col2.metric("📈 Max Return", f"{frontier_df['Return'].max()*100:.2f}%")
                    col3.metric("⭐ Max Sharpe", f"{frontier_df['Sharpe'].max():.3f}")
        
        # TAB 8: BENCHMARK (if available)
        if tab9 is not None:
            with tab8:
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
        export_tab = tab9 if tab9 is not None else tab8
        
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
