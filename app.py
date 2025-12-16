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
            
            # ============ NEW: SEASONALITY ANALYSIS ============
            st.markdown("---")
            st.markdown("## 📅 Seasonality Analysis")
            st.markdown("Analyze how the selected portfolio performs across different months and years to identify seasonal patterns.")
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                # Get available years
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
            
            # Determine which years to use
            if selected_years:
                years_to_analyze = sorted(selected_years)
            else:
                years_to_analyze = available_years[:n_years_display]
            
            # Filter returns for selected years
            mask = portfolio_returns.index.year.isin(years_to_analyze)
            filtered_returns = portfolio_returns[mask]
            
            if len(filtered_returns) > 20:
                # Create DataFrame with Year and Month
                seasonality_df = pd.DataFrame({
                    'Return': filtered_returns.values,
                    'Date': filtered_returns.index,
                    'Year': filtered_returns.index.year,
                    'Month': filtered_returns.index.month
                })
                
                # ---- CHART 1: Cumulative Performance by Year (Base 100) ----
                st.markdown("#### 📈 Year-over-Year Performance (Base 100)")
                
                fig_yearly = go.Figure()
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                for i, year in enumerate(sorted(years_to_analyze, reverse=True)):
                    year_data = seasonality_df[seasonality_df['Year'] == year].copy()
                    if len(year_data) > 0:
                        # Calculate cumulative return starting at 100
                        year_data = year_data.sort_values('Date')
                        year_data['Cumulative'] = (1 + year_data['Return']).cumprod() * 100
                        
                        # Create day-of-year index for alignment
                        year_data['DayOfYear'] = year_data['Date'].dt.dayofyear
                        
                        fig_yearly.add_trace(go.Scatter(
                            x=year_data['DayOfYear'],
                            y=year_data['Cumulative'],
                            name=str(year),
                            mode='lines',
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                            hovertemplate=f'<b>{year}</b><br>Day: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                        ))
                
                # Add month labels on x-axis
                month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
                
                fig_yearly.update_layout(
                    height=450,
                    xaxis=dict(
                        tickmode='array',
                        tickvals=month_starts,
                        ticktext=month_names,
                        title="Month"
                    ),
                    yaxis_title="Cumulative Value (Base 100)",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig_yearly.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_yearly = apply_plotly_theme(fig_yearly)
                st.plotly_chart(fig_yearly, use_container_width=True)
                
                # ---- TABLE: Monthly Returns by Year ----
                st.markdown("#### 📊 Monthly Returns Table (%)")
                
                # Calculate average daily return per month/year, then approximate monthly return
                monthly_stats = seasonality_df.groupby(['Year', 'Month']).agg(
                    avg_daily_return=('Return', 'mean'),
                    trading_days=('Return', 'count')
                ).reset_index()
                
                # Approximate monthly return: (1 + avg_daily)^trading_days - 1
                monthly_stats['Monthly_Return'] = ((1 + monthly_stats['avg_daily_return']) ** monthly_stats['trading_days'] - 1) * 100
                
                # Pivot table
                monthly_pivot = monthly_stats.pivot(index='Month', columns='Year', values='Monthly_Return')
                
                # Add month names
                month_name_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                                 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                                 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                monthly_pivot.index = monthly_pivot.index.map(month_name_map)
                
                # Calculate row averages and add as column
                monthly_pivot['Average'] = monthly_pivot.mean(axis=1)
                
                # Format values
                monthly_pivot_display = monthly_pivot.copy()
                for col in monthly_pivot_display.columns:
                    monthly_pivot_display[col] = monthly_pivot_display[col].apply(
                        lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
                    )
                
                # Reset index for display
                monthly_pivot_display = monthly_pivot_display.reset_index()
                monthly_pivot_display.columns = ['Month'] + [str(c) for c in monthly_pivot_display.columns[1:]]
                
                st.markdown(create_styled_table(monthly_pivot_display, f"Monthly Returns - {analyzer.portfolios[selected_p]['name']}"), unsafe_allow_html=True)
                
                # ---- CHART 2: Average Monthly Performance (Bar Chart) ----
                st.markdown("#### 📊 Average Monthly Performance")
                
                # Calculate average return per month across all years
                avg_monthly = monthly_stats.groupby('Month')['Monthly_Return'].mean().reset_index()
                avg_monthly['Month_Name'] = avg_monthly['Month'].map(month_name_map)
                
                # Color based on positive/negative
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
                
                fig_monthly_avg.update_layout(
                    height=350,
                    xaxis_title="Month",
                    yaxis_title="Average Monthly Return (%)",
                    xaxis_tickangle=-45
                )
                fig_monthly_avg.add_hline(y=0, line_color="rgba(255,255,255,0.5)")
                fig_monthly_avg = apply_plotly_theme(fig_monthly_avg)
                st.plotly_chart(fig_monthly_avg, use_container_width=True)
                
                # ---- Summary Statistics ----
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
                    st.metric("📊 Seasonality Strength", f"{seasonality_strength:.2f}%", 
                             "High" if seasonality_strength > 3 else "Moderate" if seasonality_strength > 1.5 else "Low")
                
                # Interpretation
                with st.expander("📖 How to Interpret Seasonality"):
                    st.markdown("""
                    **Year-over-Year Chart:**
                    - Shows how the portfolio performed throughout each year
                    - Look for consistent patterns (e.g., strong Q4, weak summer months)
                    - Divergent lines indicate high variability between years
                    
                    **Monthly Returns Table:**
                    - Green/positive values indicate gains, red/negative indicate losses
                    - The "Average" column shows the typical performance for each month
                    - Look for months that are consistently positive or negative across years
                    
                    **Average Monthly Performance:**
                    - Quick visual of which months tend to be best/worst
                    - Useful for timing decisions (with caution!)
                    
                    **Seasonality Strength:**
                    - **High (>3%)**: Strong seasonal patterns - timing may add value
                    - **Moderate (1.5-3%)**: Some patterns exist but not dominant
                    - **Low (<1.5%)**: Weak seasonality - timing likely not beneficial
                    
                    ⚠️ **Caution**: Past seasonality patterns may not persist. Use this analysis as one input among many, not as a trading signal.
                    """)
            else:
                st.warning("⚠️ Not enough data for seasonality analysis. Need at least 20 trading days.")# Part 5: Performance, Rolling, and Deep-dive Statistics Tabs

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
        # TAB 6: BACKTEST VALIDATION (MULTI-METRIC) - COMPLETE VERSION
        with tab6:
            st.markdown("### 🧪 Backtest Validation (AFML – Combinatorial Purged CV)")
            
            # ==========================
            # INTRODUCTORY SECTION
            # ==========================
            st.markdown("""
            ## 🎯 What Does This Tab Do?
            
            This tab answers a critical question that traditional backtests ignore:
            
            > **"Would this strategy have *actually* worked if I had used it in the past *without knowing the future*?"**
            
            ### The Problem with Standard Backtests
            
            In the other tabs of this app, you optimize and evaluate strategies using **all available data**. 
            This creates an **illusion of performance** because:
            
            - You optimize portfolio weights on data that includes the test period
            - The strategy "knows" future market conditions during optimization
            - Reported Sharpe ratios are **systematically overstated**
            
            **Metaphor:** It's like a student who studies the exact exam questions, then takes the same exam and claims to be a genius.
            
            ### The Solution: Walk-Forward Validation
            
            This tab implements **Combinatorial Purged Cross-Validation (CPCV)**, a rigorous framework from 
            **Marcos López de Prado's "Advances in Financial Machine Learning" (2018, Chapter 7)**.
            
            **How it works:**
            
            1. **Split time into K folds** (e.g., 5 periods of ~2 years each for 2015-2024)
            2. **Create all possible train/test combinations** (e.g., 10 different splits)
            3. **For each split:**
            - Optimize strategy on **training data only** (e.g., 2019-2024)
            - Evaluate performance on **held-out test data** (e.g., 2015-2018)
            - Apply **embargo** to prevent information leakage
            4. **Aggregate results** across all splits to get a **distribution** of out-of-sample performance

                ### What You Learn From This Tab
                
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
            st.caption(
                "🛑 **Embargo** removes a buffer of data *after* the test period to prevent "
                "information leakage due to autocorrelation. \n\n"
                "An embargo of **1%–5%** means that data immediately following the test window "
                "is excluded from training. Higher values make validation more conservative."
            )

            methods_to_test = st.multiselect(
                "Strategies",
                ["equal", "min_vol", "max_sharpe", "risk_parity"],
                default=["equal", "min_vol", "max_sharpe", "risk_parity"]
            )

            method_names = {
                "equal": "Equally Weighted",
                "min_vol": "Minimum Volatility",
                "max_sharpe": "Maximum Sharpe",
                "risk_parity": "Risk Parity",
                "max_return": "Maximum Return",
                "hrp": "Hierarchical Risk Parity"
            }

            # Primary metric for PBO calculation
            primary_metric = st.selectbox(
                "Primary metric for PBO",
                ["sharpe", "sortino", "calmar"],
                index=1,
                help="The metric used to calculate Probability of Backtest Overfitting"
            )

            # ==========================
            # ROBUST METRICS CALCULATOR
            # ==========================
            def calculate_robust_metrics(returns, rf_rate=0.0):
                """
                Calculate comprehensive risk-adjusted performance metrics.
                
                Returns dictionary with:
                - sharpe: Standard Sharpe Ratio
                - sortino: Sortino Ratio (downside deviation)
                - calmar: Calmar Ratio (return / max drawdown)
                - max_drawdown: Maximum peak-to-trough decline
                - cvar_95: Conditional Value at Risk (5% worst cases)
                - win_rate: Percentage of positive return periods
                """
                if len(returns) == 0:
                    return {
                        'sharpe': np.nan,
                        'sortino': np.nan,
                        'calmar': np.nan,
                        'max_drawdown': np.nan,
                        'cvar_95': np.nan,
                        'win_rate': np.nan
                    }
                
                returns_excess = returns - rf_rate
                
                # 1. SHARPE RATIO (volatility-adjusted)
                # Formula: (Annualized Return - Risk Free Rate) / Annualized Volatility
                # Note: rf_rate is already annualized, so we don't multiply by 252
                if returns.std() > 0:
                    annual_return = returns.mean() * 252
                    annual_vol = returns.std() * np.sqrt(252)
                    sharpe = (annual_return - rf_rate) / annual_vol
                else:
                    sharpe = np.nan
                
                # 2. SORTINO RATIO (downside risk only)
                # Formula: (Annualized Return - Risk Free Rate) / Annualized Downside Deviation
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    annual_return = returns.mean() * 252
                    downside_std = downside_returns.std() * np.sqrt(252)
                    if downside_std > 0:
                        sortino = (annual_return - rf_rate) / downside_std
                    else:
                        sortino = np.nan
                else:
                    # No negative returns - strategy never loses
                    sortino = np.inf if returns.mean() > 0 else np.nan
                
                # 3. MAXIMUM DRAWDOWN
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = abs(drawdown.min())
                
                # 4. CALMAR RATIO (return / max drawdown)
                annual_return = returns.mean() * 252
                if max_dd > 0:
                    calmar = annual_return / max_dd
                else:
                    calmar = np.inf if annual_return > 0 else np.nan
                
                # 5. CONDITIONAL VALUE AT RISK (CVaR at 95%)
                if len(returns) > 0:
                    cvar_95 = returns.quantile(0.05)
                else:
                    cvar_95 = np.nan
                
                # 6. WIN RATE
                win_rate = (returns > 0).mean()
                
                return {
                    'sharpe': sharpe,
                    'sortino': sortino,
                    'calmar': calmar,
                    'max_drawdown': max_dd,
                    'cvar_95': cvar_95,
                    'win_rate': win_rate
                }

            # ==========================
            # CORE FUNCTIONS (AFML)
            # ==========================
            from itertools import combinations

            def combinatorial_purged_cv(dates, n_splits, n_test_splits, embargo_pct):
                """
                Creates train/test splits with embargo to prevent leakage.
                """
                fold_size = len(dates) // n_splits
                
                # Validate minimum fold size
                if fold_size < 20:
                    raise ValueError(f"Fold size too small ({fold_size}). Reduce n_splits or use more data.")
                
                # Create equal-sized folds
                folds = []
                for i in range(n_splits):
                    start_idx = i * fold_size
                    end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(dates)
                    folds.append(dates[start_idx:end_idx])

                splits = []
                embargo_size = int(len(dates) * embargo_pct)
                
                for test_fold_indices in combinations(range(n_splits), n_test_splits):
                    # Collect test dates
                    test_dates = pd.Index([])
                    for fold_idx in test_fold_indices:
                        test_dates = test_dates.append(folds[fold_idx])
                    test_dates = test_dates.sort_values()
                    
                    # Start with all non-test dates
                    train_dates = dates.difference(test_dates)
                    
                    # Apply embargo: remove dates AFTER test period
                    if embargo_size > 0 and len(test_dates) > 0:
                        test_end = test_dates.max()
                        embargo_dates = dates[(dates > test_end)][:embargo_size]
                        train_dates = train_dates.difference(embargo_dates)
                    
                    # Validate split
                    if len(train_dates) < 50 or len(test_dates) < 10:
                        continue
                    
                    splits.append((train_dates.sort_values(), test_dates))

                if len(splits) == 0:
                    raise ValueError("No valid splits created. Adjust parameters.")
                
                return splits


            def run_cpcv_backtest(
                returns_df,
                methods,
                rf_rate,
                n_splits=5,
                n_test_splits=2,
                embargo_pct=0.01
            ):
                """
                Run combinatorial purged cross-validation with multi-metric evaluation.
                """
                splits = combinatorial_purged_cv(
                    returns_df.index,
                    n_splits,
                    n_test_splits,
                    embargo_pct
                )

                # Store all metrics for both IS and OOS
                is_metrics_all = {m: [] for m in methods}
                oos_metrics_all = {m: [] for m in methods}
                oos_returns = {m: [] for m in methods}
                
                n_splits_total = len(splits)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for split_idx, (train_idx, test_idx) in enumerate(splits):
                    progress_bar.progress((split_idx + 1) / n_splits_total)
                    status_text.text(f"Processing split {split_idx + 1}/{n_splits_total}...")
                    
                    train_returns = returns_df.loc[train_idx]
                    test_returns = returns_df.loc[test_idx]

                    for method in methods:
                        try:
                            # Optimize on training data
                            weights = optimize_portfolio_weights(
                                train_returns,
                                method=method,
                                rf_rate=rf_rate
                            )
                            
                            # Evaluate on IN-SAMPLE (training data)
                            train_portfolio_returns = train_returns.dot(weights)
                            is_metrics = calculate_robust_metrics(
                                train_portfolio_returns,
                                rf_rate=rf_rate
                            )
                            is_metrics_all[method].append(is_metrics)
                            
                            # Evaluate on OUT-OF-SAMPLE (test data)
                            test_portfolio_returns = test_returns.dot(weights)
                            oos_metrics = calculate_robust_metrics(
                                test_portfolio_returns,
                                rf_rate=rf_rate
                            )
                            
                            oos_returns[method].append(test_portfolio_returns)
                            oos_metrics_all[method].append(oos_metrics)
                            
                        except Exception as e:
                            st.warning(f"Split {split_idx+1}: {method} failed - {str(e)}")
                            # Append NaN metrics
                            nan_metrics = {k: np.nan for k in ['sharpe', 'sortino', 'calmar', 'max_drawdown', 'cvar_95', 'win_rate']}
                            is_metrics_all[method].append(nan_metrics)
                            oos_metrics_all[method].append(nan_metrics)
                            continue
                
                progress_bar.empty()
                status_text.empty()

                return is_metrics_all, oos_metrics_all, oos_returns


            def compute_pbo(is_metrics, oos_metrics, metric_name='sharpe'):
                """
                Compute Probability of Backtest Overfitting for specified metric.
                """
                # Extract metric values
                is_values = {method: [m[metric_name] for m in metrics] 
                            for method, metrics in is_metrics.items()}
                oos_values = {method: [m[metric_name] for m in metrics] 
                            for method, metrics in oos_metrics.items()}
                
                is_df = pd.DataFrame(is_values).dropna()
                oos_df = pd.DataFrame(oos_values).dropna()
                
                if len(is_df) == 0 or len(oos_df) == 0:
                    return np.nan
                
                # Ensure same splits
                common_idx = is_df.index.intersection(oos_df.index)
                is_df = is_df.loc[common_idx]
                oos_df = oos_df.loc[common_idx]
                
                n_strategies = len(is_df.columns)
                
                # For each split, find best IS strategy
                is_ranks = is_df.rank(axis=1, ascending=False)
                best_is_strategy = is_ranks.idxmin(axis=1)
                
                # Get OOS rank of that strategy
                oos_ranks = oos_df.rank(axis=1, ascending=False)
                oos_rank_of_best_is = np.array([
                    oos_ranks.loc[idx, best_is_strategy[idx]] 
                    for idx in common_idx
                ])
                
                # PBO: probability that best IS has OOS rank > n/2
                median_rank = (n_strategies + 1) / 2
                pbo = np.mean(oos_rank_of_best_is > median_rank)
                
                return pbo


            # ==========================
            # RUN BACKTEST
            # ==========================
            if st.button("🚀 Run Multi-Metric Validation", use_container_width=True):
                if not methods_to_test:
                    st.error("Select at least one strategy.")
                elif len(methods_to_test) < 2:
                    st.error("Select at least 2 strategies for meaningful comparison.")
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
                            st.error(f"Configuration error: {str(e)}")
                            st.stop()
                        except Exception as e:
                            st.error(f"Unexpected error: {str(e)}")
                            st.stop()

                    st.markdown("---")

                    # ==========================
                    # PBO METRIC
                    # ==========================
                    st.markdown(f"#### 📉 Probability of Backtest Overfitting ({primary_metric.title()})")
                    
                    if np.isnan(pbo):
                        st.warning("⚠️ PBO could not be calculated (insufficient valid splits)")
                    else:
                        if pbo < 0.1:
                            delta_color = "normal"
                            delta_text = "Excellent"
                        elif pbo < 0.3:
                            delta_color = "normal"
                            delta_text = "Good"
                        else:
                            delta_color = "inverse"
                            delta_text = "Warning"
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("PBO", f"{pbo:.1%}", delta_text, delta_color=delta_color)
                        with col2:
                            st.metric("Primary Metric", primary_metric.title())
                        with col3:
                            st.metric("Valid Splits", n_valid_splits)
                        
                        st.caption(
                            f"📉 **PBO** measures how often the strategy that looks best in {primary_metric} "
                            "during training actually underperforms out-of-sample. \n\n"
                            "**< 10%** = Excellent | **< 30%** = Acceptable | **> 50%** = Severe overfitting"
                        )
                    st.markdown("---")

                    # ==========================
                    # MULTI-METRIC DISTRIBUTIONS
                    # ==========================
                    st.markdown("#### 📊 Out-of-Sample Performance Distributions")
                    
                    metric_display = {
                        'sharpe': 'Sharpe Ratio',
                        'sortino': 'Sortino Ratio',
                        'calmar': 'Calmar Ratio',
                        'max_drawdown': 'Max Drawdown (%)'
                    }
                    
                    # Create tabs for each metric
                    metric_tabs = st.tabs(list(metric_display.values()))
                    
                    for tab_idx, (metric_key, metric_name) in enumerate(metric_display.items()):
                        with metric_tabs[tab_idx]:
                            fig = go.Figure()
                            
                            for i, method in enumerate(methods_to_test):
                                values = [m[metric_key] for m in oos_metrics_all[method] 
                                        if not np.isnan(m[metric_key])]
                                
                                # Handle special case for max_drawdown (convert to %)
                                if metric_key == 'max_drawdown':
                                    values = [v * 100 for v in values]
                                
                                if len(values) > 0:
                                    fig.add_trace(go.Box(
                                        y=values,
                                        name=method_names.get(method, method),
                                        boxmean='sd',
                                        marker_color=CHART_COLORS[i % len(CHART_COLORS)]
                                    ))
                            
                            y_title = metric_name
                            if metric_key == 'max_drawdown':
                                y_title += ' (lower is better)'
                            
                            fig.update_layout(
                                height=400,
                                yaxis_title=y_title,
                                showlegend=False
                            )
                            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)
                    
                    st.markdown("---")

                    # ==========================
                    # COMPREHENSIVE RANKING TABLE (NUOVO STILE)
                    # ==========================
                    st.markdown("#### 🏆 Comprehensive Multi-Metric Ranking")

                    ranking_data = []
                    ranking_data_raw = []  # Per il sorting
                    
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

                    # Sort by primary metric using raw data
                    metric_map = {'sharpe': 'sharpe', 'sortino': 'sortino', 'calmar': 'calmar'}
                    sort_key = metric_map.get(primary_metric, 'sharpe')
                    
                    # Sort raw data
                    ranking_data_raw.sort(key=lambda x: x[sort_key] if not np.isnan(x[sort_key]) else -999, reverse=True)
                    
                    # Reorder display data based on sorted raw data
                    method_order = [item['method'] for item in ranking_data_raw]
                    ranking_data = sorted(ranking_data, key=lambda x: method_order.index(x['method']))

                    # Add rank icons
                    for i, row in enumerate(ranking_data):
                        icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
                        row['#'] = f"{icon} {i+1}"

                    # Create DataFrame with proper column order
                    ranking_df = pd.DataFrame(ranking_data)
                    cols = ['#', 'Strategy', 'Sharpe', 'Sortino', 'Calmar', 'Max DD', 'CVaR (5%)', 'Win Rate']
                    ranking_df = ranking_df[cols]

                    # Display styled table
                    st.markdown(create_styled_table(ranking_df, f"Ranked by {primary_metric.title()} (Median OOS)"), unsafe_allow_html=True)

                    st.caption(
                        f"📊 Rankings are based on median OOS performance across all splits. "
                        f"Sorted by **{primary_metric.title()}**."
                    )

                    # ==========================
                    # METRIC INTERPRETATION GUIDE
                    # ==========================
                    with st.expander("📖 How to Interpret These Metrics"):
                        st.markdown("""
                        ### Risk-Adjusted Performance Metrics
                        
                        **Sharpe Ratio**
                        - Measures return per unit of total volatility
                        - Higher is better (> 1.0 is good, > 2.0 is excellent)
                        - ⚠️ Limitation: Penalizes upside volatility equally with downside
                        
                        **Sortino Ratio** ⭐ *Recommended*
                        - Measures return per unit of *downside* volatility only
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
                        - Lower is better
                        - Critical for psychological sustainability
                        - Hedge funds typically target < 20%
                        
                        **CVaR (Conditional Value at Risk)**
                        - Average of the worst 5% of returns
                        - More negative = higher tail risk
                        - Captures "black swan" events better than Sharpe
                        
                        **Win Rate**
                        - Percentage of positive return periods
                        - Higher is better
                        - ⚠️ Not sufficient alone (small wins + big losses can have high win rate)
                        
                        ### Best Practice
                        Use **multiple metrics** together:
                        - High Sortino/Calmar = robust risk-adjusted returns
                        - Low Max Drawdown = sustainable strategy
                        - Moderate CVaR = controlled tail risk
                        """)

                    st.markdown("---")

                    # ==========================
                    # STRATEGY-SPECIFIC ANALYSIS (STILE TABELLA)
                    # ==========================
                    st.markdown("### 🔍 Strategy-Specific Analysis")
                    
                    if len(ranking_data_raw) > 0:
                        # Get best and worst strategies from raw data
                        best_method = ranking_data_raw[0]['method']
                        best_name = method_names.get(best_method, best_method)
                        
                        # Find formatted data for best strategy
                        best_data = next((item for item in ranking_data if item['method'] == best_method), None)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"#### 🥇 Top Strategy: {best_name}")
                            
                            if best_data:
                                # Create compact table for top strategy
                                top_strategy_data = {
                                    'Metric': ['Sharpe', 'Sortino', 'Calmar', 'Max DD', 'CVaR (5%)', 'Win Rate'],
                                    'Value': [
                                        best_data['Sharpe'],
                                        best_data['Sortino'],
                                        best_data['Calmar'],
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
                                
                                st.markdown(f"#### 📊 vs. {worst_name}")
                                
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
                                        'Metric': ['Sortino Advantage', 'DD Advantage (pp)'],
                                        'Value': [
                                            f"+{sortino_diff:.2f} ({sortino_diff/worst_sortino*100:+.1f}%)" if worst_sortino != 0 else f"+{sortino_diff:.2f}",
                                            f"{dd_diff:.1f}pp"
                                        ]
                                    }
                                    st.markdown(create_styled_table(pd.DataFrame(comparison_data)), unsafe_allow_html=True)
                                    
                                    if sortino_diff > 0.3 and dd_diff > 10:
                                        st.success("✅ Superior risk-adjusted returns with lower tail risk")
                                    elif sortino_diff > 0.3:
                                        st.info("📊 Superior risk-adjusted returns")
                                    elif dd_diff > 10:
                                        st.info("📊 Significantly lower tail risk")
                                    else:
                                        st.warning("⚠️ Comparable performance profiles")

                    st.markdown("---")

                    # ==========================
                    # FINAL INTERPRETATION
                    # ==========================
                    st.markdown("## 🎯 Final Takeaway")

                    # Determine overall assessment
                    if not np.isnan(pbo):
                        if pbo < 0.1:
                            assessment_icon = "🟢"
                            assessment_text = "ROBUST"
                            assessment_detail = "Low overfitting risk. Strategies show consistent OOS performance."
                        elif pbo < 0.3:
                            assessment_icon = "🟡"
                            assessment_text = "ACCEPTABLE"
                            assessment_detail = "Moderate reliability. Monitor performance in live trading."
                        elif pbo < 0.5:
                            assessment_icon = "🟠"
                            assessment_text = "CAUTION"
                            assessment_detail = "High overfitting risk. Consider ensemble or simpler strategies."
                        else:
                            assessment_icon = "🔴"
                            assessment_text = "WARNING"
                            assessment_detail = "Severe overfitting. Strategy selection is unreliable."
                    else:
                        assessment_icon = "⚠️"
                        assessment_text = "INSUFFICIENT DATA"
                        assessment_detail = "Unable to assess overfitting risk."

                    st.markdown(f"### {assessment_icon} Overall Assessment: **{assessment_text}**")
                    st.markdown(f"*{assessment_detail}*")
                    
                    st.markdown("---")
                    
                    # Key insights in columns
                    st.markdown("**📌 Key Insights**")
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    with insight_col1:
                        st.metric("Primary Metric", primary_metric.title())
                    with insight_col2:
                        pbo_color = "normal" if pbo < 0.3 else "inverse"
                        st.metric(
                            f"PBO ({primary_metric})", 
                            f"{pbo:.1%}",
                            delta="Robust" if pbo < 0.3 else "Warning",
                            delta_color=pbo_color
                        )
                    with insight_col3:
                        if len(ranking_data) > 0:
                            st.metric("Best Strategy", ranking_data[0]['Strategy'])
                    
                    st.markdown("---")
                    
                    # Recommendations
                    st.markdown("**💡 Recommendations**")
                    
                    rec_col1, rec_col2 = st.columns(2)
                    
                    with rec_col1:
                        st.success("✅ **What to Look For:**")
                        st.markdown("""
                        - High **Sortino** and **Calmar** ratios
                        - **Max DD < 40%** for sustainability
                        - Low **CVaR** for tail risk control
                        - Consistent performance across splits
                        """)
                    
                    with rec_col2:
                        st.warning("⚠️ **Red Flags:**")
                        st.markdown("""
                        - **PBO > 30%** indicates overfitting
                        - Large dispersion in OOS metrics
                        - Extreme Sharpe values (> 3.0)
                        - Strategy works only in specific regimes
                        """)
                    
                    st.info("""
                    **🎓 Remember:**
                    - This validation tests **robustness**, not "the best strategy"
                    - Out-of-sample metrics are closer to real-world performance
                    - No single metric tells the full story—consider the complete profile
                    - Use **ensemble approaches** if multiple strategies show robustness
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
