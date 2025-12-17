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

# ============ TRANSACTION COSTS & REBALANCING ============

def calculate_portfolio_with_rebalancing(
    returns_df,
    weights_target,
    rebalance_freq=None,
    rebalance_threshold=None,
    cost_bps=0,
    rf_rate=0.02
):
    """
    Calculate portfolio returns with rebalancing and transaction costs.
    
    Methodology based on:
    - DeMiguel, Garlappi, Uppal (2009) "Optimal Versus Naive Diversification"
      Review of Financial Studies
    - Kirby & Ostdiek (2012) "It's All in the Timing"
      Journal of Financial and Quantitative Analysis
    
    Transaction costs are applied as proportional costs that reduce portfolio value:
        V_t = V_{t-1} * (1 + r_gross) * (1 - TC_t)
    
    where TC_t = turnover_t * cost_rate (only on rebalancing days)
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Daily returns for each asset
    weights_target : array-like
        Target portfolio weights
    rebalance_freq : str, optional
        Pandas frequency string ('D', 'W', 'M', 'Q', 'A') for calendar rebalancing
    rebalance_threshold : float, optional
        Maximum drift allowed before triggering rebalancing (e.g., 0.05 = 5%)
    cost_bps : float
        Proportional transaction cost in basis points (one-way).
        Example: 10 bps = 0.10% cost per unit of turnover
    rf_rate : float
        Annual risk-free rate for Sharpe calculation
        
    Returns:
    --------
    dict with portfolio metrics, returns series, and rebalancing details
    """
    weights_target = np.array(weights_target)
    weights_current = weights_target.copy()
    n_assets = len(weights_target)
    cost_rate = cost_bps / 10000  # Convert bps to decimal
    
    # Track portfolio value explicitly (not just returns)
    # This is critical for correct compounding of costs
    portfolio_value = 1.0  # Start with $1
    portfolio_values = []
    
    # For gross comparison (same rebalancing, no costs)
    portfolio_value_gross = 1.0
    portfolio_values_gross = []
    weights_gross = weights_target.copy()
    
    # Tracking variables
    turnover_history = []
    rebalance_dates = []
    weights_history = []
    costs_paid = []
    
    # Build rebalance schedule if calendar-based
    if rebalance_freq:
        rebalance_schedule = set(returns_df.resample(rebalance_freq).last().index)
    else:
        rebalance_schedule = set()
    
    for date, daily_returns in returns_df.iterrows():
        
        # ===== NET PORTFOLIO (with costs) =====
        
        # 1. Calculate gross return for the day
        port_return_gross = np.dot(weights_current, daily_returns.values)
        
        # 2. Update portfolio value with gross return (before costs)
        portfolio_value_pre_cost = portfolio_value * (1 + port_return_gross)
        
        # 3. Update weights based on drift (before any rebalancing)
        weights_drifted = weights_current * (1 + daily_returns.values)
        total_weight = weights_drifted.sum()
        if total_weight > 0:
            weights_drifted = weights_drifted / total_weight
        else:
            weights_drifted = weights_target.copy()
        
        # 4. Calculate current drift
        current_drift = np.max(np.abs(weights_drifted - weights_target))
        
        # 5. Check if rebalancing needed
        needs_rebalance = False
        
        if rebalance_threshold is not None:
            if current_drift > rebalance_threshold:
                needs_rebalance = True
        elif rebalance_freq and date in rebalance_schedule:
            needs_rebalance = True
        
        # 6. Execute rebalancing with transaction costs
        if needs_rebalance:
            # One-way turnover: sum of absolute weight changes
            # This represents the total fraction of portfolio traded
            # Note: we do NOT divide by 2 (one-way convention per DeMiguel et al.)
            turnover = np.sum(np.abs(weights_drifted - weights_target))
            
            # Transaction cost reduces portfolio value multiplicatively
            # This correctly captures the compounding effect of costs
            transaction_cost = turnover * cost_rate
            portfolio_value_post_cost = portfolio_value_pre_cost * (1 - transaction_cost)
            
            # Reset weights to target
            weights_current = weights_target.copy()
            
            # Track
            turnover_history.append(turnover)
            rebalance_dates.append(date)
            costs_paid.append(transaction_cost * portfolio_value_pre_cost)
        else:
            portfolio_value_post_cost = portfolio_value_pre_cost
            weights_current = weights_drifted.copy()
        
        # Update portfolio value
        portfolio_value = portfolio_value_post_cost
        portfolio_values.append(portfolio_value)
        
        # ===== GROSS PORTFOLIO (no costs, same rebalancing schedule) =====
        
        port_return_gross_track = np.dot(weights_gross, daily_returns.values)
        portfolio_value_gross *= (1 + port_return_gross_track)
        portfolio_values_gross.append(portfolio_value_gross)
        
        # Drift weights for gross tracking
        weights_gross_drifted = weights_gross * (1 + daily_returns.values)
        weights_gross_drifted = weights_gross_drifted / weights_gross_drifted.sum()
        
        # Rebalance gross at same times (but no cost)
        if needs_rebalance:
            weights_gross = weights_target.copy()
        else:
            weights_gross = weights_gross_drifted.copy()
        
        # Track weights
        weights_history.append({
            'date': date,
            'weights': weights_current.copy(),
            'max_drift': current_drift,
            'rebalanced': needs_rebalance
        })
    
    # Convert to series
    portfolio_values_series = pd.Series(portfolio_values, index=returns_df.index)
    portfolio_values_gross_series = pd.Series(portfolio_values_gross, index=returns_df.index)
    
    # Calculate returns from values
    returns_net = portfolio_values_series.pct_change().dropna()
    returns_gross = portfolio_values_gross_series.pct_change().dropna()
    
    # ===== CALCULATE METRICS =====
    
    # Time period
    n_years = len(returns_df) / 252
    
    # Total returns
    total_return_net = portfolio_values_series.iloc[-1] - 1
    total_return_gross = portfolio_values_gross_series.iloc[-1] - 1
    
    # Annualized returns (geometric - correct for compounding)
    ann_return_net = (1 + total_return_net) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_return_gross = (1 + total_return_gross) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Volatility
    ann_vol_net = returns_net.std() * np.sqrt(252)
    ann_vol_gross = returns_gross.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_net = (ann_return_net - rf_rate) / ann_vol_net if ann_vol_net > 0 else 0
    sharpe_gross = (ann_return_gross - rf_rate) / ann_vol_gross if ann_vol_gross > 0 else 0
    
    # Sortino Ratio (net)
    downside = returns_net[returns_net < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol_net
    sortino_net = (ann_return_net - rf_rate) / downside_std if downside_std > 0 else 0
    
    # Max Drawdown (net)
    cumulative_net = portfolio_values_series
    rolling_max = cumulative_net.expanding().max()
    drawdown = (cumulative_net - rolling_max) / rolling_max
    max_dd_net = drawdown.min()
    
    # Calmar Ratio (net)
    calmar_net = ann_return_net / abs(max_dd_net) if max_dd_net != 0 else 0
    
    # ===== COST ANALYSIS =====
    
    # Total turnover over the period
    total_turnover = sum(turnover_history)
    
    # Total costs as percentage of final value
    # This measures cumulative drag from all transactions
    if portfolio_values_gross_series.iloc[-1] > 0:
        total_costs_pct = (1 - portfolio_values_series.iloc[-1] / portfolio_values_gross_series.iloc[-1]) * 100
    else:
        total_costs_pct = 0
    
    # Number of rebalancing events
    n_rebalances = len(rebalance_dates)
    
    # Average turnover per rebalancing
    avg_turnover = total_turnover / n_rebalances if n_rebalances > 0 else 0
    
    # Annual cost drag (difference in annualized returns)
    cost_drag = ann_return_gross - ann_return_net
    
    # Annualized turnover
    annual_turnover = total_turnover / n_years if n_years > 0 else 0
    
    # Create weights DataFrame for visualization
    weights_df = pd.DataFrame([
        {**{'date': w['date'], 'max_drift': w['max_drift'], 'rebalanced': w['rebalanced']}, 
         **{f'weight_{i}': w['weights'][i] for i in range(n_assets)}}
        for w in weights_history
    ])
    weights_df.set_index('date', inplace=True)
    
    return {
        # Returns series
        'returns_gross': returns_gross,
        'returns_net': returns_net,
        'cumulative_gross': portfolio_values_gross_series,
        'cumulative_net': portfolio_values_series,
        
        # Gross metrics
        'ann_return_gross': ann_return_gross,
        'ann_vol_gross': ann_vol_gross,
        'sharpe_gross': sharpe_gross,
        
        # Net metrics
        'ann_return_net': ann_return_net,
        'ann_vol_net': ann_vol_net,
        'sharpe_net': sharpe_net,
        'sortino_net': sortino_net,
        'max_dd_net': max_dd_net,
        'calmar_net': calmar_net,
        
        # Cost analysis
        'total_turnover': total_turnover,
        'annual_turnover': annual_turnover,
        'total_costs_pct': total_costs_pct,
        'n_rebalances': n_rebalances,
        'avg_turnover': avg_turnover,
        'cost_drag': cost_drag,
        'rebalance_dates': rebalance_dates,
        
        # Weight tracking
        'weights_history': weights_df,
        'weights_target': weights_target
    }

def calculate_all_portfolios_with_costs(analyzer, rebalance_freq, rebalance_threshold, cost_bps, rf_rate):
    """
    Recalculate all portfolios with transaction costs and rebalancing.
    
    Returns dict with enhanced metrics for each portfolio.
    """
    results = {}
    
    for key, portfolio in analyzer.portfolios.items():
        enhanced = calculate_portfolio_with_rebalancing(
            returns_df=analyzer.returns,
            weights_target=portfolio['weights'],
            rebalance_freq=rebalance_freq,
            rebalance_threshold=rebalance_threshold,
            cost_bps=cost_bps,
            rf_rate=rf_rate
        )
        
        # Merge with original portfolio data
        results[key] = {
            'name': portfolio['name'],
            'weights': portfolio['weights'],  # Original weights array
            **enhanced  # Enhanced metrics including weights_target, weights_history, etc.
        }
    
    return results
    
def get_hrp_dendrogram_data(analyzer):
    """
    Generate dendrogram data for HRP visualization.
    Returns linkage matrix and sorted symbols for plotting.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    returns_aligned = analyzer.returns.reindex(columns=analyzer.symbols)
    corr_matrix = returns_aligned.corr()
    
    # Handle NaN
    if corr_matrix.isnull().any().any():
        corr_matrix = corr_matrix.fillna(0)
    np.fill_diagonal(corr_matrix.values, 1.0)
    
    # Distance matrix (López de Prado formula)
    distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    np.fill_diagonal(distance_matrix.values, 0)
    
    # Condensed form and linkage
    dist_condensed = squareform(distance_matrix.values, checks=False)
    link = linkage(dist_condensed, method='single')
    
    return link, analyzer.symbols, corr_matrix, distance_matrix

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
    window_years = st.slider("Rolling Window (years)", 1, 10, 3, key="window")
    
    # ============ NEW: Transaction Costs & Rebalancing ============
    st.markdown("#### 💰 Transaction Costs")
    
    enable_costs = st.checkbox("Enable transaction costs", value=True, key="enable_costs")
    
    if enable_costs:
        transaction_cost_bps = st.slider(
            "Cost per trade (bps)", 
            min_value=0, 
            max_value=50, 
            value=10, 
            step=1,
            key="tx_cost",
            help="Basis points charged per transaction. 10 bps = 0.10%"
        )
        st.caption("💡 Typical: 3-10 bps for ETFs, 10-30 bps for stocks")
    else:
        transaction_cost_bps = 0
    
    st.markdown("#### 🔄 Rebalancing Strategy")
    
    rebalance_type = st.radio(
        "Rebalancing method",
        ["Calendar-based", "Threshold-based", "Buy & Hold"],
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
            st.markdown("---")
            
            # ============================================================
            # SECTION 5: MORE ON HIERARCHICAL RISK PARITY
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
                a family tree (dendrogram) where similar assets are grouped together.
                
                The distance between assets is calculated as:
                
                $$d_{i,j} = \\sqrt{\\frac{1}{2}(1 - \\rho_{i,j})}$$
                
                where $\\rho_{i,j}$ is the correlation between assets $i$ and $j$.
                
                **Step 2: Quasi-Diagonalization**
                
                We reorder the assets so that similar ones are next to each other. Imagine rearranging 
                seats at a dinner party so that people who get along well sit together.
                
                **Step 3: Recursive Bisection**
                
                Finally, we allocate weights by repeatedly splitting the portfolio in half and giving 
                more weight to the "safer" (lower variance) half. This continues until each asset 
                has its final weight.
                
                The allocation formula at each split is:
                
                $$\\alpha = \\frac{V_{right}}{V_{left} + V_{right}}$$
                
                where $V$ is the variance of each cluster. The cluster with **lower variance gets more weight**.
                """)
                
                st.markdown("---")
                st.markdown("### 📊 Dendrogram: Your Portfolio's Family Tree")
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
        
        # TAB 4: DEEP-DIVE STATISTICS (NEW)
        with tab4:
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
        # TAB 5: BACKTEST VALIDATION (MULTI-METRIC) - COMPLETE VERSION
        with tab5:
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

        # TAB 6: FRONTIER
        with tab6:
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
