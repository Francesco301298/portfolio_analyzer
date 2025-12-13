"""Portfolio Analyzer Pro - Streamlit Cloud Ready"""

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

# Compact CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root{--bg-primary:#0a0a0f;--bg-secondary:#12121a;--bg-card:#1a1a24;--accent-primary:#6366f1;--accent-gradient:linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#a855f7 100%);--text-primary:#f8fafc;--text-secondary:#94a3b8;--success:#10b981;--warning:#f59e0b;--danger:#ef4444;--border-color:rgba(99,102,241,0.2)}
*{font-family:'Inter',sans-serif}
.main,.stApp{background:var(--bg-primary)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--bg-secondary) 0%,var(--bg-primary) 100%);border-right:1px solid var(--border-color)}
.stButton>button{width:100%;background:var(--accent-gradient);color:white;padding:0.875rem 1.5rem;font-weight:600;border-radius:12px;border:none;box-shadow:0 4px 16px rgba(0,0,0,0.4);transition:all 0.3s}
.stButton>button:hover{transform:translateY(-2px);filter:brightness(1.1)}
.dashboard-card{background:linear-gradient(145deg,var(--bg-card) 0%,var(--bg-secondary) 100%);padding:1.75rem;border-radius:16px;margin:0.75rem 0;border:1px solid var(--border-color)}
.dashboard-card h3{background:var(--accent-gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700}
.dashboard-card ul,.dashboard-card ol,.dashboard-card p{color:var(--text-secondary);line-height:1.7}
.ticker-pill{display:inline-block;background:var(--accent-gradient);color:white;padding:0.35rem 0.9rem;border-radius:20px;margin:0.25rem;font-size:0.8rem;font-weight:600}
[data-testid="stMetricValue"]{background:var(--accent-gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700}
[data-testid="stMetricLabel"]{color:var(--text-secondary)!important;font-weight:600;text-transform:uppercase;font-size:0.75rem!important}
h1{background:var(--accent-gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800}
h2{color:var(--text-primary);font-weight:700;border-bottom:2px solid transparent;border-image:var(--accent-gradient);border-image-slice:1;padding-bottom:0.75rem}
h3{color:var(--accent-primary);font-weight:600}
p,li,span{color:var(--text-secondary)}
.stTabs [data-baseweb="tab-list"]{background:var(--bg-card);border-radius:12px;padding:0.5rem}
.stTabs [data-baseweb="tab"]{color:var(--text-secondary);font-weight:600;border-radius:8px}
.stTabs [aria-selected="true"]{background:var(--accent-gradient)!important;color:white!important}
.hero-section{text-align:center;padding:3rem 0}
.hero-title{font-size:3rem;font-weight:800;background:var(--accent-gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-subtitle{font-size:1.1rem;color:var(--text-secondary);letter-spacing:2px;text-transform:uppercase}
.hero-divider{width:120px;height:3px;background:var(--accent-gradient);margin:1.5rem auto;border-radius:2px}
.footer-section{text-align:center;padding:2.5rem;background:linear-gradient(145deg,var(--bg-card) 0%,var(--bg-secondary) 100%);border-radius:20px;margin-top:3rem;border:1px solid var(--border-color)}
</style>
""", unsafe_allow_html=True)

# Ticker Database
TICKER_DATABASE = {
    "🇺🇸 US Tech": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AMD","INTC","CRM","ADBE","NFLX","PYPL","QCOM","AVGO","PLTR","CRWD","SNOW","NET"],
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

CHART_COLORS = ['#6366f1','#8b5cf6','#a855f7','#10b981','#f59e0b','#ef4444','#06b6d4','#ec4899','#84cc16','#f97316']

def apply_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8',family='Inter'),title_font=dict(color='#f8fafc',size=16),
        legend=dict(bgcolor='rgba(26,26,36,0.8)',bordercolor='rgba(99,102,241,0.3)',borderwidth=1),
        xaxis=dict(gridcolor='rgba(99,102,241,0.1)',linecolor='rgba(99,102,241,0.2)',tickfont=dict(color='#64748b')),
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)',linecolor='rgba(99,102,241,0.2)',tickfont=dict(color='#64748b')),
        hoverlabel=dict(bgcolor='#1a1a24',bordercolor='#6366f1',font=dict(color='#f8fafc'))
    )
    return fig

# Session State
for key, default in [('analyzer',None),('analysis_complete',False),('selected_tickers',[]),('saved_portfolios',{}),('alerts',[]),('benchmark','^GSPC'),('use_benchmark',True),('benchmark_returns',None)]:
    if key not in st.session_state:
        st.session_state[key] = default
# Part 2: Sidebar and Main Content

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
    st.markdown("## ⚙️ CONFIGURATION")
    selection_method = st.radio("Method", ["📋 Database", "✍️ Manual"], label_visibility="collapsed")
    selected_symbols = []
    
    if selection_method == "📋 Database":
        search_query = st.text_input("🔍 Search", placeholder="e.g., AAPL")
        for category, tickers in TICKER_DATABASE.items():
            with st.expander(category, expanded=False):
                filtered = [t for t in tickers if search_query.upper() in t.upper()] if search_query else tickers
                if filtered:
                    select_all = st.checkbox(f"Select all ({len(filtered)})", key=f"all_{category}")
                    cols = st.columns(3)
                    for idx, ticker in enumerate(filtered):
                        with cols[idx % 3]:
                            if st.checkbox(ticker, value=select_all, key=f"t_{category}_{ticker}"):
                                selected_symbols.append(ticker)
    else:
        manual_input = st.text_area("Enter tickers (comma/newline separated)", height=150, placeholder="AAPL, MSFT, GOOGL")
        if manual_input:
            import re
            selected_symbols = [s.strip().upper() for s in re.split('[,\n]', manual_input) if s.strip()]
    
    selected_symbols = list(dict.fromkeys(selected_symbols))
    st.markdown("---")
    
    st.markdown("### 📅 PARAMETERS")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=datetime(2018,1,1), min_value=datetime(2000,1,1))
    with col2:
        end_date = st.date_input("End", value=datetime.now(), min_value=start_date)
    
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
    window_years = st.slider("Rolling Window (years)", 1, 5, 3)
    
    st.markdown("### 📊 BENCHMARK")
    use_benchmark = st.checkbox("Use benchmark", value=True)
    st.session_state.use_benchmark = use_benchmark
    if use_benchmark:
        benchmark_ticker = st.selectbox("Benchmark", ["^GSPC","^DJI","^IXIC","SPY","QQQ","VTI"])
        st.session_state.benchmark = benchmark_ticker
    
    st.markdown("### 🔔 ALERTS")
    enable_alerts = st.checkbox("Enable alerts", value=True)
    max_dd_threshold = st.slider("Max DD Alert (%)", 5, 50, 20) if enable_alerts else 20
    min_sharpe_threshold = st.slider("Min Sharpe Alert", 0.0, 2.0, 0.5, 0.1) if enable_alerts else 0.5
    
    st.markdown("---")
    st.metric("Selected Tickers", len(selected_symbols))
    if selected_symbols:
        with st.expander("Show tickers"):
            st.markdown(" ".join([f"<span class='ticker-pill'>{t}</span>" for t in selected_symbols]), unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🚀 START ANALYSIS", type="primary", use_container_width=True):
        if len(selected_symbols) < 2:
            st.error("⚠️ Select at least 2 tickers!")
        else:
            st.session_state.selected_tickers = selected_symbols
            st.session_state.analysis_complete = False
            st.rerun()

# Main Content
if len(st.session_state.selected_tickers) == 0:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='dashboard-card'><h3>🎯 QUICK START</h3><ol><li>Select tickers</li><li>Configure parameters</li><li>Click Start Analysis</li><li>Explore results</li></ol></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='dashboard-card'><h3>📚 DATABASE</h3><p>Access to <strong>150+</strong> tickers:</p><ul><li>US Stocks</li><li>ETFs & Indices</li><li>Crypto & Bonds</li></ul></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='dashboard-card'><h3>⚡ FEATURES</h3><ul><li>8 optimization strategies</li><li>Rolling analysis</li><li>Efficient frontier</li><li>Benchmark comparison</li></ul></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🎲 QUICK EXAMPLES")
    examples = {
        "🔥 Tech Giants": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA"],
        "💰 Dividend": ["JNJ","PG","KO","PEP","MCD","WMT"],
        "🌍 Global": ["SPY","EFA","EEM","GLD","TLT","VNQ"],
        "🏦 Finance": ["JPM","BAC","GS","MS","V","MA"]
    }
    cols = st.columns(4)
    for idx, (name, tickers) in enumerate(examples.items()):
        with cols[idx]:
            if st.button(name, use_container_width=True, key=f"ex_{idx}"):
                st.session_state.selected_tickers = tickers
                st.rerun()
# Part 3: Analysis Logic

else:
    symbols = st.session_state.selected_tickers
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Initializing...")
        progress_bar.progress(10)
        
        from portfolio_analyzer import AdvancedPortfolioAnalyzer
        
        status_text.text("📥 Downloading data...")
        progress_bar.progress(20)
        
        analyzer = AdvancedPortfolioAnalyzer(symbols, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'), risk_free_rate=risk_free_rate)
        
        if YF_AVAILABLE:
            analyzer.download_data()
        else:
            st.error("❌ yfinance not available")
            st.stop()
        
        progress_bar.progress(40)
        status_text.text("📊 Calculating returns...")
        analyzer.calculate_returns()
        symbols = analyzer.symbols
        progress_bar.progress(60)
        
        status_text.text("💼 Optimizing portfolios...")
        analyzer.build_all_portfolios()
        progress_bar.progress(90)
        
        # Benchmark
        benchmark_returns = None
        if st.session_state.use_benchmark and st.session_state.benchmark and YF_AVAILABLE:
            try:
                status_text.text(f"📈 Loading benchmark...")
                benchmark_data = yf.download(st.session_state.benchmark, start=start_date, end=end_date, progress=False)['Close']
                benchmark_returns = benchmark_data.pct_change().dropna()
                if benchmark_returns.empty:
                    benchmark_returns = None
            except:
                benchmark_returns = None
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        st.session_state.analyzer = analyzer
        st.session_state.benchmark_returns = benchmark_returns
        st.session_state.analysis_complete = True
        
        progress_bar.empty()
        status_text.empty()
        
        # Alerts
        if enable_alerts:
            st.session_state.alerts = []
            for name, portfolio in analyzer.portfolios.items():
                if abs(portfolio['max_drawdown']) > (max_dd_threshold/100):
                    st.session_state.alerts.append({'type':'warning','portfolio':portfolio['name'],'message':f"High DD: {portfolio['max_drawdown']*100:.1f}%"})
                if portfolio['sharpe_ratio'] < min_sharpe_threshold:
                    st.session_state.alerts.append({'type':'info','portfolio':portfolio['name'],'message':f"Low Sharpe: {portfolio['sharpe_ratio']:.2f}"})
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
        st.stop()
    
    analyzer = st.session_state.analyzer
    benchmark_returns = st.session_state.benchmark_returns
    
    if st.session_state.alerts:
        with st.expander(f"🔔 {len(st.session_state.alerts)} Alerts", expanded=True):
            for alert in st.session_state.alerts:
                if alert['type'] == 'warning':
                    st.warning(f"**{alert['portfolio']}**: {alert['message']}")
                else:
                    st.info(f"**{alert['portfolio']}**: {alert['message']}")
    
    # KPI Dashboard
    st.markdown("## 📊 KPI DASHBOARD")
    portfolios_list = list(analyzer.portfolios.values())
    kpi_cols = st.columns(6 if (st.session_state.use_benchmark and benchmark_returns is not None) else 5)
    
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
        st.metric("📊 Avg Return", f"{np.mean([p['annualized_return'] for p in portfolios_list])*100:.1f}%")
    
    if st.session_state.use_benchmark and benchmark_returns is not None and len(benchmark_returns) > 0:
        with kpi_cols[5]:
            try:
                bench_cumret = (1 + benchmark_returns).prod() - 1
                n_years = len(benchmark_returns) / 252
                bench_annret = (1 + bench_cumret) ** (1/n_years) - 1 if n_years > 0 else 0
                st.metric("📈 Benchmark", f"{bench_annret*100:.1f}%", delta=st.session_state.benchmark)
            except:
                st.metric("📈 Benchmark", "N/A")
    
    st.markdown("---")
    
    # Tabs
    if st.session_state.use_benchmark and benchmark_returns is not None:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📊 Overview","💼 Portfolios","📈 Performance","🔄 Rolling","🔗 Correlations","📐 Frontier","🎯 Benchmark","📥 Export"])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Overview","💼 Portfolios","📈 Performance","🔄 Rolling","🔗 Correlations","📐 Frontier","📥 Export"])
        tab8 = None
# Part 4: Tab Contents

    # TAB 1: OVERVIEW
    with tab1:
        st.header("📊 OVERVIEW")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Asset Performance (Base 100)")
            normalized = (analyzer.data / analyzer.data.iloc[0]) * 100
            fig = go.Figure()
            for i, col_name in enumerate(normalized.columns):
                fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col_name], name=col_name, mode='lines', line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2)))
            fig.update_layout(height=500, hovermode='x unified', xaxis_title="Date", yaxis_title="Value")
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True, key="ov_prices")
        
        with col2:
            st.subheader("📊 Statistics")
            final_values = (analyzer.data.iloc[-1] / analyzer.data.iloc[0] - 1) * 100
            st.markdown("**🔥 Top 3**")
            for ticker, perf in final_values.nlargest(3).items():
                st.success(f"**{ticker}**: +{perf:.1f}%")
            st.markdown("**❄️ Bottom 3**")
            for ticker, perf in final_values.nsmallest(3).items():
                st.error(f"**{ticker}**: {perf:.1f}%")
            st.metric("Period", f"{len(analyzer.data)} days")
            st.metric("Assets", len(symbols))
        
        st.subheader("🔥 Correlation Heatmap")
        corr_matrix = analyzer.returns.corr()
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=500)
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True, key="ov_corr")
    
    # TAB 2: PORTFOLIOS
    with tab2:
        st.header("💼 PORTFOLIOS")
        st.subheader("📊 Comparison")
        st.dataframe(analyzer.compare_portfolios(), use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            port_names = [p['name'] for p in analyzer.portfolios.values()]
            ret_vals = [p['annualized_return'] * 100 for p in analyzer.portfolios.values()]
            fig = go.Figure(data=[go.Bar(x=port_names, y=ret_vals, marker_color=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(port_names))], text=[f"{v:.1f}%" for v in ret_vals], textposition='outside')])
            fig.update_layout(title="Annualized Returns", xaxis_tickangle=-45, height=400)
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True, key="p_returns")
        
        with col2:
            scatter_data = {'Portfolio': port_names, 'Volatility': [p['annualized_volatility']*100 for p in analyzer.portfolios.values()], 'Return': ret_vals, 'Sharpe': [p['sharpe_ratio'] for p in analyzer.portfolios.values()]}
            fig = px.scatter(scatter_data, x='Volatility', y='Return', text='Portfolio', size='Sharpe', color='Sharpe', color_continuous_scale='Viridis', size_max=40)
            fig.update_traces(textposition='top center', textfont_size=9)
            fig.update_layout(title="Risk-Return", height=400)
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True, key="p_scatter")
        
        st.subheader("🔍 Detail")
        portfolio_keys = list(analyzer.portfolios.keys())
        selected_p = st.selectbox("Select portfolio", portfolio_keys, format_func=lambda x: analyzer.portfolios[x]['name'])
        portfolio = analyzer.portfolios[selected_p]
        
        mcols = st.columns(6)
        for col, (lbl, val) in zip(mcols, [("Return", f"{portfolio['annualized_return']*100:.2f}%"), ("Volatility", f"{portfolio['annualized_volatility']*100:.2f}%"), ("Sharpe", f"{portfolio['sharpe_ratio']:.3f}"), ("Sortino", f"{portfolio['sortino_ratio']:.3f}"), ("Max DD", f"{portfolio['max_drawdown']*100:.2f}%"), ("Calmar", f"{portfolio['calmar_ratio']:.3f}")]):
            col.metric(lbl, val)
        
        weights_df = pd.DataFrame({'Asset': symbols, 'Weight (%)': portfolio['weights'] * 100}).sort_values('Weight (%)', ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(weights_df.style.format({'Weight (%)': '{:.2f}%'}).background_gradient(subset=['Weight (%)'], cmap='YlGn'), use_container_width=True, height=300)
        with col2:
            fig = px.treemap(weights_df[weights_df['Weight (%)'] > 0.1], path=[px.Constant("Portfolio"), 'Asset'], values='Weight (%)', color='Weight (%)', color_continuous_scale='Viridis')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="p_tree")
    
    # TAB 3: PERFORMANCE
    with tab3:
        st.header("📈 PERFORMANCE")
        sel_perf = st.multiselect("Compare portfolios", portfolio_keys, default=portfolio_keys[:4], format_func=lambda x: analyzer.portfolios[x]['name'])
        
        if sel_perf:
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Cumulative', 'Drawdown'), vertical_spacing=0.12, row_heights=[0.6, 0.4])
            for i, p_name in enumerate(sel_perf):
                p = analyzer.portfolios[p_name]
                cum_val = (1 + p['returns']).cumprod() * 100
                color = CHART_COLORS[i % len(CHART_COLORS)]
                fig.add_trace(go.Scatter(x=cum_val.index, y=cum_val.values, name=p['name'], mode='lines', line=dict(color=color, width=2)), row=1, col=1)
                roll_max = cum_val.expanding().max()
                dd = (cum_val - roll_max) / roll_max * 100
                fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=p['name'], mode='lines', line=dict(color=color, width=2), fill='tozeroy', showlegend=False), row=2, col=1)
            fig.update_layout(height=700, hovermode='x unified')
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True, key="perf_chart")
            
            stats = [{'Portfolio': analyzer.portfolios[p]['name'], 'Return': f"{analyzer.portfolios[p]['annualized_return']*100:.2f}%", 'Vol': f"{analyzer.portfolios[p]['annualized_volatility']*100:.2f}%", 'Sharpe': f"{analyzer.portfolios[p]['sharpe_ratio']:.3f}", 'Max DD': f"{analyzer.portfolios[p]['max_drawdown']*100:.2f}%"} for p in sel_perf]
            st.dataframe(pd.DataFrame(stats), use_container_width=True)
# Part 5: Remaining Tabs

    # TAB 4: ROLLING
    with tab4:
        st.header("🔄 ROLLING ANALYSIS")
        st.info(f"Window: **{window_years} years** ({window_years * 252} days)")
        sel_roll = st.multiselect("Select portfolios", portfolio_keys, default=portfolio_keys[:3], format_func=lambda x: analyzer.portfolios[x]['name'], key="roll_sel")
        
        if sel_roll:
            rolling_data = {}
            window = window_years * 252
            for p_name in sel_roll:
                p = analyzer.portfolios[p_name]
                rets = p['returns']
                if len(rets) >= window:
                    roll_ret = rets.rolling(window=window).apply(lambda x: (1+x).prod()**(252/len(x))-1 if len(x)==window else np.nan)
                    roll_vol = rets.rolling(window=window).std() * np.sqrt(252)
                    roll_sharpe = (roll_ret - risk_free_rate) / roll_vol
                    rolling_data[p['name']] = pd.DataFrame({'Return': roll_ret, 'Volatility': roll_vol, 'Sharpe': roll_sharpe}).dropna()
            
            if rolling_data:
                fig = make_subplots(rows=3, cols=1, subplot_titles=('Returns', 'Volatility', 'Sharpe'), vertical_spacing=0.08)
                for i, (name, data) in enumerate(rolling_data.items()):
                    color = CHART_COLORS[i % len(CHART_COLORS)]
                    fig.add_trace(go.Scatter(x=data.index, y=data['Return']*100, name=name, line=dict(color=color, width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['Volatility']*100, name=name, line=dict(color=color, width=2), showlegend=False), row=2, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['Sharpe'], name=name, line=dict(color=color, width=2), showlegend=False), row=3, col=1)
                fig.update_layout(height=900, hovermode='x unified')
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True, key="roll_chart")
    
    # TAB 5: CORRELATIONS
    with tab5:
        st.header("🔗 CORRELATIONS")
        col1, col2 = st.columns([2, 1])
        with col1:
            corr = analyzer.returns.corr()
            fig = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig.update_layout(height=600)
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True, key="corr_mat")
        with col2:
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    pairs.append({'A1': corr.columns[i], 'A2': corr.columns[j], 'Corr': corr.iloc[i, j]})
            pairs_df = pd.DataFrame(pairs).sort_values('Corr', ascending=False)
            st.markdown("**🔥 Most Correlated**")
            for _, r in pairs_df.head(5).iterrows():
                st.success(f"{r['A1']} ↔ {r['A2']}: {r['Corr']:.3f}")
            st.markdown("**❄️ Least Correlated**")
            for _, r in pairs_df.tail(5).iterrows():
                st.warning(f"{r['A1']} ↔ {r['A2']}: {r['Corr']:.3f}")
    
    # TAB 6: EFFICIENT FRONTIER
    with tab6:
        st.header("📐 EFFICIENT FRONTIER")
        with st.spinner("Calculating..."):
            frontier_df = analyzer.get_efficient_frontier(n_points=50)
            if not frontier_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=frontier_df['Volatility']*100, y=frontier_df['Return']*100, mode='lines', name='Frontier', line=dict(color='#6366f1', width=3)))
                for i, (p_name, p) in enumerate(analyzer.portfolios.items()):
                    fig.add_trace(go.Scatter(x=[p['annualized_volatility']*100], y=[p['annualized_return']*100], mode='markers+text', name=p['name'], marker=dict(size=15, color=CHART_COLORS[i % len(CHART_COLORS)], symbol='diamond'), text=[p['name'].split()[0]], textposition='top center'))
                max_sharpe_p = max(analyzer.portfolios.values(), key=lambda x: x['sharpe_ratio'])
                cml_x = [0, max_sharpe_p['annualized_volatility']*100*2]
                cml_y = [risk_free_rate*100, risk_free_rate*100 + max_sharpe_p['sharpe_ratio']*cml_x[1]]
                fig.add_trace(go.Scatter(x=cml_x, y=cml_y, mode='lines', name='CML', line=dict(color='#10b981', width=2, dash='dash')))
                fig.update_layout(height=600, xaxis_title="Volatility (%)", yaxis_title="Return (%)")
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True, key="frontier")
                col1, col2, col3 = st.columns(3)
                col1.metric("Min Vol", f"{frontier_df['Volatility'].min()*100:.2f}%")
                col2.metric("Max Return", f"{frontier_df['Return'].max()*100:.2f}%")
                col3.metric("Max Sharpe", f"{frontier_df['Sharpe'].max():.3f}")
    
    # TAB 7: BENCHMARK (if enabled)
    if tab8 is not None:
        with tab7:
            st.header(f"🎯 BENCHMARK ({st.session_state.benchmark})")
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                bench_cumret = float((1 + benchmark_returns).prod() - 1)
                n_yrs = len(benchmark_returns) / 252
                bench_annret = float((1 + bench_cumret)**(1/n_yrs) - 1) if n_yrs > 0 else 0
                bench_vol = float(benchmark_returns.std() * np.sqrt(252))
                bench_sharpe = (bench_annret - risk_free_rate) / bench_vol if bench_vol > 0 else 0
                
                bcols = st.columns(4)
                bcols[0].metric("Return", f"{bench_annret*100:.2f}%")
                bcols[1].metric("Volatility", f"{bench_vol*100:.2f}%")
                bcols[2].metric("Sharpe", f"{bench_sharpe:.3f}")
                bcols[3].metric("Cumulative", f"{bench_cumret*100:.2f}%")
                
                sel_bench = st.multiselect("Compare with benchmark", portfolio_keys, default=portfolio_keys[:3], format_func=lambda x: analyzer.portfolios[x]['name'], key="bench_sel")
                if sel_bench:
                    fig = go.Figure()
                    bench_cum = (1 + benchmark_returns).cumprod() * 100
                    fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name=f"Benchmark ({st.session_state.benchmark})", mode='lines', line=dict(color='#94a3b8', width=3, dash='dash')))
                    for i, p_name in enumerate(sel_bench):
                        p = analyzer.portfolios[p_name]
                        cum_val = (1 + p['returns']).cumprod() * 100
                        fig.add_trace(go.Scatter(x=cum_val.index, y=cum_val.values, name=p['name'], mode='lines', line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2)))
                    fig.update_layout(height=500, hovermode='x unified')
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True, key="bench_chart")
            else:
                st.warning("No benchmark data")
    
    # EXPORT TAB
    export_tab = tab8 if tab8 is not None else tab7
    with export_tab:
        st.header("📥 EXPORT")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Full Report")
            if OPENPYXL_AVAILABLE:
                if st.button("Generate Excel", use_container_width=True, key="exp_full"):
                    with st.spinner("Creating..."):
                        filename = analyzer.export_to_excel()
                        with open(filename, 'rb') as f:
                            st.download_button("⬇️ Download", f, filename, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_full")
            else:
                st.warning("openpyxl not installed")
        
        with col2:
            st.subheader("🔗 JSON")
            if st.button("Generate JSON", use_container_width=True, key="exp_json"):
                export_data = {'metadata': {'date': datetime.now().isoformat(), 'assets': symbols}, 'portfolios': {}}
                for name, p in analyzer.portfolios.items():
                    export_data['portfolios'][name] = {'name': p['name'], 'weights': {s: float(w) for s, w in zip(symbols, p['weights'])}, 'metrics': {'return': float(p['annualized_return']), 'volatility': float(p['annualized_volatility']), 'sharpe': float(p['sharpe_ratio'])}}
                st.download_button("⬇️ Download JSON", json.dumps(export_data, indent=2), f"portfolio_{datetime.now().strftime('%Y%m%d')}.json", "application/json", key="dl_json")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-section">
    <h3>📊 PORTFOLIO ANALYZER PRO</h3>
    <p><strong>Powered by:</strong> Python • Streamlit • Plotly • NumPy • Pandas • SciPy</p>
    <p style="margin-top:1rem;opacity:0.6;">⚠️ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Reset", use_container_width=True, key="reset"):
        st.session_state.analyzer = None
        st.session_state.analysis_complete = False
        st.session_state.selected_tickers = []
        st.session_state.benchmark_returns = None
        st.rerun()
