import { useState, useEffect } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

// Colori coerenti con Streamlit
const CHART_COLORS = ['#FF6B6B','#4ECDC4','#FFE66D','#95E1D3','#F38181','#AA96DA','#FCBAD3','#A8D8EA','#FF9F43','#6C5CE7']

function App() {
  // State
  const [symbols, setSymbols] = useState('AAPL, MSFT, GOOGL, AMZN')
  const [startDate, setStartDate] = useState('2020-01-01')
  const [endDate, setEndDate] = useState('2024-01-01')
  const [riskFreeRate, setRiskFreeRate] = useState(0.02)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [overviewData, setOverviewData] = useState(null)
  const [performanceData, setPerformanceData] = useState(null)
  const [selectedStrategies, setSelectedStrategies] = useState([])
  const [frontierData, setFrontierData] = useState(null)
  const [frontierType, setFrontierType] = useState('mean_variance')
  const [frontierLoading, setFrontierLoading] = useState(false)
  const [nPortfolios, setNPortfolios] = useState(5000)
  const [allowShort, setAllowShort] = useState(false)
  const [cvarAlpha, setCvarAlpha] = useState(0.95)
  const [frontierPosition, setFrontierPosition] = useState(30)
  const [compareStrategy, setCompareStrategy] = useState(null)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')

  // Funzione per chiamare l'API
  const analyzePortfolio = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    setOverviewData(null)
    setPerformanceData(null)
    setSelectedStrategies([])

    try {
      const symbolList = symbols.split(',').map(s => s.trim().toUpperCase()).filter(s => s.length > 0)

      // Call all endpoints in parallel
      const [analyzeResponse, overviewResponse, performanceResponse] = await Promise.all([
        axios.post(`${API_URL}/api/portfolio/analyze`, {
          symbols: symbolList,
          start_date: startDate,
          end_date: endDate,
          risk_free_rate: riskFreeRate
        }),
        axios.post(`${API_URL}/api/portfolio/overview-data`, {
          symbols: symbolList,
          start_date: startDate,
          end_date: endDate,
          risk_free_rate: riskFreeRate
        }),
        axios.post(`${API_URL}/api/portfolio/performance-data?window_years=3`, {
          symbols: symbolList,
          start_date: startDate,
          end_date: endDate,
          risk_free_rate: riskFreeRate
        })
      ])

      setResult(analyzeResponse.data)
      setOverviewData(overviewResponse.data)
      setPerformanceData(performanceResponse.data)
      
      // Default: select first 4 strategies
      const strategyKeys = Object.keys(analyzeResponse.data.portfolios)
      setSelectedStrategies(strategyKeys.slice(0, 4))
      
      setActiveTab('overview')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  // Funzione per caricare i dati della frontiera
  const loadFrontierData = async () => {
    if (!result) return
    
    setFrontierLoading(true)
    try {
      const symbolList = symbols.split(',').map(s => s.trim().toUpperCase()).filter(s => s.length > 0)
      
      const response = await axios.post(
        `${API_URL}/api/portfolio/frontier-data?frontier_type=${frontierType}&n_portfolios=${nPortfolios}&allow_short=${allowShort}&cvar_alpha=${cvarAlpha}`,
        {
          symbols: symbolList,
          start_date: startDate,
          end_date: endDate,
          risk_free_rate: riskFreeRate
        }
      )
      
      setFrontierData(response.data)
      
      // Set default compare strategy
      if (!compareStrategy && response.data.strategy_points.length > 0) {
        setCompareStrategy(response.data.strategy_points[0].key)
      }
    } catch (err) {
      console.error('Frontier error:', err)
    } finally {
      setFrontierLoading(false)
    }
  }

  // Auto-reload frontier when settings change (if data was already loaded)
  useEffect(() => {
    // Only auto-reload if we already have frontier data
    if (frontierData !== null && result !== null) {
      // Use a small delay to avoid rapid re-fetches
      const timer = setTimeout(() => {
        loadFrontierData()
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [frontierType, nPortfolios, allowShort, cvarAlpha])

  // Prepara dati per grafico Asset Performance (Base 100)
  const getAssetPerformanceData = () => {
    if (!overviewData || !overviewData.price_series) return []
    
    return overviewData.price_series.map((asset, idx) => ({
      x: asset.prices.map(p => p.date),
      y: asset.prices.map(p => p.value),
      type: 'scatter',
      mode: 'lines',
      name: asset.ticker,
      line: { color: CHART_COLORS[idx % CHART_COLORS.length], width: 2.5 },
      hovertemplate: `<b>${asset.ticker}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>`
    }))
  }

  // Prepara dati per Correlation Heatmap
  const getCorrelationData = () => {
    if (!overviewData || !overviewData.correlation_matrix) return null
    
    const symbols = Object.keys(overviewData.correlation_matrix)
    const z = symbols.map(row => symbols.map(col => overviewData.correlation_matrix[row][col]))
    
    return { z, x: symbols, y: symbols }
  }

  // Prepara dati per grafico Returns (per tab Portfolios)
  const getReturnsChartData = () => {
    if (!result) return { x: [], y: [], colors: [] }
    
    const sorted = Object.entries(result.portfolios)
      .map(([key, p]) => ({ name: p.name.replace(' Portfolio', ''), ret: p.annualized_return * 100 }))
      .sort((a, b) => b.ret - a.ret)
    
    return {
      x: sorted.map(d => d.name),
      y: sorted.map(d => d.ret),
      colors: sorted.map((_, i) => CHART_COLORS[i % CHART_COLORS.length])
    }
  }

  // Prepara dati per Risk vs Return scatter
  const getRiskReturnData = () => {
    if (!result) return []
    
    return Object.entries(result.portfolios).map(([key, p], i) => ({
      x: [p.annualized_volatility * 100],
      y: [p.annualized_return * 100],
      name: p.name,
      color: CHART_COLORS[i % CHART_COLORS.length]
    }))
  }

  // Prepara dati per Cumulative Performance chart
  const getCumulativeData = () => {
    if (!performanceData || !performanceData.portfolio_series) return []
    
    return performanceData.portfolio_series
      .filter(p => selectedStrategies.includes(p.key))
      .map((portfolio, idx) => ({
        x: portfolio.dates,
        y: portfolio.cumulative_returns,
        type: 'scatter',
        mode: 'lines',
        name: portfolio.name,
        line: { color: CHART_COLORS[idx % CHART_COLORS.length], width: 2.5 },
        hovertemplate: `<b>${portfolio.name}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>`
      }))
  }

  // Prepara dati per Drawdown chart
  const getDrawdownData = () => {
    if (!performanceData || !performanceData.portfolio_series) return []
    
    return performanceData.portfolio_series
      .filter(p => selectedStrategies.includes(p.key))
      .map((portfolio, idx) => {
        const color = CHART_COLORS[idx % CHART_COLORS.length]
        // Convert hex to rgba for fill
        const r = parseInt(color.slice(1, 3), 16)
        const g = parseInt(color.slice(3, 5), 16)
        const b = parseInt(color.slice(5, 7), 16)
        
        return {
          x: portfolio.dates,
          y: portfolio.drawdown,
          type: 'scatter',
          mode: 'lines',
          name: portfolio.name,
          line: { color: color, width: 2 },
          fill: 'tozeroy',
          fillcolor: `rgba(${r},${g},${b},0.2)`,
          hovertemplate: `<b>${portfolio.name}</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>`
        }
      })
  }

  // Prepara dati per Rolling Returns chart
  const getRollingReturnsData = () => {
    if (!performanceData || !performanceData.rolling_metrics) return []
    
    return performanceData.rolling_metrics
      .filter(p => selectedStrategies.includes(p.key))
      .map((portfolio, idx) => ({
        x: portfolio.dates,
        y: portfolio.rolling_return,
        type: 'scatter',
        mode: 'lines',
        name: portfolio.name,
        line: { color: CHART_COLORS[idx % CHART_COLORS.length], width: 2.5 },
        connectgaps: false,
        hovertemplate: `<b>${portfolio.name}</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>`
      }))
  }

  // Prepara dati per Rolling Volatility chart
  const getRollingVolatilityData = () => {
    if (!performanceData || !performanceData.rolling_metrics) return []
    
    return performanceData.rolling_metrics
      .filter(p => selectedStrategies.includes(p.key))
      .map((portfolio, idx) => ({
        x: portfolio.dates,
        y: portfolio.rolling_volatility,
        type: 'scatter',
        mode: 'lines',
        name: portfolio.name,
        line: { color: CHART_COLORS[idx % CHART_COLORS.length], width: 2.5 },
        connectgaps: false,
        hovertemplate: `<b>${portfolio.name}</b><br>Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>`
      }))
  }

  // Toggle strategy selection
  const toggleStrategy = (key) => {
    if (selectedStrategies.includes(key)) {
      setSelectedStrategies(selectedStrategies.filter(k => k !== key))
    } else {
      setSelectedStrategies([...selectedStrategies, key])
    }
  }

  // Check if rolling data is available
  const hasRollingData = () => {
    if (!performanceData || !performanceData.rolling_metrics) return false
    return performanceData.rolling_metrics.some(m => 
      m.rolling_return.some(v => v !== null)
    )
  }

  // ==================== FRONTIER HELPER FUNCTIONS ====================
  
  // Prepara dati per il cloud di portfoli random
  const getRandomPortfoliosData = () => {
    if (!frontierData) return null
    
    return {
      x: frontierData.random_portfolios.map(p => p.risk_pct),
      y: frontierData.random_portfolios.map(p => p.return_pct),
      type: 'scatter',
      mode: 'markers',
      name: `Random Portfolios (${frontierData.n_portfolios.toLocaleString()})`,
      marker: {
        size: 3,
        color: frontierType === 'mean_variance' 
          ? 'rgba(255, 107, 107, 0.4)' 
          : 'rgba(147, 112, 219, 0.4)',
        symbol: 'circle'
      },
      hovertemplate: `Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>`
    }
  }

  // Prepara dati per le strategie ottimizzate
  const getStrategyPointsData = () => {
    if (!frontierData) return []
    
    const colors = ['#FFE66D', '#A855F7', '#6366F1', '#FF9F43', '#EC4899', '#10B981', '#F59E0B']
    const symbols_markers = ['star', 'diamond', 'hexagon', 'pentagon', 'circle', 'square', 'triangle-up']
    
    return frontierData.strategy_points.map((s, i) => ({
      x: [s.risk_pct],
      y: [s.return_pct],
      type: 'scatter',
      mode: 'markers+text',
      name: s.name,
      marker: {
        size: 18,
        color: colors[i % colors.length],
        symbol: symbols_markers[i % symbols_markers.length],
        line: { width: 2, color: 'white' }
      },
      text: [s.name.split(' ')[0]],
      textposition: 'top center',
      textfont: { color: '#E2E8F0', size: 9 },
      hovertemplate: `<b>${s.name}</b><br>Return: %{y:.2f}%<br>Risk: %{x:.2f}%<br>Sharpe: ${s.sharpe.toFixed(3)}<extra></extra>`
    }))
  }

  // Ottieni il punto della frontiera selezionato
  const getSelectedFrontierPoint = () => {
    if (!frontierData || frontierData.frontier_points.length === 0) return null
    
    const idx = Math.floor((frontierPosition / 100) * (frontierData.frontier_points.length - 1))
    return frontierData.frontier_points[idx]
  }

  // Ottieni il risk profile label
  const getRiskProfileLabel = () => {
    if (frontierPosition < 25) return { label: '🛡️ Conservative', color: '#4ECDC4' }
    if (frontierPosition < 50) return { label: '⚖️ Moderate', color: '#FFE66D' }
    if (frontierPosition < 75) return { label: '📈 Growth', color: '#FF9F43' }
    return { label: '🚀 Aggressive', color: '#FF6B6B' }
  }

  // Ottieni i dati della strategia di confronto
  const getCompareStrategyData = () => {
    if (!frontierData || !compareStrategy) return null
    return frontierData.strategy_points.find(s => s.key === compareStrategy)
  }

  // Calcola cumulative returns per i pesi del frontier portfolio selezionato
  const calculateFrontierCumulative = () => {
    if (!frontierData || !overviewData || !overviewData.price_series) return null
    
    const selectedPoint = getSelectedFrontierPoint()
    if (!selectedPoint) return null
    
    const weights = selectedPoint.weights
    const dates = overviewData.price_series[0].prices.map(p => p.date)
    
    // Calcola rendimenti giornalieri per ogni asset
    const assetReturns = overviewData.price_series.map(asset => {
      const returns = []
      for (let i = 1; i < asset.prices.length; i++) {
        returns.push((asset.prices[i].value - asset.prices[i-1].value) / asset.prices[i-1].value)
      }
      return returns
    })
    
    // Calcola rendimenti del portfolio
    const portfolioReturns = []
    for (let i = 0; i < assetReturns[0].length; i++) {
      let dayReturn = 0
      for (let j = 0; j < weights.length; j++) {
        dayReturn += weights[j] * assetReturns[j][i]
      }
      portfolioReturns.push(dayReturn)
    }
    
    // Calcola cumulativo (base 100)
    const cumulative = [100]
    for (let i = 0; i < portfolioReturns.length; i++) {
      cumulative.push(cumulative[cumulative.length - 1] * (1 + portfolioReturns[i]))
    }
    
    // Calcola metriche
    const annReturn = portfolioReturns.reduce((a, b) => a + b, 0) / portfolioReturns.length * 252
    const variance = portfolioReturns.reduce((sum, r) => sum + Math.pow(r - portfolioReturns.reduce((a, b) => a + b, 0) / portfolioReturns.length, 2), 0) / portfolioReturns.length
    const annVol = Math.sqrt(variance) * Math.sqrt(252)
    const sharpe = annVol > 0 ? (annReturn - riskFreeRate) / annVol : 0
    const cumReturn = (cumulative[cumulative.length - 1] / 100) - 1
    
    // Max Drawdown
    let maxDD = 0
    let peak = cumulative[0]
    for (let i = 1; i < cumulative.length; i++) {
      if (cumulative[i] > peak) peak = cumulative[i]
      const dd = (cumulative[i] - peak) / peak
      if (dd < maxDD) maxDD = dd
    }
    
    return {
      dates: dates,
      values: cumulative,
      metrics: {
        cumReturn: cumReturn * 100,
        annReturn: annReturn * 100,
        annVol: annVol * 100,
        sharpe: sharpe,
        maxDD: maxDD * 100
      }
    }
  }

  // Calcola il rendimento cumulativo per un set di pesi
  const calculateCumulativeForWeights = (weights) => {
    if (!overviewData || !overviewData.price_series) return null
    
    // Semplificazione: usiamo i rendimenti giornalieri per calcolare il cumulativo
    // In un'implementazione completa, dovremmo chiamare l'API
    const dates = overviewData.price_series[0].prices.map(p => p.date)
    const n = dates.length
    
    // Calcola rendimenti giornalieri per ogni asset
    const assetReturns = overviewData.price_series.map(asset => {
      const returns = []
      for (let i = 1; i < asset.prices.length; i++) {
        returns.push((asset.prices[i].value - asset.prices[i-1].value) / asset.prices[i-1].value)
      }
      return returns
    })
    
    // Calcola rendimenti del portfolio
    const portfolioReturns = []
    for (let i = 0; i < assetReturns[0].length; i++) {
      let dayReturn = 0
      for (let j = 0; j < weights.length; j++) {
        dayReturn += weights[j] * assetReturns[j][i]
      }
      portfolioReturns.push(dayReturn)
    }
    
    // Calcola cumulativo (base 100)
    const cumulative = [100]
    for (let i = 0; i < portfolioReturns.length; i++) {
      cumulative.push(cumulative[cumulative.length - 1] * (1 + portfolioReturns[i]))
    }
    
    return {
      dates: dates,
      values: cumulative
    }
  }

  // Tab button component
  const TabButton = ({ id, label, icon }) => (
    <button
      className={`tab-button ${activeTab === id ? 'active' : ''}`}
      onClick={() => setActiveTab(id)}
    >
      {icon} {label}
    </button>
  )

  // Plotly layout comune
  const plotlyLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#E2E8F0', family: 'Inter, sans-serif' },
    margin: { t: 40, b: 60, l: 60, r: 40 },
    xaxis: { gridcolor: 'rgba(99,102,241,0.15)', linecolor: 'rgba(99,102,241,0.3)' },
    yaxis: { gridcolor: 'rgba(99,102,241,0.15)', linecolor: 'rgba(99,102,241,0.3)' }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>📊 PORTFOLIO ANALYZER PRO</h1>
        <p>Advanced Multi-Asset Optimization System</p>
      </header>

      <div className="layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <h3>⚙️ Configuration</h3>
          
          <div className="input-group">
            <label>Symbols (comma-separated)</label>
            <input
              type="text"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="AAPL, MSFT, GOOGL"
            />
          </div>

          <div className="input-group">
            <label>Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>Risk-Free Rate: {(riskFreeRate * 100).toFixed(1)}%</label>
            <input
              type="range"
              min="0"
              max="0.1"
              step="0.005"
              value={riskFreeRate}
              onChange={(e) => setRiskFreeRate(parseFloat(e.target.value))}
            />
          </div>

          <button 
            className="analyze-button"
            onClick={analyzePortfolio} 
            disabled={loading}
          >
            {loading ? '⏳ Analyzing...' : '🚀 START ANALYSIS'}
          </button>

          {error && <div className="error">❌ {error}</div>}

          {result && (
            <div className="info-box">
              <p><strong>📊 {result.symbols.length}</strong> assets</p>
              <p><strong>📅 {result.trading_days}</strong> trading days</p>
              <p><strong>📈 {Object.keys(result.portfolios).length}</strong> strategies</p>
            </div>
          )}
        </aside>

        {/* Main Content */}
        <main className="main-content">
          {!result ? (
            <div className="welcome">
              <h2>👋 Welcome to Portfolio Analyzer Pro</h2>
              <p>Enter asset symbols and date range in the sidebar, then click "START ANALYSIS".</p>
              <div className="features">
                <div className="feature">
                  <span className="feature-icon">📊</span>
                  <h4>7 Strategies</h4>
                  <p>Equal Weight, Min Vol, Max Sharpe, Risk Parity, HRP, and more</p>
                </div>
                <div className="feature">
                  <span className="feature-icon">📈</span>
                  <h4>Complete Metrics</h4>
                  <p>Sharpe, Sortino, Calmar, VaR, CVaR, Max Drawdown</p>
                </div>
                <div className="feature">
                  <span className="feature-icon">🎯</span>
                  <h4>Advanced Analysis</h4>
                  <p>Correlation, Efficient Frontier, Backtest Validation</p>
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* KPI Dashboard */}
              <div className="kpi-dashboard">
                <div className="kpi-card">
                  <span className="kpi-label">🏆 Best Return</span>
                  <span className="kpi-value">
                    {(Math.max(...Object.values(result.portfolios).map(p => p.annualized_return)) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">⭐ Best Sharpe</span>
                  <span className="kpi-value">
                    {Math.max(...Object.values(result.portfolios).map(p => p.sharpe_ratio)).toFixed(2)}
                  </span>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">🛡️ Min Volatility</span>
                  <span className="kpi-value">
                    {(Math.min(...Object.values(result.portfolios).map(p => p.annualized_volatility)) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">📉 Best Max DD</span>
                  <span className="kpi-value">
                    {(Math.max(...Object.values(result.portfolios).map(p => p.max_drawdown)) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Tab Navigation */}
              <nav className="tabs">
                <TabButton id="overview" label="Overview" icon="📊" />
                <TabButton id="portfolios" label="Portfolios" icon="💼" />
                <TabButton id="performance" label="Performance" icon="📈" />
                <TabButton id="deepdive" label="Deep-dive" icon="🔬" />
                <TabButton id="backtest" label="Backtest" icon="🧪" />
                <TabButton id="frontier" label="Frontier" icon="📐" />
                <TabButton id="export" label="Export" icon="📥" />
              </nav>

              {/* Tab Content */}
              <div className="tab-content">
                
                {/* ==================== TAB: OVERVIEW ==================== */}
                {activeTab === 'overview' && overviewData && (
                  <div className="tab-panel">
                    <h2>📊 Overview</h2>
                    
                    {/* Asset Performance Chart */}
                    <div className="section">
                      <h3>📈 Asset Performance (Base 100)</h3>
                      <div className="chart-container">
                        <Plot
                          data={getAssetPerformanceData()}
                          layout={{
                            ...plotlyLayout,
                            height: 450,
                            xaxis: { 
                              ...plotlyLayout.xaxis, 
                              title: 'Date',
                              tickformat: '%Y-%m'
                            },
                            yaxis: { 
                              ...plotlyLayout.yaxis, 
                              title: 'Value (Base 100)' 
                            },
                            hovermode: 'x unified',
                            legend: {
                              orientation: 'h',
                              yanchor: 'bottom',
                              y: 1.02,
                              xanchor: 'center',
                              x: 0.5,
                              font: { color: '#E2E8F0' }
                            },
                            shapes: [{
                              type: 'line',
                              x0: overviewData.period.start,
                              x1: overviewData.period.end,
                              y0: 100,
                              y1: 100,
                              line: { color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dash' }
                            }]
                          }}
                          config={{ responsive: true, displayModeBar: false }}
                          style={{ width: '100%' }}
                        />
                      </div>
                    </div>

                    {/* Statistics Row */}
                    <div className="section">
                      <div className="stats-row">
                        {/* Top/Bottom Performers */}
                        <div className="stats-card">
                          <h4>📊 Statistics</h4>
                          
                          <div className="performers-section">
                            <p className="performers-title">🔥 Top 3</p>
                            {overviewData.top_performers.map((p, idx) => (
                              <div key={idx} className={`performer-item ${p.return >= 0 ? 'positive' : 'negative'}`}>
                                <span className="performer-ticker">{p.ticker}</span>
                                <span className="performer-return">
                                  {p.return >= 0 ? '+' : ''}{p.return.toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                          
                          <div className="performers-section">
                            <p className="performers-title">❄️ Bottom 3</p>
                            {overviewData.bottom_performers.map((p, idx) => (
                              <div key={idx} className={`performer-item ${p.return >= 0 ? 'positive' : 'negative'}`}>
                                <span className="performer-ticker">{p.ticker}</span>
                                <span className="performer-return">
                                  {p.return >= 0 ? '+' : ''}{p.return.toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                          
                          <div className="stats-metrics">
                            <div className="stat-item">
                              <span className="stat-label">📅 Days</span>
                              <span className="stat-value">{overviewData.trading_days}</span>
                            </div>
                            <div className="stat-item">
                              <span className="stat-label">📦 Assets</span>
                              <span className="stat-value">{overviewData.symbols.length}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Individual Asset Statistics Table */}
                    <div className="section">
                      <h3>📋 Individual Asset Statistics</h3>
                      <div className="table-container">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>Asset</th>
                              <th>Ticker</th>
                              <th>Return</th>
                              <th>Vol</th>
                              <th>Sharpe</th>
                              <th>Sortino</th>
                              <th>Max DD</th>
                              <th>Calmar</th>
                            </tr>
                          </thead>
                          <tbody>
                            {overviewData.asset_stats.map((asset, idx) => (
                              <tr key={asset.ticker}>
                                <td>{idx + 1}</td>
                                <td>{asset.name}</td>
                                <td>{asset.ticker}</td>
                                <td className={asset.annualized_return >= 0 ? 'positive' : 'negative'}>
                                  {(asset.annualized_return * 100).toFixed(2)}%
                                </td>
                                <td>{(asset.annualized_volatility * 100).toFixed(2)}%</td>
                                <td>{asset.sharpe_ratio.toFixed(3)}</td>
                                <td>{asset.sortino_ratio.toFixed(3)}</td>
                                <td className="negative">{(asset.max_drawdown * 100).toFixed(2)}%</td>
                                <td>{asset.calmar_ratio.toFixed(3)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="table-caption">Ranked by Sharpe Ratio</p>
                    </div>

                    {/* Correlation Heatmap */}
                    <div className="section">
                      <h3>🔥 Correlation Heatmap</h3>
                      {getCorrelationData() && (
                        <div className="chart-container">
                          <Plot
                            data={[{
                              type: 'heatmap',
                              z: getCorrelationData().z,
                              x: getCorrelationData().x,
                              y: getCorrelationData().y,
                              colorscale: [
                                [0, '#3b82f6'],      // -1 = Blue
                                [0.5, '#1e1e2e'],    // 0 = Dark
                                [1, '#ef4444']       // +1 = Red
                              ],
                              zmin: -1,
                              zmax: 1,
                              text: getCorrelationData().z.map(row => row.map(v => v.toFixed(2))),
                              texttemplate: '%{text}',
                              textfont: { size: 11, color: '#fff' },
                              hovertemplate: '%{x} vs %{y}: %{z:.3f}<extra></extra>',
                              showscale: true,
                              colorbar: {
                                tickfont: { color: '#E2E8F0' },
                                title: { text: 'Correlation', font: { color: '#E2E8F0' } }
                              }
                            }]}
                            layout={{
                              ...plotlyLayout,
                              height: 400,
                              xaxis: { 
                                ...plotlyLayout.xaxis, 
                                tickangle: -45,
                                side: 'bottom'
                              },
                              yaxis: { 
                                ...plotlyLayout.yaxis, 
                                autorange: 'reversed' 
                              },
                              margin: { t: 20, b: 80, l: 80, r: 100 }
                            }}
                            config={{ responsive: true, displayModeBar: false }}
                            style={{ width: '100%' }}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* ==================== TAB: PORTFOLIOS ==================== */}
                {activeTab === 'portfolios' && (
                  <div className="tab-panel">
                    <h2>💼 Portfolio Analysis</h2>
                    
                    <div className="section">
                      <h3>🏆 Strategy Ranking</h3>
                      <div className="table-container">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>Strategy</th>
                              <th>Return</th>
                              <th>Volatility</th>
                              <th>Sharpe</th>
                              <th>Sortino</th>
                              <th>Max DD</th>
                              <th>Calmar</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(result.portfolios)
                              .sort((a, b) => b[1].sharpe_ratio - a[1].sharpe_ratio)
                              .map(([key, p], idx) => (
                                <tr key={key}>
                                  <td>{idx + 1}</td>
                                  <td>{p.name}</td>
                                  <td className={p.annualized_return >= 0 ? 'positive' : 'negative'}>
                                    {(p.annualized_return * 100).toFixed(2)}%
                                  </td>
                                  <td>{(p.annualized_volatility * 100).toFixed(2)}%</td>
                                  <td>{p.sharpe_ratio.toFixed(3)}</td>
                                  <td>{p.sortino_ratio.toFixed(3)}</td>
                                  <td className="negative">{(p.max_drawdown * 100).toFixed(2)}%</td>
                                  <td>{p.calmar_ratio.toFixed(3)}</td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="table-caption">Ranked by Sharpe Ratio (Theoretical)</p>
                    </div>

                    <div className="charts-row">
                      <div className="chart-half">
                        <h3>📊 Returns by Strategy</h3>
                        <div className="chart-container">
                          <Plot
                            data={[{
                              type: 'bar',
                              x: getReturnsChartData().x,
                              y: getReturnsChartData().y,
                              marker: { color: getReturnsChartData().colors },
                              text: getReturnsChartData().y.map(v => `${v.toFixed(1)}%`),
                              textposition: 'outside',
                              textfont: { color: '#E2E8F0', size: 10 }
                            }]}
                            layout={{
                              ...plotlyLayout,
                              height: 350,
                              xaxis: { ...plotlyLayout.xaxis, tickangle: -45 },
                              yaxis: { ...plotlyLayout.yaxis, title: 'Annualized Return (%)' }
                            }}
                            config={{ responsive: true, displayModeBar: false }}
                            style={{ width: '100%' }}
                          />
                        </div>
                      </div>

                      <div className="chart-half">
                        <h3>⚖️ Risk vs Return</h3>
                        <div className="chart-container">
                          <Plot
                            data={getRiskReturnData().map((d, i) => ({
                              type: 'scatter',
                              mode: 'markers+text',
                              x: d.x,
                              y: d.y,
                              name: d.name,
                              marker: { size: 16, color: d.color },
                              text: [d.name.split(' ')[0]],
                              textposition: 'top center',
                              textfont: { color: '#E2E8F0', size: 9 },
                              hovertemplate: `<b>${d.name}</b><br>Return: %{y:.2f}%<br>Vol: %{x:.2f}%<extra></extra>`
                            }))}
                            layout={{
                              ...plotlyLayout,
                              height: 350,
                              showlegend: false,
                              xaxis: { ...plotlyLayout.xaxis, title: 'Volatility (%)' },
                              yaxis: { ...plotlyLayout.yaxis, title: 'Return (%)' }
                            }}
                            config={{ responsive: true, displayModeBar: false }}
                            style={{ width: '100%' }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="section">
                      <h3>⚖️ Portfolio Weights</h3>
                      <div className="portfolios-grid">
                        {Object.entries(result.portfolios).map(([key, p], idx) => (
                          <div key={key} className="portfolio-card" style={{ borderLeftColor: CHART_COLORS[idx % CHART_COLORS.length] }}>
                            <h4>{p.name}</h4>
                            <div className="weights-list">
                              {Object.entries(p.weights)
                                .filter(([_, w]) => w > 0.001)
                                .sort((a, b) => b[1] - a[1])
                                .map(([symbol, weight]) => (
                                  <div key={symbol} className="weight-item">
                                    <span className="weight-symbol">{symbol}</span>
                                    <div className="weight-bar-container">
                                      <div 
                                        className="weight-bar-fill" 
                                        style={{ 
                                          width: `${weight * 100}%`,
                                          backgroundColor: CHART_COLORS[idx % CHART_COLORS.length]
                                        }} 
                                      />
                                    </div>
                                    <span className="weight-value">{(weight * 100).toFixed(1)}%</span>
                                  </div>
                                ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* ==================== TAB: PERFORMANCE ==================== */}
                {activeTab === 'performance' && performanceData && (
                  <div className="tab-panel">
                    <h2>📈 Performance Analysis</h2>
                    
                    <p className="tab-description">
                      This section analyzes the performance of selected strategies from different angles:
                      cumulative capital growth, drawdown phases, and the consistency of returns
                      and volatility over time through rolling windows.
                    </p>

                    {/* Strategy Selector */}
                    <div className="section">
                      <h3>🎯 Select Strategies to Compare</h3>
                      <div className="strategy-selector">
                        {result && Object.entries(result.portfolios).map(([key, p], idx) => (
                          <button
                            key={key}
                            className={`strategy-chip ${selectedStrategies.includes(key) ? 'selected' : ''}`}
                            onClick={() => toggleStrategy(key)}
                            style={{
                              borderColor: selectedStrategies.includes(key) 
                                ? CHART_COLORS[idx % CHART_COLORS.length] 
                                : 'var(--border-color)',
                              backgroundColor: selectedStrategies.includes(key)
                                ? `${CHART_COLORS[idx % CHART_COLORS.length]}22`
                                : 'transparent'
                            }}
                          >
                            <span 
                              className="strategy-chip-dot"
                              style={{ backgroundColor: CHART_COLORS[idx % CHART_COLORS.length] }}
                            />
                            {p.name.replace(' Portfolio', '')}
                          </button>
                        ))}
                      </div>
                    </div>

                    {selectedStrategies.length === 0 ? (
                      <div className="empty-state">
                        <p>👆 Select at least one strategy to view performance analysis</p>
                      </div>
                    ) : (
                      <>
                        {/* Cumulative Performance */}
                        <div className="section">
                          <h3>📊 Capital Growth (Base 100)</h3>
                          <div className="info-card">
                            <p>
                              <strong>How to read this chart:</strong> it shows how an initial investment of 100 
                              would have grown over time for each strategy. A line reaching 150 means the capital 
                              grew by 50%. The steeper the upward slope, the better the performance during that period.
                            </p>
                          </div>
                          <div className="chart-container">
                            <Plot
                              data={getCumulativeData()}
                              layout={{
                                ...plotlyLayout,
                                height: 450,
                                xaxis: { 
                                  ...plotlyLayout.xaxis, 
                                  title: 'Date',
                                  tickformat: '%Y-%m'
                                },
                                yaxis: { 
                                  ...plotlyLayout.yaxis, 
                                  title: 'Value (Base 100)' 
                                },
                                hovermode: 'x unified',
                                legend: {
                                  orientation: 'h',
                                  yanchor: 'bottom',
                                  y: 1.02,
                                  xanchor: 'center',
                                  x: 0.5,
                                  font: { color: '#E2E8F0' }
                                },
                                shapes: [{
                                  type: 'line',
                                  x0: performanceData.period.start,
                                  x1: performanceData.period.end,
                                  y0: 100,
                                  y1: 100,
                                  line: { color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dash' }
                                }],
                                annotations: [{
                                  x: performanceData.period.end,
                                  y: 100,
                                  xanchor: 'right',
                                  yanchor: 'bottom',
                                  text: 'Initial capital',
                                  showarrow: false,
                                  font: { color: 'rgba(255,255,255,0.5)', size: 10 }
                                }]
                              }}
                              config={{ responsive: true, displayModeBar: false }}
                              style={{ width: '100%' }}
                            />
                          </div>
                          
                          {/* Summary metrics */}
                          <div className="performance-summary">
                            {performanceData.portfolio_series
                              .filter(p => selectedStrategies.includes(p.key))
                              .map((p, idx) => {
                                const finalValue = p.cumulative_returns[p.cumulative_returns.length - 1]
                                const totalReturn = finalValue - 100
                                return (
                                  <div key={p.key} className="perf-metric-card">
                                    <span 
                                      className="perf-metric-dot"
                                      style={{ backgroundColor: CHART_COLORS[idx % CHART_COLORS.length] }}
                                    />
                                    <span className="perf-metric-name">{p.name.replace(' Portfolio', '')}</span>
                                    <span className={`perf-metric-value ${totalReturn >= 0 ? 'positive' : 'negative'}`}>
                                      {totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(1)}%
                                    </span>
                                  </div>
                                )
                              })}
                          </div>
                        </div>

                        {/* Drawdown Analysis */}
                        <div className="section">
                          <h3>📉 Drawdown Analysis</h3>
                          <div className="info-card">
                            <p>
                              <strong>How to read this chart:</strong> drawdown measures how much the portfolio 
                              has declined from its historical peak. A drawdown of -20% means that at that moment 
                              you had lost 20% from the peak. It's a measure of investment "pain": deep and prolonged 
                              drawdowns are psychologically difficult to endure.
                            </p>
                          </div>
                          <div className="chart-container">
                            <Plot
                              data={getDrawdownData()}
                              layout={{
                                ...plotlyLayout,
                                height: 380,
                                xaxis: { 
                                  ...plotlyLayout.xaxis, 
                                  title: 'Date',
                                  tickformat: '%Y-%m'
                                },
                                yaxis: { 
                                  ...plotlyLayout.yaxis, 
                                  title: 'Drawdown (%)' 
                                },
                                hovermode: 'x unified',
                                legend: {
                                  orientation: 'h',
                                  yanchor: 'bottom',
                                  y: 1.02,
                                  xanchor: 'center',
                                  x: 0.5,
                                  font: { color: '#E2E8F0' }
                                }
                              }}
                              config={{ responsive: true, displayModeBar: false }}
                              style={{ width: '100%' }}
                            />
                          </div>

                          {/* Drawdown Summary Table */}
                          <div className="table-container">
                            <table className="data-table">
                              <thead>
                                <tr>
                                  <th>Strategy</th>
                                  <th>Max Drawdown</th>
                                  <th>Max DD Date</th>
                                  <th>Recovery Time</th>
                                </tr>
                              </thead>
                              <tbody>
                                {performanceData.drawdown_stats
                                  .filter(d => selectedStrategies.includes(d.key))
                                  .map((d) => (
                                    <tr key={d.key}>
                                      <td>{d.name}</td>
                                      <td className="negative">{d.max_drawdown.toFixed(2)}%</td>
                                      <td>{d.max_drawdown_date}</td>
                                      <td>{d.recovery_time}</td>
                                    </tr>
                                  ))}
                              </tbody>
                            </table>
                          </div>
                          <p className="table-caption">Drawdown Summary</p>
                        </div>

                        {/* Rolling Analysis */}
                        <div className="section">
                          <h3>🔄 Rolling Analysis (Moving Windows)</h3>
                          <div className="info-card">
                            <p>
                              <strong>What are rolling metrics?</strong> Instead of calculating a single number for the entire 
                              period, rolling metrics calculate values over a moving "window" of <strong>{Math.round(performanceData.window_days / 252)} years</strong> ({performanceData.window_days} days)
                              that slides through time. This shows how performance and risk have changed across different 
                              market phases, revealing whether a strategy is consistent or only worked in certain periods.
                            </p>
                          </div>

                          {!hasRollingData() ? (
                            <div className="warning-card">
                              <p>⚠️ <strong>Insufficient data for rolling analysis.</strong></p>
                              <p>At least {Math.round(performanceData.window_days / 252)} years of data ({performanceData.window_days} trading days) required, 
                              but only {performanceData.trading_days} available.</p>
                            </div>
                          ) : (
                            <>
                              {/* Rolling Returns */}
                              <h4>📈 Rolling Returns (annualized)</h4>
                              <p className="chart-subtitle">Annualized return over {Math.round(performanceData.window_days / 252)}-year rolling window</p>
                              <div className="chart-container">
                                <Plot
                                  data={getRollingReturnsData()}
                                  layout={{
                                    ...plotlyLayout,
                                    height: 380,
                                    xaxis: { 
                                      ...plotlyLayout.xaxis, 
                                      title: 'Date',
                                      tickformat: '%Y-%m'
                                    },
                                    yaxis: { 
                                      ...plotlyLayout.yaxis, 
                                      title: 'Return (%)' 
                                    },
                                    hovermode: 'x unified',
                                    legend: {
                                      orientation: 'h',
                                      yanchor: 'bottom',
                                      y: 1.02,
                                      xanchor: 'center',
                                      x: 0.5,
                                      font: { color: '#E2E8F0' }
                                    },
                                    shapes: [{
                                      type: 'line',
                                      x0: performanceData.period.start,
                                      x1: performanceData.period.end,
                                      y0: 0,
                                      y1: 0,
                                      line: { color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dash' }
                                    }]
                                  }}
                                  config={{ responsive: true, displayModeBar: false }}
                                  style={{ width: '100%' }}
                                />
                              </div>

                              {/* Rolling Volatility */}
                              <h4>📊 Rolling Volatility (annualized)</h4>
                              <p className="chart-subtitle">Annualized volatility over {Math.round(performanceData.window_days / 252)}-year rolling window</p>
                              <div className="chart-container">
                                <Plot
                                  data={getRollingVolatilityData()}
                                  layout={{
                                    ...plotlyLayout,
                                    height: 380,
                                    xaxis: { 
                                      ...plotlyLayout.xaxis, 
                                      title: 'Date',
                                      tickformat: '%Y-%m'
                                    },
                                    yaxis: { 
                                      ...plotlyLayout.yaxis, 
                                      title: 'Volatility (%)' 
                                    },
                                    hovermode: 'x unified',
                                    legend: {
                                      orientation: 'h',
                                      yanchor: 'bottom',
                                      y: 1.02,
                                      xanchor: 'center',
                                      x: 0.5,
                                      font: { color: '#E2E8F0' }
                                    }
                                  }}
                                  config={{ responsive: true, displayModeBar: false }}
                                  style={{ width: '100%' }}
                                />
                              </div>

                              {/* Rolling Statistics Table */}
                              {performanceData.rolling_summary && performanceData.rolling_summary.length > 0 && (
                                <>
                                  <h4>📋 Rolling Statistics</h4>
                                  <div className="table-container">
                                    <table className="data-table">
                                      <thead>
                                        <tr>
                                          <th>Strategy</th>
                                          <th>Avg Return</th>
                                          <th>Min Return</th>
                                          <th>Max Return</th>
                                          <th>Avg Vol</th>
                                          <th>Min Vol</th>
                                          <th>Max Vol</th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {performanceData.rolling_summary
                                          .filter(s => selectedStrategies.includes(s.key))
                                          .map((s) => (
                                            <tr key={s.key}>
                                              <td>{s.name}</td>
                                              <td className={s.avg_return >= 0 ? 'positive' : 'negative'}>
                                                {s.avg_return.toFixed(2)}%
                                              </td>
                                              <td className={s.min_return >= 0 ? 'positive' : 'negative'}>
                                                {s.min_return.toFixed(2)}%
                                              </td>
                                              <td className={s.max_return >= 0 ? 'positive' : 'negative'}>
                                                {s.max_return.toFixed(2)}%
                                              </td>
                                              <td>{s.avg_vol.toFixed(2)}%</td>
                                              <td>{s.min_vol.toFixed(2)}%</td>
                                              <td>{s.max_vol.toFixed(2)}%</td>
                                            </tr>
                                          ))}
                                      </tbody>
                                    </table>
                                  </div>
                                  <p className="table-caption">Statistics over {Math.round(performanceData.window_days / 252)}-year rolling window</p>
                                </>
                              )}

                              {/* Interpretation Guide */}
                              <div className="info-card expandable">
                                <details>
                                  <summary><strong>📖 How to interpret rolling analysis</strong></summary>
                                  <div className="details-content">
                                    <h5>What to look for in rolling charts:</h5>
                                    <p><strong>Rolling Returns:</strong></p>
                                    <ul>
                                      <li>Stable lines above zero = consistent performance over time</li>
                                      <li>High variability = the strategy only works in certain market regimes</li>
                                      <li>Prolonged periods below zero = phases that are psychologically difficult to endure</li>
                                    </ul>
                                    <p><strong>Rolling Volatility:</strong></p>
                                    <ul>
                                      <li>Flat lines = predictable and stable risk</li>
                                      <li>Sudden spikes = the strategy becomes riskier during stress periods</li>
                                      <li>Increasing volatility over time = potential warning signal</li>
                                    </ul>
                                    <h5>Comparing strategies:</h5>
                                    <p>An ideal strategy shows:</p>
                                    <ul>
                                      <li>Consistently positive rolling returns</li>
                                      <li>Stable and contained rolling volatility</li>
                                      <li>Low correlation between volatility spikes and return drops</li>
                                    </ul>
                                  </div>
                                </details>
                              </div>
                            </>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                )}

                {/* ==================== TAB: DEEP-DIVE ==================== */}
                {activeTab === 'deepdive' && (
                  <div className="tab-panel">
                    <h2>🔬 Deep-Dive Statistics</h2>
                    <div className="coming-soon">
                      <h3>🚧 Coming Soon</h3>
                      <p>This tab will include:</p>
                      <ul>
                        <li>📈 Single Asset Analysis</li>
                        <li>📊 Distribution Analysis & Normality Tests</li>
                        <li>⚡ GARCH Volatility Modeling</li>
                        <li>🔗 DCC Correlation Dynamics</li>
                      </ul>
                    </div>
                  </div>
                )}

                {/* ==================== TAB: BACKTEST ==================== */}
                {activeTab === 'backtest' && (
                  <div className="tab-panel">
                    <h2>🧪 Backtest Validation</h2>
                    <div className="coming-soon">
                      <h3>🚧 Coming Soon</h3>
                      <p>This tab will include:</p>
                      <ul>
                        <li>📊 Walk-Forward Analysis</li>
                        <li>🧪 Combinatorial Purged Cross-Validation (CPCV)</li>
                        <li>📉 Probability of Backtest Overfitting (PBO)</li>
                      </ul>
                    </div>
                  </div>
                )}

                {/* ==================== TAB: FRONTIER ==================== */}
                {activeTab === 'frontier' && (
                  <div className="tab-panel">
                    <h2>📐 Efficient Frontier</h2>
                    
                    <p className="tab-description">
                      The <strong>Efficient Frontier</strong> represents the set of optimal portfolios that offer 
                      the highest expected return for a given level of risk (or the lowest risk for a given return level).
                    </p>

                    {/* Frontier Type Selector */}
                    <div className="section">
                      <div className="frontier-type-selector">
                        <button
                          className={`frontier-type-btn ${frontierType === 'mean_variance' ? 'active' : ''}`}
                          onClick={() => setFrontierType('mean_variance')}
                        >
                          📊 Mean-Variance (Traditional)
                        </button>
                        <button
                          className={`frontier-type-btn ${frontierType === 'mean_cvar' ? 'active' : ''}`}
                          onClick={() => setFrontierType('mean_cvar')}
                        >
                          📉 Mean-CVaR (Tail Risk)
                        </button>
                      </div>
                      
                      {frontierLoading && (
                        <div className="frontier-loading">
                          ⏳ Regenerating frontier...
                        </div>
                      )}
                    </div>

                    {/* Configuration */}
                    <div className="section">
                      <div className="frontier-config">
                        <div className="config-item">
                          <label>Random Portfolios</label>
                          <select 
                            value={nPortfolios} 
                            onChange={(e) => setNPortfolios(parseInt(e.target.value))}
                          >
                            <option value={2000}>2,000</option>
                            <option value={5000}>5,000</option>
                            <option value={10000}>10,000</option>
                            <option value={20000}>20,000</option>
                          </select>
                        </div>
                        
                        <div className="config-item">
                          <label>
                            <input
                              type="checkbox"
                              checked={allowShort}
                              onChange={(e) => setAllowShort(e.target.checked)}
                            />
                            Allow Short Selling
                          </label>
                        </div>
                        
                        {frontierType === 'mean_cvar' && (
                          <div className="config-item">
                            <label>CVaR Confidence</label>
                            <select 
                              value={cvarAlpha} 
                              onChange={(e) => setCvarAlpha(parseFloat(e.target.value))}
                            >
                              <option value={0.90}>90% (worst 10%)</option>
                              <option value={0.95}>95% (worst 5%)</option>
                              <option value={0.99}>99% (worst 1%)</option>
                            </select>
                          </div>
                        )}
                        
                        <button 
                          className="generate-frontier-btn"
                          onClick={loadFrontierData}
                          disabled={frontierLoading}
                        >
                          {frontierLoading ? '⏳ Generating...' : '🚀 Generate Frontier'}
                        </button>
                      </div>
                    </div>

                    {/* Educational Section */}
                    <div className="section">
                      <div className="info-card expandable">
                        <details>
                          <summary><strong>📚 Understanding the Efficient Frontier</strong></summary>
                          <div className="details-content">
                            <h5>What is the Efficient Frontier?</h5>
                            <p>
                              The Efficient Frontier is a cornerstone concept of <strong>Modern Portfolio Theory (MPT)</strong>, 
                              introduced by Harry Markowitz in 1952. It answers a fundamental question:
                            </p>
                            <p><em>"Given a set of assets, what is the best possible combination of risk and return I can achieve?"</em></p>
                            
                            <h5>Key Concepts</h5>
                            <ul>
                              <li><strong>Risk-Return Trade-off:</strong> Higher returns typically come with higher risk. The Efficient Frontier shows the optimal trade-off.</li>
                              <li><strong>Diversification Benefit:</strong> By combining assets that don't move perfectly together, you can achieve lower portfolio volatility than holding any single asset.</li>
                              <li><strong>Efficient vs Dominated:</strong> Portfolios on the frontier are efficient—no other portfolio offers higher return for the same risk.</li>
                            </ul>
                            
                            <h5>Mean-Variance vs Mean-CVaR</h5>
                            <p><strong>Mean-Variance (Markowitz, 1952):</strong></p>
                            <ul>
                              <li>Uses standard deviation (volatility) as risk measure</li>
                              <li>Assumes normally distributed returns</li>
                              <li>Penalizes upside and downside volatility equally</li>
                            </ul>
                            <p><strong>Mean-CVaR (Rockafellar & Uryasev, 2000):</strong></p>
                            <ul>
                              <li>Uses Conditional Value-at-Risk (average loss in worst scenarios)</li>
                              <li>Better captures tail risk and "black swan" events</li>
                              <li>More conservative in extreme scenarios</li>
                            </ul>
                            
                            <h5>How to Read the Chart</h5>
                            <ul>
                              <li><strong>Cloud of dots:</strong> Random portfolio combinations (feasible region)</li>
                              <li><strong>Upper-left boundary:</strong> The Efficient Frontier itself</li>
                              <li><strong>Colored markers:</strong> Your optimized portfolio strategies</li>
                              <li><strong>X-axis:</strong> Risk measure (Volatility or CVaR)</li>
                              <li><strong>Y-axis:</strong> Expected annualized return</li>
                            </ul>
                          </div>
                        </details>
                      </div>
                    </div>

                    {/* Frontier Chart */}
                    {frontierData ? (
                      <>
                        <div className="section">
                          <h3>
                            {frontierType === 'mean_variance' 
                              ? '📊 Mean-Variance Efficient Frontier' 
                              : `📉 Mean-CVaR Efficient Frontier (${(cvarAlpha * 100).toFixed(0)}%)`}
                          </h3>
                          <div className="chart-container frontier-chart">
                            <Plot
                              data={[
                                getRandomPortfoliosData(),
                                ...getStrategyPointsData()
                              ].filter(Boolean)}
                              layout={{
                                ...plotlyLayout,
                                height: 600,
                                xaxis: {
                                  ...plotlyLayout.xaxis,
                                  title: {
                                    text: frontierType === 'mean_variance' 
                                      ? 'Annualized Volatility (%)' 
                                      : `Annualized CVaR${(cvarAlpha * 100).toFixed(0)} (%)`,
                                    font: { color: '#E2E8F0', size: 14 },
                                    standoff: 20
                                  },
                                  tickfont: { color: '#94a3b8', size: 11 }
                                },
                                yaxis: {
                                  ...plotlyLayout.yaxis,
                                  title: {
                                    text: 'Annualized Return (%)',
                                    font: { color: '#E2E8F0', size: 14 },
                                    standoff: 20
                                  },
                                  tickfont: { color: '#94a3b8', size: 11 }
                                },
                                legend: {
                                  orientation: 'v',
                                  yanchor: 'top',
                                  y: 0.99,
                                  xanchor: 'left',
                                  x: 1.02,
                                  bgcolor: 'rgba(26,26,36,0.95)',
                                  bordercolor: 'rgba(99,102,241,0.5)',
                                  borderwidth: 1,
                                  font: { size: 10, color: '#E2E8F0' }
                                },
                                hovermode: 'closest',
                                margin: { t: 30, b: 80, l: 80, r: 180 }
                              }}
                              config={{ responsive: true, displayModeBar: false }}
                              style={{ width: '100%' }}
                            />
                          </div>
                        </div>

                        {/* Quick Stats */}
                        <div className="section">
                          <div className="frontier-stats">
                            <div className="frontier-stat-card">
                              <span className="stat-icon">🎲</span>
                              <span className="stat-label">Simulated</span>
                              <span className="stat-value">{frontierData.n_portfolios.toLocaleString()}</span>
                            </div>
                            <div className="frontier-stat-card">
                              <span className="stat-icon">⭐</span>
                              <span className="stat-label">Best Random Sharpe</span>
                              <span className="stat-value">{frontierData.stats.best_random_sharpe.toFixed(3)}</span>
                            </div>
                            <div className="frontier-stat-card">
                              <span className="stat-icon">📉</span>
                              <span className="stat-label">Min Risk</span>
                              <span className="stat-value">{frontierData.stats.min_risk.toFixed(2)}%</span>
                            </div>
                            <div className="frontier-stat-card">
                              <span className="stat-icon">📈</span>
                              <span className="stat-label">Max Return</span>
                              <span className="stat-value">{frontierData.stats.max_return.toFixed(2)}%</span>
                            </div>
                          </div>
                        </div>

                        {/* Why Strategies Inside Cloud */}
                        <div className="section">
                          <div className="info-card expandable">
                            <details>
                              <summary><strong>📚 Why Are Some Strategies Inside the Cloud?</strong></summary>
                              <div className="details-content">
                                <p>You might wonder: <em>"If I optimized these portfolios, why aren't they all on the efficient frontier?"</em></p>
                                
                                <h5>The Efficient Frontier is "Optimal" Only in Hindsight</h5>
                                <p>
                                  The frontier you see is calculated using <strong>historical data</strong>. It shows the best you 
                                  could have done if you had known the future perfectly. But when these strategies 
                                  were "optimized," they only had access to past data—just like in real investing.
                                </p>
                                
                                <h5>The Problem: Estimation Error</h5>
                                <p>
                                  Markowitz optimization requires estimating expected returns and covariances. 
                                  These estimates are extremely noisy. Merton (1980) showed that estimating expected 
                                  returns with the same precision as volatility would require ~500 years of data.
                                </p>
                                
                                <h5>Simple Often Beats "Optimal"</h5>
                                <p>
                                  DeMiguel, Garlappi & Uppal (2009) compared 14 optimization strategies against 
                                  the naive 1/N (Equal Weight) portfolio. Result: No optimized strategy consistently 
                                  beat Equal Weight out-of-sample!
                                </p>
                                
                                <p>
                                  <strong>The bottom line:</strong> Strategies like Equal Weight, Risk Parity, and HRP 
                                  accept being "suboptimal" in-sample to be more robust when the future is uncertain.
                                </p>
                              </div>
                            </details>
                          </div>
                        </div>

                        {/* Frontier Explorer */}
                        <div className="section">
                          <h3>🔍 Explore the Frontier</h3>
                          <p className="section-description">
                            Use the slider below to explore portfolios along the efficient frontier. 
                            Select different risk levels and compare with your optimized strategies.
                          </p>
                          
                          <div className="frontier-explorer">
                            <div className="explorer-controls">
                              <div className="slider-container">
                                <label>Risk Level: {frontierPosition}%</label>
                                <input
                                  type="range"
                                  min="0"
                                  max="100"
                                  step="5"
                                  value={frontierPosition}
                                  onChange={(e) => setFrontierPosition(parseInt(e.target.value))}
                                  className="frontier-slider"
                                />
                                <div className="slider-labels">
                                  <span>0% Min Risk</span>
                                  <span>100% Max Risk</span>
                                </div>
                              </div>
                              
                              <div className="compare-selector">
                                <label>Compare with Strategy</label>
                                <select
                                  value={compareStrategy || ''}
                                  onChange={(e) => setCompareStrategy(e.target.value)}
                                >
                                  {frontierData.strategy_points.map(s => (
                                    <option key={s.key} value={s.key}>{s.name}</option>
                                  ))}
                                </select>
                              </div>
                            </div>
                            
                            {/* Selected Frontier Portfolio */}
                            {getSelectedFrontierPoint() && (
                              <div className="explorer-results">
                                <div className="explorer-card selected-portfolio">
                                  <h4>📊 Selected Frontier Portfolio</h4>
                                  <div 
                                    className="risk-profile-badge"
                                    style={{ backgroundColor: `${getRiskProfileLabel().color}22`, borderColor: getRiskProfileLabel().color }}
                                  >
                                    {getRiskProfileLabel().label}
                                  </div>
                                  
                                  <div className="explorer-metrics">
                                    <div className="explorer-metric">
                                      <span className="metric-label">Return</span>
                                      <span className="metric-value">{getSelectedFrontierPoint().return_pct.toFixed(2)}%</span>
                                    </div>
                                    <div className="explorer-metric">
                                      <span className="metric-label">Volatility</span>
                                      <span className="metric-value">{getSelectedFrontierPoint().volatility_pct.toFixed(2)}%</span>
                                    </div>
                                    <div className="explorer-metric">
                                      <span className="metric-label">Sharpe</span>
                                      <span className="metric-value">{getSelectedFrontierPoint().sharpe.toFixed(3)}</span>
                                    </div>
                                    <div className="explorer-metric">
                                      <span className="metric-label">Risk Level</span>
                                      <span className="metric-value">{frontierPosition}%</span>
                                    </div>
                                  </div>
                                  
                                  {/* Weights */}
                                  <h5>Asset Allocation</h5>
                                  <div className="frontier-weights">
                                    {getSelectedFrontierPoint().weights
                                      .map((w, i) => ({ symbol: frontierData.symbols[i], weight: w }))
                                      .filter(w => Math.abs(w.weight) > 0.01)
                                      .sort((a, b) => b.weight - a.weight)
                                      .map((w, i) => (
                                        <div key={w.symbol} className="frontier-weight-item">
                                          <span className="weight-symbol">{w.symbol}</span>
                                          <div className="weight-bar-container">
                                            <div 
                                              className="weight-bar-fill"
                                              style={{ 
                                                width: `${Math.abs(w.weight) * 100}%`,
                                                backgroundColor: w.weight >= 0 ? '#4ECDC4' : '#FF6B6B'
                                              }}
                                            />
                                          </div>
                                          <span className="weight-value">{(w.weight * 100).toFixed(1)}%</span>
                                        </div>
                                      ))}
                                  </div>
                                </div>
                                
                                {/* Comparison with Strategy */}
                                {getCompareStrategyData() && (
                                  <div className="explorer-card comparison-card">
                                    <h4>⚖️ vs {getCompareStrategyData().name}</h4>
                                    
                                    <div className="comparison-table">
                                      <div className="comparison-row header">
                                        <span>Metric</span>
                                        <span>Frontier ({frontierPosition}%)</span>
                                        <span>{getCompareStrategyData().name.split(' ')[0]}</span>
                                      </div>
                                      <div className="comparison-row">
                                        <span>Return</span>
                                        <span>{getSelectedFrontierPoint().return_pct.toFixed(2)}%</span>
                                        <span>{getCompareStrategyData().return_pct.toFixed(2)}%</span>
                                      </div>
                                      <div className="comparison-row">
                                        <span>Volatility</span>
                                        <span>{getSelectedFrontierPoint().volatility_pct.toFixed(2)}%</span>
                                        <span>{getCompareStrategyData().volatility_pct.toFixed(2)}%</span>
                                      </div>
                                      <div className="comparison-row">
                                        <span>Sharpe</span>
                                        <span>{getSelectedFrontierPoint().sharpe.toFixed(3)}</span>
                                        <span>{getCompareStrategyData().sharpe.toFixed(3)}</span>
                                      </div>
                                    </div>
                                    
                                    {/* Quick Insight */}
                                    <div className="comparison-insight">
                                      {getSelectedFrontierPoint().sharpe > getCompareStrategyData().sharpe ? (
                                        <p className="insight positive">
                                          ✅ <strong>Frontier portfolio dominates</strong> on a risk-adjusted basis 
                                          (Sharpe +{(getSelectedFrontierPoint().sharpe - getCompareStrategyData().sharpe).toFixed(3)})
                                        </p>
                                      ) : (
                                        <p className="insight neutral">
                                          ℹ️ <strong>{getCompareStrategyData().name.split(' ')[0]} remains superior</strong> on a risk-adjusted basis 
                                          (Sharpe {getCompareStrategyData().sharpe.toFixed(3)} vs {getSelectedFrontierPoint().sharpe.toFixed(3)})
                                        </p>
                                      )}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        </div>

                        {/* ================================================================ */}
                        {/* HISTORICAL PERFORMANCE COMPARISON - Sezione separata */}
                        {/* ================================================================ */}
                        {getSelectedFrontierPoint() && compareStrategy && calculateFrontierCumulative() && frontierData.strategy_cumulative && (
                          <div className="section historical-section">
                            <h3>📈 Historical Performance Comparison</h3>
                            <p className="section-description">
                              Compare the <strong>selected frontier portfolio</strong> (risk level {frontierPosition}%) 
                              with <strong>{getCompareStrategyData()?.name}</strong> over the analysis period.
                            </p>
                            
                            {/* Performance Chart - Full Width */}
                            <div className="chart-container">
                              <Plot
                                data={[
                                  // Frontier portfolio
                                  {
                                    x: calculateFrontierCumulative().dates,
                                    y: calculateFrontierCumulative().values,
                                    type: 'scatter',
                                    mode: 'lines',
                                    name: `Frontier Portfolio (${frontierPosition}% risk)`,
                                    line: { color: getRiskProfileLabel().color, width: 2.5 },
                                    hovertemplate: '<b>Frontier Portfolio</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                                  },
                                  // Comparison strategy
                                  {
                                    x: frontierData.dates,
                                    y: frontierData.strategy_cumulative[compareStrategy],
                                    type: 'scatter',
                                    mode: 'lines',
                                    name: getCompareStrategyData()?.name,
                                    line: { color: '#A855F7', width: 2.5, dash: 'dash' },
                                    hovertemplate: `<b>${getCompareStrategyData()?.name}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>`
                                  }
                                ]}
                                layout={{
                                  ...plotlyLayout,
                                  height: 450,
                                  xaxis: {
                                    ...plotlyLayout.xaxis,
                                    title: { text: 'Date', font: { color: '#E2E8F0', size: 14 }, standoff: 20 },
                                    tickformat: '%Y-%m'
                                  },
                                  yaxis: {
                                    ...plotlyLayout.yaxis,
                                    title: { text: 'Portfolio Value (Base 100)', font: { color: '#E2E8F0', size: 14 }, standoff: 20 }
                                  },
                                  legend: {
                                    orientation: 'h',
                                    yanchor: 'bottom',
                                    y: 1.02,
                                    xanchor: 'center',
                                    x: 0.5,
                                    font: { color: '#E2E8F0', size: 12 }
                                  },
                                  hovermode: 'x unified',
                                  shapes: [{
                                    type: 'line',
                                    x0: frontierData.dates[0],
                                    x1: frontierData.dates[frontierData.dates.length - 1],
                                    y0: 100,
                                    y1: 100,
                                    line: { color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dot' }
                                  }],
                                  annotations: [{
                                    x: frontierData.dates[frontierData.dates.length - 1],
                                    y: 100,
                                    xanchor: 'right',
                                    yanchor: 'bottom',
                                    text: 'Initial Investment',
                                    showarrow: false,
                                    font: { color: 'rgba(255,255,255,0.5)', size: 10 }
                                  }],
                                  margin: { t: 40, b: 80, l: 80, r: 40 }
                                }}
                                config={{ responsive: true, displayModeBar: false }}
                                style={{ width: '100%' }}
                              />
                            </div>
                            
                            {/* Performance Summary Table - Full Width */}
                            <h4>📊 Performance Summary</h4>
                            <div className="table-container">
                              <table className="data-table">
                                <thead>
                                  <tr>
                                    <th>Metric</th>
                                    <th>Frontier Portfolio ({frontierPosition}%)</th>
                                    <th>{getCompareStrategyData()?.name}</th>
                                    <th>Difference</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  <tr>
                                    <td>Cumulative Return</td>
                                    <td className={calculateFrontierCumulative().metrics.cumReturn >= 0 ? 'positive' : 'negative'}>
                                      {calculateFrontierCumulative().metrics.cumReturn >= 0 ? '+' : ''}{calculateFrontierCumulative().metrics.cumReturn.toFixed(2)}%
                                    </td>
                                    <td className={
                                      (frontierData.strategy_cumulative[compareStrategy][frontierData.strategy_cumulative[compareStrategy].length - 1] - 100) >= 0 
                                        ? 'positive' : 'negative'
                                    }>
                                      {(frontierData.strategy_cumulative[compareStrategy][frontierData.strategy_cumulative[compareStrategy].length - 1] - 100) >= 0 ? '+' : ''}
                                      {(frontierData.strategy_cumulative[compareStrategy][frontierData.strategy_cumulative[compareStrategy].length - 1] - 100).toFixed(2)}%
                                    </td>
                                    <td className={
                                      calculateFrontierCumulative().metrics.cumReturn > 
                                      (frontierData.strategy_cumulative[compareStrategy][frontierData.strategy_cumulative[compareStrategy].length - 1] - 100)
                                        ? 'positive' : 'negative'
                                    }>
                                      {(calculateFrontierCumulative().metrics.cumReturn - 
                                        (frontierData.strategy_cumulative[compareStrategy][frontierData.strategy_cumulative[compareStrategy].length - 1] - 100)).toFixed(2)}%
                                    </td>
                                  </tr>
                                  <tr>
                                    <td>Annualized Return</td>
                                    <td className={calculateFrontierCumulative().metrics.annReturn >= 0 ? 'positive' : 'negative'}>
                                      {calculateFrontierCumulative().metrics.annReturn >= 0 ? '+' : ''}{calculateFrontierCumulative().metrics.annReturn.toFixed(2)}%
                                    </td>
                                    <td className={getCompareStrategyData()?.return_pct >= 0 ? 'positive' : 'negative'}>
                                      {getCompareStrategyData()?.return_pct >= 0 ? '+' : ''}{getCompareStrategyData()?.return_pct.toFixed(2)}%
                                    </td>
                                    <td className={
                                      calculateFrontierCumulative().metrics.annReturn > getCompareStrategyData()?.return_pct
                                        ? 'positive' : 'negative'
                                    }>
                                      {(calculateFrontierCumulative().metrics.annReturn - getCompareStrategyData()?.return_pct).toFixed(2)}%
                                    </td>
                                  </tr>
                                  <tr>
                                    <td>Annualized Volatility</td>
                                    <td>{calculateFrontierCumulative().metrics.annVol.toFixed(2)}%</td>
                                    <td>{getCompareStrategyData()?.volatility_pct.toFixed(2)}%</td>
                                    <td className={
                                      calculateFrontierCumulative().metrics.annVol < getCompareStrategyData()?.volatility_pct
                                        ? 'positive' : 'negative'
                                    }>
                                      {(calculateFrontierCumulative().metrics.annVol - getCompareStrategyData()?.volatility_pct).toFixed(2)}%
                                    </td>
                                  </tr>
                                  <tr>
                                    <td>Sharpe Ratio</td>
                                    <td>{calculateFrontierCumulative().metrics.sharpe.toFixed(3)}</td>
                                    <td>{getCompareStrategyData()?.sharpe.toFixed(3)}</td>
                                    <td className={
                                      calculateFrontierCumulative().metrics.sharpe > getCompareStrategyData()?.sharpe
                                        ? 'positive' : 'negative'
                                    }>
                                      {(calculateFrontierCumulative().metrics.sharpe - getCompareStrategyData()?.sharpe).toFixed(3)}
                                    </td>
                                  </tr>
                                  <tr>
                                    <td>Max Drawdown</td>
                                    <td className="negative">{calculateFrontierCumulative().metrics.maxDD.toFixed(2)}%</td>
                                    <td className="negative">
                                      {result.portfolios[compareStrategy] 
                                        ? (result.portfolios[compareStrategy].max_drawdown * 100).toFixed(2) 
                                        : 'N/A'}%
                                    </td>
                                    <td className={
                                      calculateFrontierCumulative().metrics.maxDD > 
                                      (result.portfolios[compareStrategy]?.max_drawdown * 100 || 0)
                                        ? 'positive' : 'negative'
                                    }>
                                      {(calculateFrontierCumulative().metrics.maxDD - 
                                        (result.portfolios[compareStrategy]?.max_drawdown * 100 || 0)).toFixed(2)}%
                                    </td>
                                  </tr>
                                </tbody>
                              </table>
                            </div>
                            <p className="table-caption">Side-by-Side Comparison</p>
                            
                            {/* Quick Insight */}
                            <div className="quick-insight-box">
                              {calculateFrontierCumulative().metrics.sharpe > getCompareStrategyData()?.sharpe ? (
                                <div className="insight-content positive">
                                  <span className="insight-icon">✅</span>
                                  <div className="insight-text">
                                    <strong>Frontier portfolio dominates on a risk-adjusted basis</strong>
                                    <span>Sharpe Ratio: {calculateFrontierCumulative().metrics.sharpe.toFixed(3)} vs {getCompareStrategyData()?.sharpe.toFixed(3)} 
                                    (+{(calculateFrontierCumulative().metrics.sharpe - getCompareStrategyData()?.sharpe).toFixed(3)})</span>
                                  </div>
                                </div>
                              ) : (
                                <div className="insight-content neutral">
                                  <span className="insight-icon">ℹ️</span>
                                  <div className="insight-text">
                                    <strong>{getCompareStrategyData()?.name} remains superior on a risk-adjusted basis</strong>
                                    <span>Sharpe Ratio: {getCompareStrategyData()?.sharpe.toFixed(3)} vs {calculateFrontierCumulative().metrics.sharpe.toFixed(3)} 
                                    ({(calculateFrontierCumulative().metrics.sharpe - getCompareStrategyData()?.sharpe).toFixed(3)})</span>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="empty-state">
                        <p>👆 Configure settings above and click <strong>"Generate Frontier"</strong> to visualize the efficient frontier</p>
                      </div>
                    )}
                  </div>
                )}

                {/* ==================== TAB: EXPORT ==================== */}
                {activeTab === 'export' && (
                  <div className="tab-panel">
                    <h2>📥 Export Data</h2>
                    
                    <div className="export-options">
                      <div className="export-card">
                        <h3>📊 JSON Export</h3>
                        <p>Download complete analysis results in JSON format.</p>
                        <button 
                          className="export-button"
                          onClick={() => {
                            const dataStr = JSON.stringify({ portfolios: result, overview: overviewData }, null, 2)
                            const blob = new Blob([dataStr], { type: 'application/json' })
                            const url = URL.createObjectURL(blob)
                            const a = document.createElement('a')
                            a.href = url
                            a.download = `portfolio_analysis_${new Date().toISOString().split('T')[0]}.json`
                            a.click()
                          }}
                        >
                          ⬇️ Download JSON
                        </button>
                      </div>
                      
                      <div className="export-card">
                        <h3>📋 Summary</h3>
                        <div className="export-summary">
                          <p><strong>Assets:</strong> {result.symbols.join(', ')}</p>
                          <p><strong>Period:</strong> {result.period.start} → {result.period.end}</p>
                          <p><strong>Trading Days:</strong> {result.trading_days}</p>
                          <p><strong>Strategies:</strong> {Object.keys(result.portfolios).length}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

              </div>
            </>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="footer">
        <p><strong>Portfolio Analyzer Pro</strong> • 7 Strategies • Advanced Optimization</p>
        <p className="disclaimer">⚠️ Educational purposes only. Not financial advice.</p>
      </footer>
    </div>
  )
}

export default App