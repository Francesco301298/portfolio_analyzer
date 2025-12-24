import { useState } from 'react'
import axios from 'axios'
import './App.css'

// URL del tuo backend FastAPI
const API_URL = 'http://127.0.0.1:8000'

function App() {
  // State per i dati
  const [symbols, setSymbols] = useState('AAPL, MSFT, GOOGL')
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2024-01-01')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // Funzione per chiamare l'API
  const analyzePortfolio = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Prepara i simboli come array
      const symbolList = symbols.split(',').map(s => s.trim().toUpperCase())

      // Chiama la tua API FastAPI!
      const response = await axios.post(`${API_URL}/api/portfolio/analyze`, {
        symbols: symbolList,
        start_date: startDate,
        end_date: endDate,
        risk_free_rate: 0.02
      })

      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>🎯 Portfolio Analyzer Pro</h1>
      <p>Analisi quantitativa di portafoglio</p>

      {/* Form di input */}
      <div className="form-section">
        <div className="input-group">
          <label>Simboli (separati da virgola):</label>
          <input
            type="text"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            placeholder="AAPL, MSFT, GOOGL"
          />
        </div>

        <div className="input-group">
          <label>Data Inizio:</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
        </div>

        <div className="input-group">
          <label>Data Fine:</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div>

        <button onClick={analyzePortfolio} disabled={loading}>
          {loading ? '⏳ Analisi in corso...' : '🚀 Analizza Portfolio'}
        </button>
      </div>

      {/* Messaggio di errore */}
      {error && (
        <div className="error">
          ❌ Errore: {error}
        </div>
      )}

      {/* Risultati */}
      {result && (
        <div className="results">
          <h2>📊 Risultati Analisi</h2>
          
          <div className="info-box">
            <p><strong>Simboli:</strong> {result.symbols.join(', ')}</p>
            <p><strong>Periodo:</strong> {result.period.start} → {result.period.end}</p>
            <p><strong>Giorni di trading:</strong> {result.trading_days}</p>
          </div>

          <h3>📈 Strategie di Portafoglio</h3>
          
          <div className="portfolios-grid">
            {Object.entries(result.portfolios).map(([key, portfolio]) => (
              <div key={key} className="portfolio-card">
                <h4>{portfolio.name}</h4>
                <div className="metrics">
                  <div className="metric">
                    <span className="label">Rendimento Ann.</span>
                    <span className="value">{(portfolio.annualized_return * 100).toFixed(2)}%</span>
                  </div>
                  <div className="metric">
                    <span className="label">Volatilità Ann.</span>
                    <span className="value">{(portfolio.annualized_volatility * 100).toFixed(2)}%</span>
                  </div>
                  <div className="metric">
                    <span className="label">Sharpe Ratio</span>
                    <span className="value">{portfolio.sharpe_ratio.toFixed(3)}</span>
                  </div>
                  <div className="metric">
                    <span className="label">Max Drawdown</span>
                    <span className="value">{(portfolio.max_drawdown * 100).toFixed(2)}%</span>
                  </div>
                </div>
                <div className="weights">
                  <strong>Pesi:</strong>
                  {Object.entries(portfolio.weights).map(([symbol, weight]) => (
                    <span key={symbol} className="weight-badge">
                      {symbol}: {(weight * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default App