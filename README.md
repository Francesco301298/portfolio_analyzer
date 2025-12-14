# 📊 Portfolio Analyzer Pro

Advanced Multi-Asset Portfolio Optimization System built with Streamlit.

## 🚀 Features

- **7 Portfolio Optimization Strategies**:
  - Equally Weighted
  - Minimum Volatility
  - Maximum Return
  - Maximum Sharpe Ratio
  - Risk Parity
  - Markowitz Mean-Variance
  - Hierarchical Risk Parity (HRP)

- **Advanced Analytics**:
  - Rolling performance analysis
  - Deep-dive Statistics
  - Correlation matrix
  - Efficient frontier visualization
  - Benchmark comparison
  - VaR and CVaR metrics

- **Export Options**:
  - Excel reports
  - JSON data

## 📦 Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/portfolio-analyzer.git
cd portfolio-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. **Create a GitHub Repository**:
   - Go to [github.com](https://github.com) and create a new repository
   - Name it `portfolio-analyzer` (or any name you prefer)

2. **Upload Files**:
   Upload these files to your repository:
   - `app.py`
   - `portfolio_analyzer.py`
   - `requirements.txt`
   - `.streamlit/config.toml`

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

4. **Your app will be live at**:
   `https://YOUR_APP_NAME.streamlit.app`

## 📁 Project Structure

```
portfolio-analyzer/
├── app.py                 # Main Streamlit application
├── portfolio_analyzer.py  # Portfolio optimization engine
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit theme configuration
```

## 🛠️ Technologies

- **Python 3.9+**
- **Streamlit** - Web framework
- **Plotly** - Interactive charts
- **yfinance** - Market data
- **NumPy/Pandas** - Data processing
- **SciPy** - Optimization algorithms

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute financial advice. Always consult with a qualified financial advisor before making investment decisions.

## 📄 License

MIT License - feel free to use and modify.

---

Made with ❤️ using Streamlit
