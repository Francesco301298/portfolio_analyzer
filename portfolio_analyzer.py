"""
Portfolio Analyzer - Complete Advanced Version v7
Optimized for Streamlit Cloud Integration

Includes all portfolio optimization strategies:
- Equally Weighted
- Minimum Volatility
- Maximum Return
- Maximum Sharpe
- Risk Parity
- Markowitz Mean-Variance
- Hierarchical Risk Parity (HRP)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
import sys
import os

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class AdvancedPortfolioAnalyzer:
    """
    Advanced Portfolio Analyzer with multiple optimization strategies.
    Supports loading data from pre-downloaded files or via yfinance.
    """
    
    def __init__(self, symbols, start_date='2005-01-01', end_date=None, risk_free_rate=0.02):
        if not symbols or len(symbols) < 2:
            raise ValueError("At least 2 tickers are required!")
        
        self.symbols = list(dict.fromkeys([s.strip().upper() for s in symbols]))
        self.start_date = start_date
        self.end_date = end_date if end_date else (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        self.risk_free_rate = risk_free_rate

        self.data = None
        self.returns = None
        self.portfolios = {}
        self.valid_tickers = []
        self.invalid_tickers = []

    def load_data_from_colab(self, data_path):
        """Load data from pre-downloaded files (Colab export)."""
        if not os.path.exists(data_path):
            raise ValueError(f"Path not found: {data_path}")
        
        if os.path.isfile(data_path):
            if data_path.endswith('.pkl'):
                data = pd.read_pickle(data_path)
            elif data_path.endswith('.csv'):
                data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            elif data_path.endswith('.xlsx'):
                data = pd.read_excel(data_path, index_col=0)
            else:
                raise ValueError(f"Unsupported format: {data_path}")
        
        elif os.path.isdir(data_path):
            possible_files = [
                os.path.join(data_path, 'prices_data.pkl'),
                os.path.join(data_path, 'prices_data.csv'),
                os.path.join(data_path, 'prices_data.xlsx'),
            ]
            
            file_to_load = None
            for f in possible_files:
                if os.path.exists(f):
                    file_to_load = f
                    break
            
            if file_to_load is None:
                raise ValueError(f"No data file found in: {data_path}")
            
            if file_to_load.endswith('.pkl'):
                data = pd.read_pickle(file_to_load)
            elif file_to_load.endswith('.csv'):
                data = pd.read_csv(file_to_load, index_col=0, parse_dates=True)
            else:
                data = pd.read_excel(file_to_load, index_col=0)
        else:
            raise ValueError(f"Invalid path: {data_path}")
        
        if data.empty:
            raise ValueError("Empty DataFrame!")
        
        available_tickers = [t for t in self.symbols if t in data.columns]
        missing_tickers = [t for t in self.symbols if t not in data.columns]
        
        if len(available_tickers) < 2:
            raise ValueError(f"Less than 2 valid tickers found! Missing: {missing_tickers}")
        
        self.data = data[available_tickers]
        self.symbols = available_tickers
        self.valid_tickers = available_tickers
        self.invalid_tickers = missing_tickers
        
        if self.start_date:
            self.data = self.data[self.data.index >= self.start_date]
        if self.end_date:
            self.data = self.data[self.data.index <= self.end_date]
        
        self.data = self.data.dropna()
        
        if len(self.data) < 30:
            raise ValueError("Not enough data for analysis (minimum 30 days required)")
        
        return self.data

    def download_data(self):
        """Download data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required! Install with: pip install yfinance")
        
        all_data = {}
        valid_tickers = []
        invalid_tickers = []

        for ticker in self.symbols:
            try:
                stock_data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                
                if stock_data.empty or len(stock_data) < 30:
                    invalid_tickers.append(ticker)
                    continue
                
                if 'Close' in stock_data.columns:
                    close_prices = stock_data['Close']
                elif 'Adj Close' in stock_data.columns:
                    close_prices = stock_data['Adj Close']
                else:
                    close_col = [col for col in stock_data.columns if 'Close' in str(col)]
                    if close_col:
                        close_prices = stock_data[close_col[0]]
                    else:
                        invalid_tickers.append(ticker)
                        continue
                
                if hasattr(close_prices, 'values'):
                    close_prices = pd.Series(close_prices.values.flatten(), index=stock_data.index)
                
                all_data[ticker] = close_prices
                valid_tickers.append(ticker)
                
            except Exception as e:
                invalid_tickers.append(ticker)
                continue

        if len(valid_tickers) < 2:
            raise ValueError(f"Not enough valid data! Invalid tickers: {invalid_tickers}")

        self.data = pd.DataFrame(all_data)
        self.data = self.data.dropna()
        self.symbols = valid_tickers
        self.valid_tickers = valid_tickers
        self.invalid_tickers = invalid_tickers

        return self.data

    def calculate_returns(self):
        """Calculate daily returns."""
        if self.data is None:
            raise ValueError("Download data first!")
        
        self.returns = self.data.pct_change().dropna()
        return self.returns

    def _calculate_portfolio_performance(self, weights, name):
        """Calculate portfolio performance metrics."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        
        portfolio_returns = returns_aligned.dot(weights)
        
        annualized_return = portfolio_returns.mean() * 252
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
        
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_volatility
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        var_5 = np.percentile(portfolio_returns, 5)
        cvar_5 = portfolio_returns[portfolio_returns <= var_5].mean() if len(portfolio_returns[portfolio_returns <= var_5]) > 0 else var_5
        
        cumulative_return = cumulative.iloc[-1] - 1
        
        performance = {
            'name': name,
            'weights': np.array(weights),
            'returns': portfolio_returns,
            'cumulative': cumulative,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'cumulative_return': cumulative_return
        }
        
        return performance

    def equally_weighted_portfolio(self):
        """Create equally weighted portfolio."""
        n = len(self.symbols)
        weights = np.array([1/n] * n)
        self.portfolios['equally_weighted'] = self._calculate_portfolio_performance(weights, "Equally Weighted")
        return self.portfolios['equally_weighted']

    def minimum_volatility_portfolio(self):
        """Create minimum volatility portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        n_assets = len(self.symbols)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        self.portfolios['min_volatility'] = self._calculate_portfolio_performance(
            optimal_weights, "Minimum Volatility")
        return self.portfolios['min_volatility']

    def maximum_return_portfolio(self):
        """Create maximum return portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        expected_returns = returns_aligned.mean() * 252
        
        def negative_return(weights):
            return -np.dot(weights, expected_returns)
        
        n_assets = len(self.symbols)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(negative_return, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        self.portfolios['max_return'] = self._calculate_portfolio_performance(
            optimal_weights, "Maximum Return")
        return self.portfolios['max_return']

    def maximum_sharpe_portfolio(self):
        """Create maximum Sharpe ratio portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        expected_returns = returns_aligned.mean() * 252
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8
        
        def negative_sharpe(weights):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        n_assets = len(self.symbols)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        self.portfolios['max_sharpe'] = self._calculate_portfolio_performance(
            optimal_weights, "Maximum Sharpe")
        return self.portfolios['max_sharpe']

    def risk_parity_portfolio(self):
        """Create risk parity portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8
        
        def risk_contribution_error(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol == 0:
                return 1e10
            
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            target_risk = port_vol / len(self.symbols)
            
            return np.sum((risk_contrib - target_risk) ** 2)
        
        n_assets = len(self.symbols)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(risk_contribution_error, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        self.portfolios['risk_parity'] = self._calculate_portfolio_performance(
            optimal_weights, "Risk Parity")
        return self.portfolios['risk_parity']

    def markowitz_portfolio(self, target_return=None):
        """Create Markowitz mean-variance optimized portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        expected_returns = returns_aligned.mean() * 252
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8
        
        if target_return is None:
            target_return = expected_returns.mean()
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        n_assets = len(self.symbols)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        self.portfolios['markowitz'] = self._calculate_portfolio_performance(
            optimal_weights, "Markowitz Mean-Variance")
        return self.portfolios['markowitz']

    def hierarchical_risk_parity_portfolio(self):
        """Create HRP portfolio using hierarchical clustering."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        cov_matrix = returns_aligned.cov()
        corr_matrix = returns_aligned.corr()
        
        if cov_matrix.isnull().any().any():
            cov_matrix = cov_matrix.fillna(0)
        if corr_matrix.isnull().any().any():
            corr_matrix = corr_matrix.fillna(0)
        
        np.fill_diagonal(corr_matrix.values, 1)
        
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        try:
            dist_condensed = squareform(distance_matrix.values, checks=False)
            link = linkage(dist_condensed, method='ward')
        except Exception:
            n = len(self.symbols)
            weights = np.array([1/n] * n)
            self.portfolios['hrp'] = self._calculate_portfolio_performance(weights, "HRP")
            return self.portfolios['hrp']
        
        sort_idx = self._get_quasi_diag(link)
        weights = self._get_rec_bipart(cov_matrix, sort_idx)
        
        weights = weights / weights.sum()
        
        self.portfolios['hrp'] = self._calculate_portfolio_performance(
            weights.values, "Hierarchical Risk Parity")
        return self.portfolios['hrp']

    def _get_quasi_diag(self, link):
        """Reorganize covariance matrix to quasi-diagonal form."""
        link = link.astype(int)
        sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_idx.max() >= num_items:
            sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
            df0 = sort_idx[sort_idx >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_idx[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_idx = pd.concat([sort_idx, df0])
            sort_idx = sort_idx.sort_index()
            sort_idx.index = range(sort_idx.shape[0])

        return sort_idx.tolist()

    def _get_rec_bipart(self, cov, sort_idx):
        """Calculate HRP weights using recursive bisection."""
        sort_idx_symbols = [self.symbols[i] for i in sort_idx]
        weights = pd.Series(1, index=sort_idx_symbols)
        clustered_alphas = [sort_idx_symbols]

        while len(clustered_alphas) > 0:
            new_clusters = []
            for cluster in clustered_alphas:
                N = len(cluster)
                if N > 1:
                    split = N // 2
                    new_clusters.append(cluster[:split])
                    new_clusters.append(cluster[split:])

            if not new_clusters:
                break

            clustered_alphas = new_clusters

            if len(clustered_alphas) < 2:
                break

            for i in range(0, len(clustered_alphas) - 1, 2):
                if i + 1 >= len(clustered_alphas):
                    break

                cluster0 = clustered_alphas[i]
                cluster1 = clustered_alphas[i+1]

                try:
                    cov0 = cov.loc[cluster0, cluster0]
                    cov1 = cov.loc[cluster1, cluster1]

                    ivp0 = 1. / np.diag(cov0)
                    ivp1 = 1. / np.diag(cov1)

                    alpha0 = ivp0 / ivp0.sum()
                    alpha1 = ivp1 / ivp1.sum()

                    var0 = np.dot(alpha0, np.dot(cov0, alpha0))
                    var1 = np.dot(alpha1, np.dot(cov1, alpha1))

                    if (var0 + var1) == 0:
                        alpha = 0.5
                    else:
                        alpha = 1 - var0 / (var0 + var1)

                    weights[cluster0] *= alpha
                    weights[cluster1] *= 1 - alpha

                except (KeyError, IndexError, ZeroDivisionError):
                    weights[cluster0] *= 0.5
                    weights[cluster1] *= 0.5

        return weights.reindex(self.symbols)

    def build_all_portfolios(self):
        """Build all portfolio types (7 strategies, no Black-Litterman)."""
        if self.returns is None:
            raise ValueError("Calculate returns first!")
        
        portfolio_builders = [
            ('equally_weighted', self.equally_weighted_portfolio),
            ('min_volatility', self.minimum_volatility_portfolio),
            ('max_return', self.maximum_return_portfolio),
            ('max_sharpe', self.maximum_sharpe_portfolio),
            ('risk_parity', self.risk_parity_portfolio),
            ('markowitz', self.markowitz_portfolio),
            ('hrp', self.hierarchical_risk_parity_portfolio),
        ]
        
        for name, builder in portfolio_builders:
            try:
                builder()
            except Exception as e:
                print(f"Warning: Could not build {name} portfolio: {e}")

    def compare_portfolios(self):
        """Create comparison table of all portfolios."""
        if not self.portfolios:
            raise ValueError("Build portfolios first!")

        comparison_data = []

        for name, portfolio in self.portfolios.items():
            comparison_data.append({
                'Portfolio': portfolio['name'],
                'Ann. Return': f"{portfolio['annualized_return']:.2%}",
                'Ann. Volatility': f"{portfolio['annualized_volatility']:.2%}",
                'Sharpe Ratio': f"{portfolio['sharpe_ratio']:.3f}",
                'Sortino Ratio': f"{portfolio['sortino_ratio']:.3f}",
                'Max Drawdown': f"{portfolio['max_drawdown']:.2%}",
                'Calmar Ratio': f"{portfolio['calmar_ratio']:.3f}",
                'VaR 5%': f"{portfolio['var_5']:.4f}",
                'Cum. Return': f"{portfolio['cumulative_return']:.2%}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['Sharpe_Numeric'] = [self.portfolios[name]['sharpe_ratio']
                                          for name in self.portfolios.keys()]
        comparison_df = comparison_df.sort_values('Sharpe_Numeric', ascending=False)
        comparison_df = comparison_df.drop('Sharpe_Numeric', axis=1)

        return comparison_df

    def get_efficient_frontier(self, n_points=50):
        """Calculate efficient frontier points."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        expected_returns = returns_aligned.mean() * 252
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8
        
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_data = []
        
        for target in target_returns:
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            n_assets = len(self.symbols)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, t=target: np.dot(x, expected_returns) - t}
            ]
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_guess = np.array([1/n_assets] * n_assets)
            
            try:
                result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints,
                                options={'ftol': 1e-9, 'maxiter': 1000})
                
                if result.success:
                    vol = np.sqrt(result.fun)
                    ret = target
                    sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
                    
                    frontier_data.append({
                        'Return': ret,
                        'Volatility': vol,
                        'Sharpe': sharpe
                    })
            except:
                continue
        
        return pd.DataFrame(frontier_data)

    def export_to_excel(self, filename=None):
        """Export analysis to Excel file."""
        if filename is None:
            filename = f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.data.to_excel(writer, sheet_name='Prices')
            self.returns.to_excel(writer, sheet_name='Returns')
            
            weights_data = {}
            metrics_data = []
            
            for name, portfolio in self.portfolios.items():
                weights_data[portfolio['name']] = portfolio['weights']
                metrics_data.append({
                    'Portfolio': portfolio['name'],
                    'Annualized Return': portfolio['annualized_return'],
                    'Annualized Volatility': portfolio['annualized_volatility'],
                    'Sharpe Ratio': portfolio['sharpe_ratio'],
                    'Sortino Ratio': portfolio['sortino_ratio'],
                    'Max Drawdown': portfolio['max_drawdown'],
                    'Calmar Ratio': portfolio['calmar_ratio'],
                    'VaR 5%': portfolio['var_5'],
                    'CVaR 5%': portfolio['cvar_5'],
                    'Cumulative Return': portfolio['cumulative_return']
                })
            
            weights_df = pd.DataFrame(weights_data, index=self.symbols)
            weights_df.to_excel(writer, sheet_name='Weights')
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            frontier_df = self.get_efficient_frontier()
            frontier_df.to_excel(writer, sheet_name='Efficient Frontier', index=False)
        
        return filename
