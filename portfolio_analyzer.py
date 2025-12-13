"""
Portfolio Analyzer - Complete Advanced Version
Optimized for Streamlit Cloud Integration

Includes all portfolio optimization strategies:
- Equally Weighted
- Minimum Volatility
- Maximum Return
- Maximum Sharpe
- Risk Parity
- Markowitz Mean-Variance
- Hierarchical Risk Parity (HRP)
- Black-Litterman
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

# Fix encoding for Windows
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
            raise ValueError(
                f"Only {len(available_tickers)} tickers available!\n"
                f"Requested: {', '.join(self.symbols)}\n"
                f"Available: {', '.join(data.columns)}\n"
                f"Missing: {', '.join(missing_tickers)}"
            )
        
        self.data = data[available_tickers]
        self.valid_tickers = available_tickers
        self.invalid_tickers = missing_tickers
        self.symbols = available_tickers
        
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        self.data = self.data[(self.data.index >= start) & (self.data.index <= end)]
        self.data = self.data.dropna(how='all').ffill().bfill()
        
        return True

    def download_data(self):
        """Download data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Use: pip install yfinance")
        
        self.data = yf.download(self.symbols, start=self.start_date, end=self.end_date)['Close']

        if len(self.symbols) == 1:
            self.data = self.data.to_frame(name=self.symbols[0])

        self.data = self.data.dropna(how='all')
        self.data = self.data.loc[self.data.first_valid_index():self.data.last_valid_index()]
        self.data = self.data.ffill()
        self.data = self.data.reindex(columns=self.symbols)
        
        self.valid_tickers = self.symbols
        self.invalid_tickers = []

    def calculate_returns(self):
        """Calculate daily returns for all assets."""
        if self.data is None:
            raise ValueError("Load data first!")
        
        self.returns = self.data.pct_change().dropna()
        self.returns = self.returns.reindex(columns=self.symbols)
        return True

    def _calculate_portfolio_performance(self, weights, returns_series=None):
        """Calculate portfolio performance metrics."""
        if returns_series is None:
            returns_series = self.returns

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        if list(returns_series.columns) != self.symbols:
            returns_series = returns_series.reindex(columns=self.symbols)

        portfolio_returns = pd.Series(0.0, index=returns_series.index)
        for i, symbol in enumerate(self.symbols):
            portfolio_returns += weights[i] * returns_series[symbol]

        cumulative_return = (1 + portfolio_returns).prod() - 1
        n_years = len(portfolio_returns) / 252
        
        if n_years > 0:
            annualized_return = (1 + cumulative_return) ** (1/n_years) - 1
        else:
            annualized_return = 0.0
            
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        sharpe_ratio = 0.0
        if annualized_volatility > 1e-10:
            sharpe_ratio = float((annualized_return - self.risk_free_rate) / annualized_volatility)

        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = 0.0

        cumulative_value = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_value.expanding().max()
        drawdown = (cumulative_value - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        var_5 = np.percentile(portfolio_returns, 5)
        cvar_5 = portfolio_returns[portfolio_returns <= var_5].mean() if len(portfolio_returns[portfolio_returns <= var_5]) > 0 else var_5

        return {
            'returns': portfolio_returns,
            'cumulative_return': float(cumulative_return),
            'annualized_return': float(annualized_return),
            'annualized_volatility': float(annualized_volatility),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_drawdown': max_drawdown,
            'var_5': float(var_5),
            'cvar_5': float(cvar_5),
            'weights': weights
        }

    def equally_weighted_portfolio(self):
        """Create equally weighted (1/N) portfolio."""
        n_assets = len(self.symbols)
        weights = np.array([1/n_assets] * n_assets)
        performance = self._calculate_portfolio_performance(weights)
        performance['name'] = 'Equally Weighted'
        self.portfolios['equally_weighted'] = performance
        return performance

    def minimum_volatility_portfolio(self):
        """Create minimum volatility portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8

        def portfolio_volatility(weights):
            return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

        n_assets = len(self.symbols)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)

        result = minimize(portfolio_volatility, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})

        weights = result.x
        performance = self._calculate_portfolio_performance(weights)
        performance['name'] = 'Minimum Volatility'
        self.portfolios['min_volatility'] = performance
        return performance

    def maximum_return_portfolio(self):
        """Create maximum expected return portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        expected_returns = returns_aligned.mean() * 252

        def negative_return(weights):
            weights = np.array(weights)
            return -float(np.dot(weights, expected_returns.values))

        n_assets = len(self.symbols)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)

        result = minimize(negative_return, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'ftol': 1e-9, 'maxiter': 1000})

        weights = result.x
        performance = self._calculate_portfolio_performance(weights)
        performance['name'] = 'Maximum Return'
        self.portfolios['max_return'] = performance
        return performance

    def maximum_sharpe_portfolio(self):
        """Create maximum Sharpe ratio portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        expected_returns = returns_aligned.mean() * 252
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8

        def negative_sharpe(weights):
            weights = np.array(weights)
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_vol < 1e-10:
                return 1e10
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol

        n_assets = len(self.symbols)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)

        result = minimize(negative_sharpe, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})

        weights = result.x
        performance = self._calculate_portfolio_performance(weights)
        performance['name'] = 'Maximum Sharpe'
        self.portfolios['max_sharpe'] = performance
        return performance

    def risk_parity_portfolio(self):
        """Create Risk Parity portfolio (equal risk contribution)."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8

        def risk_budget_objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if portfolio_vol < 1e-10:
                return 1e10
            
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / len(weights)
            
            return np.sum((contrib - target_contrib) ** 2)

        n_assets = len(self.symbols)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.001, 0.999) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)

        result = minimize(risk_budget_objective, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})

        weights = result.x
        performance = self._calculate_portfolio_performance(weights)
        performance['name'] = 'Risk Parity'
        self.portfolios['risk_parity'] = performance
        return performance

    def markowitz_portfolio(self, target_return=None):
        """Create Markowitz mean-variance optimized portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        expected_returns = returns_aligned.mean() * 252
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8

        if target_return is None:
            target_return = float(expected_returns.mean())

        def portfolio_variance(weights):
            weights = np.array(weights)
            return float(np.dot(weights.T, np.dot(cov_matrix, weights)))

        n_assets = len(self.symbols)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)

        result = minimize(portfolio_variance, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})

        weights = result.x
        performance = self._calculate_portfolio_performance(weights)
        performance['name'] = f'Markowitz (target: {target_return:.1%})'
        self.portfolios['markowitz'] = performance
        return performance

    def hierarchical_risk_parity_portfolio(self):
        """Create Hierarchical Risk Parity (HRP) portfolio."""
        returns_aligned = self.returns.reindex(columns=self.symbols)
        
        corr_matrix = returns_aligned.corr()
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        distance_array = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(distance_array, method='single')
        
        sorted_idx = self._get_quasi_diag(linkage_matrix)
        
        cov_matrix = returns_aligned.cov() * 252
        weights = self._get_rec_bipart(cov_matrix, sorted_idx)
        
        weights_array = np.zeros(len(self.symbols))
        for i, symbol in enumerate(self.symbols):
            weights_array[i] = weights[symbol]

        performance = self._calculate_portfolio_performance(weights_array)
        performance['name'] = 'Hierarchical Risk Parity'
        self.portfolios['hrp'] = performance
        return performance

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

    def black_litterman_portfolio(self, views_returns=None, views_uncertainty=None):
        """Create Black-Litterman portfolio."""
        delta = 2.5
        tau = 0.025

        returns_aligned = self.returns.reindex(columns=self.symbols)
        cov_matrix = returns_aligned.cov() * 252
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues <= 0):
            cov_matrix = cov_matrix + np.eye(len(self.symbols)) * 1e-8

        market_weights = np.array([1/len(self.symbols)] * len(self.symbols))
        pi = delta * np.dot(cov_matrix, market_weights)

        if views_returns is None:
            mu_bl = pi
        else:
            mu_bl = pi

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        def portfolio_return(weights):
            return np.dot(weights, mu_bl)

        target_return = float(mu_bl.mean())

        n_assets = len(self.symbols)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = market_weights

        result = minimize(portfolio_variance, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})

        weights = result.x
        performance = self._calculate_portfolio_performance(weights)
        performance['name'] = 'Black-Litterman'
        self.portfolios['black_litterman'] = performance
        return performance

    def build_all_portfolios(self):
        """Build all portfolio types."""
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
            ('black_litterman', self.black_litterman_portfolio),
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
                result = minimize(portfolio_variance, initial_guess,
                                 method='SLSQP', bounds=bounds, constraints=constraints,
                                 options={'maxiter': 500})
                
                if result.success:
                    vol = np.sqrt(result.fun)
                    sharpe = (target - self.risk_free_rate) / vol if vol > 0 else 0
                    frontier_data.append({
                        'Return': target,
                        'Volatility': vol,
                        'Sharpe': sharpe
                    })
            except Exception:
                pass
        
        return pd.DataFrame(frontier_data)

    def export_to_excel(self, filename=None):
        """Export all data to Excel file with multiple sheets."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'portfolio_analysis_{timestamp}.xlsx'

        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl not installed. Use: pip install openpyxl")

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.data.to_excel(writer, sheet_name='Prices')
            self.returns.to_excel(writer, sheet_name='Returns')
            self.compare_portfolios().to_excel(writer, sheet_name='Comparison', index=False)
            
            weights_df = pd.DataFrame(index=self.symbols)
            for name, portfolio in self.portfolios.items():
                weights_df[portfolio['name']] = portfolio['weights']
            weights_df.to_excel(writer, sheet_name='Weights')
            
            portfolio_values_df = pd.DataFrame(index=self.data.index)
            for name, portfolio in self.portfolios.items():
                returns = portfolio['returns']
                common_dates = portfolio_values_df.index.intersection(returns.index)
                cumulative = (1 + returns.loc[common_dates]).cumprod() * 100
                portfolio_values_df.loc[common_dates, portfolio['name']] = cumulative.values
            portfolio_values_df = portfolio_values_df.ffill().bfill().fillna(100)
            portfolio_values_df.to_excel(writer, sheet_name='Portfolio_Values')
            
            stats_data = []
            for name, portfolio in self.portfolios.items():
                returns = portfolio['returns']
                stats_data.append({
                    'Portfolio': portfolio['name'],
                    'Daily_Mean': returns.mean(),
                    'Daily_Std': returns.std(),
                    'Min': returns.min(),
                    'Max': returns.max(),
                    'Skewness': stats.skew(returns),
                    'Kurtosis': stats.kurtosis(returns),
                    'VaR_5%': np.percentile(returns, 5),
                    'VaR_1%': np.percentile(returns, 1),
                })
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)

        return filename

    def export_selected_portfolios_to_excel(self, portfolio_names, filename=None, 
                                           include_rolling=False, window_years=3):
        """Export selected portfolios to Excel with optional rolling analysis."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'selected_portfolios_{timestamp}.xlsx'

        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl not installed. Use: pip install openpyxl")

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            portfolio_values_df = pd.DataFrame(index=self.data.index)
            for p_name in portfolio_names:
                if p_name in self.portfolios:
                    portfolio = self.portfolios[p_name]
                    returns = portfolio['returns']
                    common_dates = portfolio_values_df.index.intersection(returns.index)
                    cumulative = (1 + returns.loc[common_dates]).cumprod() * 100
                    portfolio_values_df.loc[common_dates, portfolio['name']] = cumulative.values
            portfolio_values_df = portfolio_values_df.ffill().bfill().fillna(100)
            portfolio_values_df.to_excel(writer, sheet_name='Values')
            
            weights_df = pd.DataFrame(index=self.symbols)
            for p_name in portfolio_names:
                if p_name in self.portfolios:
                    portfolio = self.portfolios[p_name]
                    weights_df[portfolio['name']] = portfolio['weights']
            weights_df.to_excel(writer, sheet_name='Weights')
            
            comparison_data = []
            for p_name in portfolio_names:
                if p_name in self.portfolios:
                    portfolio = self.portfolios[p_name]
                    comparison_data.append({
                        'Portfolio': portfolio['name'],
                        'Cum_Return': portfolio['cumulative_return'],
                        'Ann_Return': portfolio['annualized_return'],
                        'Ann_Volatility': portfolio['annualized_volatility'],
                        'Sharpe_Ratio': portfolio['sharpe_ratio'],
                        'Sortino_Ratio': portfolio['sortino_ratio'],
                        'Max_Drawdown': portfolio['max_drawdown'],
                        'Calmar_Ratio': portfolio['calmar_ratio']
                    })
            pd.DataFrame(comparison_data).to_excel(writer, sheet_name='Metrics', index=False)

        return filename
