import numpy as np
from scipy.optimize import minimize
from core.metrics import calculate_portfolio_metrics

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
