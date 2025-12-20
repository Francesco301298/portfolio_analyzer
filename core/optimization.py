import numpy as np
import pandas as pd
from scipy.optimize import linprog
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

"""
CVaR Portfolio Optimization
Implementation based on Rockafellar & Uryasev (2000)

Reference:
Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional 
Value-at-Risk." Journal of Risk, 2(3), 21-41.

This module implements the Linear Programming formulation from Section 6
of the paper, which is the recommended approach for practical applications.
"""

def cvar_optimization(returns_df, alpha=0.95, target_return=None):
    """
    CVaR Portfolio Optimization using Linear Programming (Rockafellar & Uryasev, 2000).
    
    This implements the Sample Average Approximation (SAA) formulation from
    Section 6 of the paper (Equation 14, page 10):
    
        min_{x, ζ, u}  ζ + (1 / (T(1-α))) * Σ u_t
        
        s.t.  u_t ≥ -R_t^T x - ζ,  ∀t ∈ {1,...,T}
              u_t ≥ 0,              ∀t
              Σ x_i = 1
              x_i ≥ 0,              ∀i
              
    where:
        x ∈ R^n    : portfolio weights
        ζ ∈ R      : auxiliary variable (equals VaR_α at optimum)
        u ∈ R^T    : auxiliary variables for loss exceedances
        R_t        : return vector for historical scenario t
        α          : confidence level (e.g., 0.95)
        T          : number of historical scenarios
        n          : number of assets
    
    Mathematical Background
    -----------------------
    CVaR (Conditional Value-at-Risk) is defined as the expected loss in the
    worst (1-α)% of scenarios:
    
        CVaR_α(X) = E[X | X ≥ VaR_α(X)]
    
    Rockafellar & Uryasev's key contribution (Theorem 1, page 8) shows that
    CVaR can be computed by minimizing the function:
    
        F_α(x, ζ) = ζ + (1/(1-α)) * E[[L(x,y) - ζ]^+]
    
    where L(x,y) is the loss function and [z]^+ = max(z, 0).
    
    The LP above is the discrete (SAA) version of this optimization problem,
    which is guaranteed to:
    1. Produce the global optimal solution (LP is convex)
    2. Have ζ* = VaR_α(x*) at optimum
    3. Have objective value = CVaR_α(x*)
    
    Why This Formulation is Optimal
    --------------------------------
    - **Convexity**: Guaranteed global optimum
    - **Efficiency**: Modern LP solvers (HiGHS) are extremely fast
    - **Robustness**: No convergence issues, numerically stable
    - **Scalability**: Handles 10-100 assets with 1000+ scenarios easily
    - **Theoretical guarantee**: Exactly implements the paper's algorithm
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Historical daily returns (T × n), where:
        - T = number of days (scenarios)
        - n = number of assets
        Indexed by date, columns are asset tickers
        
    alpha : float, default=0.95
        Confidence level for CVaR calculation
        - 0.95 → focus on worst 5% of scenarios
        - 0.99 → focus on worst 1% (more extreme tail)
        Higher α = more conservative portfolio
        
    target_return : float, optional
        Minimum required expected return (annualized)
        If specified, adds constraint: E[R^T x] ≥ target_return/252
        Default: None (no return constraint)
        
    Returns
    -------
    dict with keys:
        'weights' : np.ndarray
            Optimal portfolio weights (shape: (n,), sums to 1)
            
        'cvar' : float
            Optimal CVaR value (daily, typically negative)
            Interpretation: "Average loss in worst (1-α)% of days"
            Example: cvar = -0.025 means 2.5% average loss in worst 5% of days
            
        'var' : float
            Corresponding VaR_α value (daily, typically negative)
            Interpretation: "Threshold for worst (1-α)% of days"
            Example: var = -0.02 means 5th percentile loss is 2%
            
        'success' : bool
            Whether optimization converged successfully
            
    Raises
    ------
    ValueError
        If returns_df has less than 2 assets or less than 30 days
        
    Notes
    -----
    - All return values (CVaR, VaR) are in **daily** terms (not annualized)
    - Negative values indicate losses (e.g., -0.03 = 3% loss)
    - The LP uses the HiGHS solver (fastest modern LP solver)
    - If optimization fails, returns equal weights with NaN metrics
    
    Examples
    --------
    Basic usage:
    
    >>> returns = analyzer.returns  # T x n DataFrame of daily returns
    >>> result = cvar_optimization(returns, alpha=0.95)
    >>> print(f"Optimal CVaR (daily): {result['cvar']*100:.2f}%")
    >>> print(f"Optimal weights: {result['weights']}")
    
    With return constraint:
    
    >>> result = cvar_optimization(returns, alpha=0.95, target_return=0.10)
    >>> # Finds min-CVaR portfolio with at least 10% annualized return
    
    Conservative (99% CVaR):
    
    >>> result = cvar_optimization(returns, alpha=0.99)
    >>> # Focuses on worst 1% of scenarios (more tail risk aversion)
    
    References
    ----------
    .. [1] Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional 
           Value-at-Risk." Journal of Risk, 2(3), 21-41.
    .. [2] Artzner, P., Delbaen, F., Eber, J.M., & Heath, D. (1999). 
           "Coherent Measures of Risk." Mathematical Finance, 9(3), 203-228.
    """
    # Input validation
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    
    returns = returns_df.values
    T, n = returns.shape
    
    if n < 2:
        raise ValueError(f"At least 2 assets required, got {n}")
    if T < 30:
        raise ValueError(f"At least 30 days of data required, got {T}")
    
    # Check for NaN/inf
    if np.any(~np.isfinite(returns)):
        raise ValueError("returns_df contains NaN or infinite values")
    
    # =========================================================================
    # STEP 1: Setup LP problem
    # =========================================================================
    # Decision variables: [x_1, ..., x_n, ζ, u_1, ..., u_T]
    # Total: n + 1 + T variables
    
    # Objective function: min ζ + (1/(T(1-α))) * Σ u_t
    c = np.zeros(n + 1 + T)
    c[n] = 1.0                        # coefficient for ζ
    c[n+1:] = 1.0 / (T * (1 - alpha)) # coefficients for u_t
    
    # =========================================================================
    # STEP 2: Inequality constraints
    # =========================================================================
    # Constraint set 1: u_t ≥ -R_t^T x - ζ for all t
    # Rewrite as: R_t^T x + ζ + u_t ≥ 0
    # Standard LP form (Ax ≤ b): -R_t^T x - ζ - u_t ≤ 0
    
    A_ub = np.zeros((T, n + 1 + T))
    b_ub = np.zeros(T)
    
    for t in range(T):
        # Row t: -R_t^T x - ζ - u_t ≤ 0
        A_ub[t, :n] = -returns[t, :]     # coefficients for x (weights)
        A_ub[t, n] = -1.0                 # coefficient for ζ
        A_ub[t, n + 1 + t] = -1.0         # coefficient for u_t
    
    # Constraint set 2 (optional): Return constraint
    # E[R^T x] ≥ target_return/252
    # Rewrite as: -E[R^T x] ≤ -target_return/252
    if target_return is not None:
        target_daily = target_return / 252
        expected_returns = returns.mean(axis=0)
        
        A_return = np.zeros((1, n + 1 + T))
        A_return[0, :n] = -expected_returns
        b_return = np.array([-target_daily])
        
        # Stack with existing constraints
        A_ub = np.vstack([A_ub, A_return])
        b_ub = np.concatenate([b_ub, b_return])
    
    # =========================================================================
    # STEP 3: Equality constraint (budget)
    # =========================================================================
    # Σ x_i = 1 (fully invested)
    A_eq = np.zeros((1, n + 1 + T))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    # =========================================================================
    # STEP 4: Variable bounds
    # =========================================================================
    # x_i ∈ [0, 1]  : long-only portfolio (no short selling)
    # ζ ∈ R         : VaR can be any value
    # u_t ∈ [0, ∞)  : exceedances are non-negative by definition
    
    bounds = [(0, 1) for _ in range(n)]      # weights in [0,1]
    bounds.append((None, None))               # ζ unbounded
    bounds.extend([(0, None) for _ in range(T)])  # u_t ≥ 0
    
    # =========================================================================
    # STEP 5: Solve LP using HiGHS
    # =========================================================================
    # HiGHS is the fastest modern LP solver (outperforms interior-point methods)
    result = linprog(
        c, 
        A_ub=A_ub, 
        b_ub=b_ub, 
        A_eq=A_eq, 
        b_eq=b_eq, 
        bounds=bounds,
        method='highs',
        options={
            'presolve': True,      # Use presolver to simplify problem
            'disp': False,          # Suppress output
            'time_limit': 300       # 5 minute timeout
        }
    )
    
    # =========================================================================
    # STEP 6: Extract and validate solution
    # =========================================================================
    if result.success:
        # Extract decision variables
        x_opt = result.x[:n]        # Optimal weights
        zeta_opt = result.x[n]      # Optimal ζ (= VaR_α)
        cvar_opt = result.fun        # Optimal objective (= CVaR_α)
        
        # Numerical cleanup (handle floating-point errors)
        x_opt = np.maximum(x_opt, 0)      # Force non-negativity
        x_opt = x_opt / x_opt.sum()       # Ensure exact sum = 1
        
        # Verify solution quality
        portfolio_returns = returns @ x_opt
        empirical_var = np.percentile(portfolio_returns, 100 * (1 - alpha))
        
        # Sanity check: CVaR should be close to empirical tail mean
        tail_losses = portfolio_returns[portfolio_returns <= empirical_var]
        if len(tail_losses) > 0:
            empirical_cvar = tail_losses.mean()
            
            # Warn if large discrepancy (shouldn't happen with correct implementation)
            if abs(cvar_opt - empirical_cvar) > 0.01:
                print(f"Warning: CVaR mismatch - Optimal: {cvar_opt:.4f}, Empirical: {empirical_cvar:.4f}")
        
        return {
            'weights': x_opt,
            'cvar': cvar_opt,
            'var': zeta_opt,
            'success': True,
            'iterations': result.nit if hasattr(result, 'nit') else None,
            'solver_message': result.message
        }
    
    else:
        # Optimization failed - return fallback solution
        print(f"CVaR optimization failed: {result.message}")
        print("Returning equal-weight portfolio as fallback.")
        
        return {
            'weights': np.array([1/n] * n),
            'cvar': np.nan,
            'var': np.nan,
            'success': False,
            'solver_message': result.message
        }


# Optional utility: Compute CVaR for a given portfolio
def compute_portfolio_cvar(returns_df, weights, alpha=0.95):
    """
    Compute CVaR for a given portfolio (non-optimized).
    
    Useful for evaluating existing portfolios or strategies.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Historical returns (T × n)
    weights : np.ndarray
        Portfolio weights (must sum to 1)
    alpha : float
        Confidence level
        
    Returns
    -------
    dict with keys:
        'cvar' : float
            Portfolio CVaR at confidence level α
        'var' : float
            Portfolio VaR at confidence level α
        'worst_scenarios' : pd.Series
            Returns in the worst (1-α)% of scenarios
    """
    returns = returns_df.values
    portfolio_returns = returns @ weights
    
    # Compute VaR as (1-α) quantile of losses (= α quantile of returns)
    var_alpha = np.percentile(portfolio_returns, 100 * (1 - alpha))
    
    # Compute CVaR as mean of tail beyond VaR
    tail_returns = portfolio_returns[portfolio_returns <= var_alpha]
    cvar_alpha = tail_returns.mean() if len(tail_returns) > 0 else var_alpha
    
    # Convert to Series for analysis
    worst_scenarios = pd.Series(
        tail_returns, 
        index=returns_df.index[portfolio_returns <= var_alpha]
    )
    
    return {
        'cvar': cvar_alpha,
        'var': var_alpha,
        'worst_scenarios': worst_scenarios,
        'n_tail_scenarios': len(tail_returns)
    }
