import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import t as t_dist
from scipy import special

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

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

def fit_locdisp_mlfp(eps, p=None, nu=1000, threshold=1e-3, maxiter=1000):
    """
    Maximum Likelihood with Flexible Probabilities (MLFP) estimation.
    
    Estimates location and dispersion parameters of a multivariate time series
    using the maximum likelihood method with flexible probabilities.
    
    Based on Meucci, A. (2010). "Fully Flexible Views: Theory and Practice"
    
    Parameters
    ----------
    eps : array, shape (t_bar,) or (t_bar, i_bar)
        Time series of residuals/invariants
    p : array, shape (t_bar,), optional
        Flexible probabilities (default: equal weights)
    nu : float, optional
        Degrees of freedom for Student-t (default: 1000 ≈ Gaussian)
    threshold : float, optional
        Convergence threshold
    maxiter : int, optional
        Maximum iterations
        
    Returns
    -------
    mu : array
        Location parameter
    sigma2 : array
        Dispersion parameter (covariance matrix or variance)
    """
    if len(eps.shape) == 1:
        eps = eps.reshape(-1, 1)

    t_bar, i_bar = eps.shape

    if p is None:
        p = np.ones(t_bar) / t_bar

    # Step 0: Initialize with method of moments
    mu = p @ eps
    sigma2 = ((eps - mu).T * p) @ (eps - mu)

    if nu > 2.:
        sigma2 = sigma2 * (nu - 2.) / nu

    for i in range(maxiter):
        # Step 1: Update weights
        if nu >= 1e3 and np.linalg.det(sigma2) < 1e-13:
            w = np.ones(t_bar)
        else:
            if eps.shape[1] > 1:
                w = (nu + i_bar) / (nu + np.squeeze(
                    np.sum((eps - mu).T * np.linalg.solve(sigma2, (eps - mu).T), axis=0)))
            else:
                w = (nu + i_bar) / (nu + np.squeeze((eps - mu) / sigma2 * (eps - mu)))
        q = w * p
        
        # Step 2: Update location and dispersion
        mu_old, sigma2_old = mu.copy(), sigma2.copy()
        mu = q @ eps
        sigma2 = ((eps - mu).T * q) @ (eps - mu)
        mu = mu / np.sum(q)

        # Step 3: Check convergence
        er = max(
            np.linalg.norm(mu - mu_old, ord=np.inf) / (np.linalg.norm(mu_old, ord=np.inf) + 1e-10),
            np.linalg.norm(sigma2 - sigma2_old, ord=np.inf) / (np.linalg.norm(sigma2_old, ord=np.inf) + 1e-10)
        )

        if er <= threshold:
            break
            
    return np.squeeze(mu), np.squeeze(sigma2)


def fit_dcc_t(dx, p=None, rho2=None, param0=None, g=0.99):
    """
    Dynamic Conditional Correlation (DCC) model estimation.
    
    Estimates a DCC model for multivariate time series by minimizing
    the negative log-likelihood.
    
    Based on Engle, R. (2002). "Dynamic Conditional Correlation"
    
    The DCC model evolves as:
        Q_t = (1-a-b)*rho_bar + a*eps_{t-1}*eps_{t-1}' + b*Q_{t-1}
        R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
    
    Parameters
    ----------
    dx : array, shape (t_bar, i_bar)
        Standardized residuals (quasi-invariants)
    p : array, shape (t_bar,), optional
        Flexible probabilities
    rho2 : array, shape (i_bar, i_bar), optional
        Target (unconditional) correlation matrix
    param0 : list, optional
        Initial parameters [a, b]
    g : float, optional
        Stationarity constraint: a + b < g
        
    Returns
    -------
    params : list [c, a, b]
        DCC parameters where c = 1 - a - b
    r2_t : array, shape (t_bar, i_bar, i_bar)
        Time series of conditional correlation matrices
    eps : array, shape (t_bar, i_bar)
        Transformed residuals
    q2_t_bar : array, shape (i_bar, i_bar)
        Final Q matrix (for forecasting)
    """
    t_bar, i_bar = dx.shape
    
    # Default: equal probabilities
    if p is None:
        p = np.ones(t_bar) / t_bar
    
    # Default: sample correlation as target
    if rho2 is None:
        cov = np.cov(dx.T, aweights=p)
        rho2 = np.diag(1 / np.sqrt(np.diag(cov))) @ cov @ np.diag(1 / np.sqrt(np.diag(cov)))
    
    # Default: initial parameters
    if param0 is None:
        param0 = [0.01, g - 0.01]

    def neg_log_likelihood(params):
        """Compute negative log-likelihood of DCC model."""
        a, b = params
        mu = np.zeros(i_bar)
        q2_t = rho2.copy()
        r2_t = np.diag(1 / np.sqrt(np.diag(q2_t))) @ q2_t @ np.diag(1 / np.sqrt(np.diag(q2_t)))
        llh = 0.0
        
        for t in range(t_bar):
            try:
                llh = llh - p[t] * multivariate_normal.logpdf(dx[t, :], mu, r2_t)
            except:
                llh = llh + 1e10  # Penalty for non-positive definite
            q2_t = rho2 * (1 - a - b) + a * np.outer(dx[t, :], dx[t, :]) + b * q2_t
            r2_t = np.diag(1 / np.sqrt(np.diag(q2_t) + 1e-10)) @ q2_t @ np.diag(1 / np.sqrt(np.diag(q2_t) + 1e-10))
        
        return llh

    # Optimization with bounds and constraints
    bounds = ((1e-6, 0.5), (1e-6, 0.99))
    constraints = {'type': 'ineq', 'fun': lambda param: g - param[0] - param[1]}
    
    try:
        result = minimize(neg_log_likelihood, param0, bounds=bounds, constraints=constraints, method='SLSQP')
        a, b = result['x']
    except:
        a, b = 0.05, 0.90  # Fallback values

    # Compute realized correlations and residuals
    q2_t = rho2.copy()
    r2_t = np.zeros((t_bar, i_bar, i_bar))
    r2_t[0, :, :] = np.diag(1 / np.sqrt(np.diag(q2_t))) @ q2_t @ np.diag(1 / np.sqrt(np.diag(q2_t)))
    
    for t in range(t_bar - 1):
        q2_t = rho2 * (1 - a - b) + a * np.outer(dx[t, :], dx[t, :]) + b * q2_t
        r2_t[t + 1, :, :] = np.diag(1 / np.sqrt(np.diag(q2_t) + 1e-10)) @ q2_t @ np.diag(1 / np.sqrt(np.diag(q2_t) + 1e-10))
    
    # Transform residuals
    l_t = np.linalg.cholesky(r2_t + np.eye(i_bar) * 1e-8)
    eps = np.zeros_like(dx)
    for t in range(t_bar):
        eps[t, :] = np.linalg.solve(l_t[t], dx[t, :])

    return [1. - a - b, a, b], r2_t, eps, q2_t


def compute_flexible_probabilities(t_bar, tau_hl=120):
    """
    Compute exponential decay flexible probabilities.
    
    Parameters
    ----------
    t_bar : int
        Number of observations
    tau_hl : float
        Half-life in days
        
    Returns
    -------
    p : array, shape (t_bar,)
        Probability weights (sum to 1)
    """
    p = np.exp(-(np.log(2) / tau_hl) * np.abs(t_bar - np.arange(0, t_bar)))
    return p / np.sum(p)


def extract_garch_residuals(returns_df, p=None):
    """
    Extract GARCH(1,1) standardized residuals for each asset.
    
    Parameters
    ----------
    returns_df : DataFrame
        Log-returns for each asset
    p : array, optional
        Flexible probabilities
        
    Returns
    -------
    eps : array, shape (t_bar, n_assets)
        Standardized residuals (quasi-invariants)
    cond_vols : dict
        Conditional volatilities for each asset
    garch_params : dict
        GARCH parameters for each asset
    """
    from arch import arch_model
    
    t_bar, n_assets = returns_df.shape
    eps = np.zeros((t_bar, n_assets))
    cond_vols = {}
    garch_params = {}
    
    if p is None:
        p = np.ones(t_bar) / t_bar
    
    for i, col in enumerate(returns_df.columns):
        try:
            # Fit GARCH(1,1)
            garch = arch_model(returns_df[col].values, vol='garch', p=1, o=0, q=1, rescale=True)
            garch_fitted = garch.fit(disp='off')
            
            # Store results
            eps[:, i] = garch_fitted.std_resid
            cond_vols[col] = garch_fitted.conditional_volatility
            garch_params[col] = {
                'mu': garch_fitted.params.get('mu', 0),
                'omega': garch_fitted.params['omega'],
                'alpha': garch_fitted.params['alpha[1]'],
                'beta': garch_fitted.params['beta[1]']
            }
        except Exception as e:
            # Fallback: use standardized returns
            eps[:, i] = (returns_df[col].values - returns_df[col].mean()) / returns_df[col].std()
            cond_vols[col] = np.ones(t_bar) * returns_df[col].std()
            garch_params[col] = None
    
    return eps, cond_vols, garch_params
