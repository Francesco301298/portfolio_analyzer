import pandas as pd
import numpy as np
import warnings
from scipy import linalg
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import t as t_dist
from scipy import special
from scipy import special
from sklearn.linear_model import Lasso
from statsmodels.api import add_constant, WLS

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




def fit_var1(x, p=None, *, nu=10**9, tol=1e-2, maxiter=500, shrink=False, lam=0.):
    
    """This function fits a VAR(1) model to the input time series data, estimating coefficients for autoregressive relationships, handling both stationary and cointegrated processes, 
    and providing parameter estimates such as coefficients, mean errors, and covariance matrices.

    Parameters
    ----------
        x : array, shape (t_bar, n_bar)
        p : array, shape(t_bar, )
        nu : scalar, optional
        tol : scalar, optional
        maxiter: scalar, optional
        shrink : bool, optional
        lam : float, optional

    Returns
    -------
        b_hat : array, shape(n_bar, n_bar)
        mu_epsi_hat : array, shape(n_bar, )
        sigma2_hat : array, shape(n_bar, n_bar)

    """

    t_bar = x.shape[0]

    if len(x.shape) == 1:
        x = x.reshape((t_bar, 1))
        n_bar = 1
    else:
        n_bar = x.shape[1]

    if p is None:
        p = np.ones(t_bar)/t_bar

    # step 0: find cointegrated relationships
    c_hat, _ = cointegration_fp(x, p, b_threshold=1)
    l_bar = c_hat.shape[1]

    # step 1: fit VAR(1)
    if l_bar == n_bar:
        # step 1a: fit stationary process
        x_t = x[1:, :]
        x_tm1 = x[:-1, :]
        mu_epsi_hat, b_hat, sig2_hat, _ = fit_lfm_mlfp(x_t, x_tm1, p[1:]/np.sum(p[1:]), nu, tol=tol,
                                          maxiter=maxiter, shrink=shrink, lam=lam)
    if l_bar < n_bar and l_bar > 0:
        # step 1b: fit cointegrated process
        x_t = np.diff(x, axis=0)
        x_tm1 = (x@c_hat)[:-1, :]
        mu_epsi_hat, d_hat, sig2_hat, _ = fit_lfm_mlfp(x_t, x_tm1, p[1:]/np.sum(p[1:]), nu, tol=tol,
                                          maxiter=maxiter, shrink=shrink, lam=lam)
        if np.ndim(d_hat) < 2:
            d_hat = d_hat.reshape((-1, 1))
        b_hat = np.eye(n_bar) + d_hat@c_hat.T

    if l_bar == 0:
        # step 1c: fit VAR(1) to differences
        warnings.warn('Warning: non-cointegrated series detected. ' +
                      'Fit performed on differences')
        delta_x_t = np.diff(x, axis=0)[1:]
        delta_x_tm1 = np.diff(x, axis=0)[:-1]
        mu_epsi_hat, b_hat, sig2_hat, _ = fit_lfm_mlfp(delta_x_t, delta_x_tm1, p[2:]/np.sum(p[2:]), nu,
                                          tol=tol, maxiter=maxiter, shrink=shrink, lam=lam)
        # b_hat is close to zero: detect random walk
        if np.linalg.norm(b_hat.reshape(-1, 1)) < tol:
            # random walk
            warnings.warn('Warning: a random walk has been fitted')
            b_hat = np.eye(n_bar)

    return np.squeeze(b_hat), np.squeeze(mu_epsi_hat), np.squeeze(sig2_hat)



def cointegration_fp(x, p=None, *, b_threshold=0.99):
    
    """This function estimates cointegration vectors and their coefficients from a given dataset. 

    Parameters
    ----------
         x : array, shape(t_bar, d_bar)
         p : array, shape(t_bar, d_bar)
         b_threshold : scalar

    Returns
    -------
        c_hat : array, shape(d_bar, l_bar)
        beta_hat : array, shape(l_bar, )

    """

    t_bar = x.shape[0]
    if len(x.shape) == 1:
        x = x.reshape((t_bar, 1))
        d_bar = 1
    else:
        d_bar = x.shape[1]

    if p is None:
        p = np.ones(t_bar)/t_bar  # uniform probabilities

    # estimate HFP covariance matrix
    sigma2_hat = DescrStatsW(x, weights=p).cov 

    # find eigenvectors
    e_hat = linalg.eigh(sigma2_hat, subset_by_index=(0, sigma2_hat.shape[0] - 1))[1]
    e_hat = e_hat[:, ::-1]
    # enforce a sign convention on the coefficients
    # the largest element in each eigenvector will have a positive sign
    ind = np.argmax(abs(e_hat), axis=0)
    ind = np.diag(e_hat[ind, :]) < 0
    e_hat[:, ind] = -e_hat[:, ind]

    # detect cointegration vectors
    c_hat = []
    b_hat = []
    p = p[:-1]
    for d in np.arange(0, d_bar):
        # define series
        y_t = e_hat[:, d]@x.T

        # fit AR(1)
        yt = y_t[1:].reshape((-1, 1))
        ytm1 = y_t[:-1].reshape((-1, 1))
        _, b, _, _ = fit_lfm_mlfp(yt, ytm1, p/np.sum(p))
        if np.ndim(b) < 2:
            b = np.array(b).reshape(-1, 1)

        # check stationarity
        if abs(b[0, 0]) <= b_threshold:
            c_hat.append(list(e_hat[:, d]))
            b_hat.append(b[0, 0])

    # output
    c_hat = np.array(c_hat).T
    b_hat = np.array(b_hat)

    # sort according to the AR(1) parameters beta_hat
    c_hat = c_hat[:, np.argsort(b_hat)]
    b_hat = np.sort(b_hat)

    return c_hat, b_hat


def fit_lfm_mlfp(x, z, p=None, nu=4, tol=1e-3, fit_intercept=True, maxiter=500, 
                 print_iter=False, rescale=False, shrink=False, lam=0.):
    """This function fits a Linear Factor Model (LFM) using Maximum Likelihood with Flexible Probabilities (MLFP). 
    It estimates the factor loadings, intercepts, residual covariances, and residuals based on the input time series data.

    Parameters
    ----------
        x : array, shape (t_bar, n_bar) if n_bar>1 or (t_bar, ) for n_bar=1
        z : array, shape (t_bar, k_bar) if k_bar>1 or (t_bar, ) for k_bar=1
        p : array, optional, shape (t_bar,)
        nu : scalar, optional
        tol : float, optional
        fit_intercept: bool, optional
        maxiter : scalar, optional
        print_iter : bool, optional
        rescale : bool, optional
        shrink : bool, optional
        lam : float, optional

    Returns
    -------
       alpha : array, shape (n_bar,)
       beta : array, shape (n_bar, k_bar) if k_bar>1 or (n_bar, ) for k_bar=1
       sigma2 : array, shape (n_bar, n_bar)
       eps : shape (t_bar, n_bar) if n_bar>1 or (t_bar, ) for n_bar=1
    """

    if np.ndim(x) < 2:
        x = x.reshape(-1, 1).copy()
    t_bar, n_bar = x.shape
    if np.ndim(z) < 2:
        z = z.reshape(-1, 1).copy()
    t_bar, n_bar = x.shape
    k_bar = z.shape[1]

    if p is None:
        p = np.ones(t_bar)/t_bar

    # rescale the variables
    if rescale is True:
        sigma2_x = np.cov(x.T, aweights=p)
        sigma_x = np.sqrt(np.diag(sigma2_x))
        x = x.copy()/sigma_x

        sigma2_z = np.cov(z.T, aweights=p)
        sigma_z = np.sqrt(np.diag(sigma2_z))
        z = z.copy()/sigma_z

    # Step 0: Set initial values using method of moments
    if shrink:
        if lam == 0:
            # weighted least squares regression model
            model = WLS(x, add_constant(z), weights=p).fit(fit_intercept=fit_intercept)  
            # parameters of transition equation
            alpha = np.atleast_1d(model.params[0])  # shifts
            beta = np.atleast_2d(model.params[1:])  # loadings
            eps = model.resid  # residuals
            sigma2 = np.atleast_2d(np.cov(eps.T, aweights=p))  # dispersion
        else:
            if fit_intercept is True:
                m_x = p@x
                m_z = p@z
            else:
                m_x = np.zeros(n_bar,)
                m_z = np.zeros(k_bar,)
            x_p = ((x - m_x).T*np.sqrt(p)).T
            z_p = ((z - m_z).T*np.sqrt(p)).T
            clf = Lasso(alpha=lam/(2.*t_bar), fit_intercept=False)
            clf.fit(z_p, x_p)
            beta = clf.coef_
            if k_bar == 1:
                alpha = m_x - beta*m_z
                eps = x - alpha - z*beta
            else:
                alpha = m_x - beta@m_z
                eps = x - alpha - z@np.atleast_2d(beta).T
            sigma2 = np.cov(eps.T, aweights=p)
    else:
        # weighted least squares regression model
        model = WLS(x, add_constant(z), weights=p).fit(fit_intercept=fit_intercept)  
        # parameters of transition equation
        alpha = np.atleast_1d(model.params[0])  # shifts
        beta = np.atleast_2d(model.params[1:])  # loadings
        eps = model.resid  # residuals
        sigma2 = np.atleast_2d(np.cov(eps.T, aweights=p))  # dispersion
    alpha, beta, sigma2, eps = alpha.reshape((n_bar, 1)), beta.reshape((n_bar, k_bar)),\
                               sigma2.reshape((n_bar, n_bar)), eps.reshape((t_bar, n_bar))
    if nu > 2.:
        # if nu <=2, then the covariance is not defined
        sigma2 = (nu - 2.)/nu*sigma2

    mu_eps = np.zeros(n_bar)
    for i in range(maxiter):
        # step 1: update the weights and historical flexible probabilities
        if nu >= 1e3 and np.linalg.det(sigma2) < 1e-13:
            w = np.ones(t_bar)
        else:
            w = (nu + n_bar)/(nu + np.sum((eps - mu_eps).T*np.linalg.solve(sigma2, (eps - mu_eps).T), axis=0))
        q = w*p
        q = q/np.sum(q)

        # step 2: update shift parameters, factor loadings and covariance
        alpha_old, beta_old, sigma2_old = alpha, beta, sigma2
        if shrink:
            if lam == 0:
                # weighted least squares regression model
                model = WLS(x, add_constant(z), weights=q).fit(fit_intercept=fit_intercept)  
                # parameters of transition equation
                alpha = np.atleast_1d(model.params[0])  # shifts
                beta = np.atleast_2d(model.params[1:])  # loadings
                eps = model.resid  # residuals
                sigma2 = np.atleast_2d(np.cov(eps.T, aweights=q))  # dispersion
            else:
                if fit_intercept is True:
                    m_x = q@x
                    m_z = q@z
                else:
                    m_x = np.zeros(n_bar,)
                    m_z = np.zeros(k_bar,)
                x_q = ((x - m_x).T * np.sqrt(q)).T
                z_q = ((z - m_z).T * np.sqrt(q)).T
                clf = Lasso(alpha=lam/(2.*t_bar), fit_intercept=False)
                clf.fit(z_q, x_q)
                beta = clf.coef_
                if k_bar == 1:
                    alpha = m_x - beta*m_z
                    eps = x - alpha - z*beta
                else:
                    alpha = m_x - beta@m_z
                    eps = x - alpha - z@np.atleast_2d(beta).T
                sigma2 = np.cov(eps.T, aweights=q)
        else:
            # weighted least squares regression model
            model = WLS(x, add_constant(z), weights=q).fit(fit_intercept=fit_intercept)  
            # parameters of transition equation
            alpha = np.atleast_1d(model.params[0])  # shifts
            beta = np.atleast_2d(model.params[1:])  # loadings
            eps = model.resid  # residuals
            sigma2 = np.atleast_2d(np.cov(eps.T, aweights=q))  # dispersion
        alpha, beta, sigma2, eps = alpha.reshape((n_bar, 1)), beta.reshape((n_bar, k_bar)),\
                                   sigma2.reshape((n_bar, n_bar)), eps.reshape((t_bar, n_bar))
        sigma2 = (w@q)*sigma2

        # step 3: check convergence
        beta_tilde_old = np.column_stack((alpha_old, beta_old))
        beta_tilde = np.column_stack((alpha, beta))
        errors = [np.linalg.norm(beta_tilde - beta_tilde_old, ord=np.inf)/np.linalg.norm(beta_tilde_old, ord=np.inf),
                  np.linalg.norm(sigma2 - sigma2_old, ord=np.inf)/np.linalg.norm(sigma2_old, ord=np.inf)]
        # print the loglikelihood and the error
        if print_iter is True:
            if np.shape(sigma2) == 0:
                # univaraite student t
                lf = stats.t.logpdf(eps, nu, mu_eps, sigma2)
            else:
                # multivariate student t
                n_bar = sigma2.shape[0]
                d2 = np.sum((eps - mu_eps).T*np.linalg.solve(sigma2, (eps - mu_eps).T), axis=0)
                lf = -((nu + n_bar)/2.)*np.log(1. + d2/nu) + special.gammaln((nu + n_bar)/2.) -\
                     special.gammaln(nu/2.) - (n_bar/2.)*np.log(nu*np.pi) - 0.5*np.linalg.slogdet(sigma2)[1]
            print('Iter: %i; Loglikelihood: %.5f; Errors: %.3e' %(i, q@lf, max(errors)))
        if max(errors) < tol:
            break
    if rescale is True:
        alpha = alpha*sigma_x
        beta = ((beta/sigma_z).T*sigma_x).T
        sigma2 = (sigma2.T*sigma_x).T*sigma_x

    return np.squeeze(alpha), np.squeeze(beta), np.squeeze(sigma2), np.squeeze(eps)
