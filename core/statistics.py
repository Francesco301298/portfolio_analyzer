import pandas as pd
import numpy as np
from scipy import stats

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
