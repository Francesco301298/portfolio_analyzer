"""
Performance Metrics and Statistical Tests for Backtesting

This module provides functions for computing performance metrics,
statistical significance tests, and robustness analysis.

References:
    - Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
    - Harvey, C. et al. (2016). "...and the Cross-Section of Expected Returns"
    - Ledoit, O. & Wolf, M. (2008). "Robust Performance Hypothesis Testing"

Author: Portfolio Analyzer
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str


def compute_sharpe_ratio(
    returns: pd.Series,
    rf_rate: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = 252
) -> float:
    """
    Compute Sharpe Ratio.
    
    Args:
        returns: Series of returns
        rf_rate: Risk-free rate (annualized)
        annualize: Whether to annualize the ratio
        periods_per_year: Trading periods per year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - rf_rate / periods_per_year
    
    if annualize:
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    else:
        return excess_returns.mean() / returns.std()


def compute_sortino_ratio(
    returns: pd.Series,
    rf_rate: float = 0.0,
    target_return: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = 252
) -> float:
    """
    Compute Sortino Ratio (using downside deviation).
    
    Args:
        returns: Series of returns
        rf_rate: Risk-free rate (annualized)
        target_return: Minimum acceptable return
        annualize: Whether to annualize the ratio
        periods_per_year: Trading periods per year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - rf_rate / periods_per_year
    
    # Downside returns (below target)
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        return np.inf  # No downside
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std == 0:
        return np.inf
    
    if annualize:
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    else:
        return excess_returns.mean() / downside_std


def compute_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Compute maximum drawdown and its timing.
    
    Args:
        equity_curve: Cumulative equity curve
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    rolling_max = equity_curve.expanding().max()
    drawdowns = equity_curve / rolling_max - 1
    
    max_dd = drawdowns.min()
    trough_idx = drawdowns.idxmin()
    
    # Find peak (before trough)
    peak_idx = equity_curve[:trough_idx].idxmax()
    
    return max_dd, peak_idx, trough_idx


def compute_calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute Calmar Ratio (annualized return / max drawdown).
    
    Args:
        returns: Series of returns
        equity_curve: Cumulative equity curve
        periods_per_year: Trading periods per year
        
    Returns:
        Calmar ratio
    """
    ann_return = returns.mean() * periods_per_year
    max_dd, _, _ = compute_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return np.inf
    
    return ann_return / abs(max_dd)


def compute_var_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Value at Risk and Conditional VaR (Expected Shortfall).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    alpha = 1 - confidence_level
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean()
    
    return var, cvar


def compute_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True,
    periods_per_year: int = 252
) -> float:
    """
    Compute Information Ratio.
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        annualize: Whether to annualize
        periods_per_year: Trading periods per year
        
    Returns:
        Information ratio
    """
    excess_returns = strategy_returns - benchmark_returns
    tracking_error = excess_returns.std()
    
    if tracking_error == 0:
        return 0.0
    
    ir = excess_returns.mean() / tracking_error
    
    if annualize:
        ir *= np.sqrt(periods_per_year)
    
    return ir


def compute_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> Tuple[float, float, float, float]:
    """
    Compute alpha and beta via OLS regression.
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Trading periods per year
        
    Returns:
        Tuple of (alpha_annualized, beta, alpha_tstat, alpha_pvalue)
    """
    X = np.column_stack([np.ones(len(benchmark_returns)), benchmark_returns.values])
    y = strategy_returns.values
    
    # OLS
    try:
        coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        alpha_daily, beta = coef[0], coef[1]
    except:
        return 0, 1, 0, 1
    
    # Standard errors
    y_pred = X @ coef
    residuals = y - y_pred
    n = len(y)
    k = 2  # Number of parameters
    
    mse = np.sum(residuals ** 2) / (n - k)
    
    try:
        var_coef = mse * np.linalg.inv(X.T @ X)
        se_alpha = np.sqrt(var_coef[0, 0])
    except:
        se_alpha = 1
    
    # T-statistic for alpha
    t_stat = alpha_daily / se_alpha if se_alpha > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
    
    # Annualize alpha
    alpha_annual = alpha_daily * periods_per_year
    
    return alpha_annual, beta, t_stat, p_value


# ═══════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════════

def test_sharpe_significance(
    returns: pd.Series,
    null_sharpe: float = 0.0,
    confidence_level: float = 0.95
) -> StatisticalTestResult:
    """
    Test if Sharpe ratio is significantly different from null hypothesis.
    
    Uses the Lo (2002) adjustment for autocorrelation.
    
    Args:
        returns: Series of returns
        null_sharpe: Null hypothesis Sharpe ratio
        confidence_level: Confidence level for the test
        
    Returns:
        StatisticalTestResult object
    """
    n = len(returns)
    sr = compute_sharpe_ratio(returns, annualize=False)
    
    # Standard error of Sharpe ratio (Lo, 2002)
    # SE(SR) ≈ sqrt((1 + 0.5 * SR^2) / n)
    se_sr = np.sqrt((1 + 0.5 * sr ** 2) / n)
    
    # T-statistic
    t_stat = (sr - null_sharpe) / se_sr if se_sr > 0 else 0
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    
    is_significant = p_value < (1 - confidence_level)
    
    if is_significant:
        interpretation = f"Sharpe ratio ({sr:.3f}) is significantly different from {null_sharpe} (p={p_value:.4f})"
    else:
        interpretation = f"Cannot reject that Sharpe ratio equals {null_sharpe} (p={p_value:.4f})"
    
    return StatisticalTestResult(
        test_name="Sharpe Ratio Significance Test",
        statistic=t_stat,
        p_value=p_value,
        is_significant=is_significant,
        confidence_level=confidence_level,
        interpretation=interpretation
    )


def test_strategy_vs_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    confidence_level: float = 0.95
) -> StatisticalTestResult:
    """
    Test if strategy returns are significantly different from benchmark.
    
    Uses paired t-test on excess returns.
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        confidence_level: Confidence level
        
    Returns:
        StatisticalTestResult object
    """
    excess_returns = strategy_returns - benchmark_returns
    
    # Paired t-test (H0: mean excess return = 0)
    t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
    
    is_significant = p_value < (1 - confidence_level)
    
    mean_excess = excess_returns.mean() * 252  # Annualized
    
    if is_significant:
        if mean_excess > 0:
            interpretation = f"Strategy significantly outperforms benchmark by {mean_excess*100:.2f}% annually (p={p_value:.4f})"
        else:
            interpretation = f"Strategy significantly underperforms benchmark by {abs(mean_excess)*100:.2f}% annually (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference between strategy and benchmark (p={p_value:.4f})"
    
    return StatisticalTestResult(
        test_name="Strategy vs Benchmark Test",
        statistic=t_stat,
        p_value=p_value,
        is_significant=is_significant,
        confidence_level=confidence_level,
        interpretation=interpretation
    )


def test_win_rate_significance(
    wins: int,
    total_trades: int,
    null_win_rate: float = 0.5,
    confidence_level: float = 0.95
) -> StatisticalTestResult:
    """
    Test if win rate is significantly different from random (50%).
    
    Uses binomial test.
    
    Args:
        wins: Number of winning trades
        total_trades: Total number of trades
        null_win_rate: Null hypothesis win rate
        confidence_level: Confidence level
        
    Returns:
        StatisticalTestResult object
    """
    if total_trades == 0:
        return StatisticalTestResult(
            test_name="Win Rate Significance Test",
            statistic=0,
            p_value=1,
            is_significant=False,
            confidence_level=confidence_level,
            interpretation="No trades to analyze"
        )
    
    # Binomial test
    result = stats.binomtest(wins, total_trades, null_win_rate)
    p_value = result.pvalue
    
    observed_win_rate = wins / total_trades
    is_significant = p_value < (1 - confidence_level)
    
    if is_significant:
        if observed_win_rate > null_win_rate:
            interpretation = f"Win rate ({observed_win_rate:.1%}) is significantly above {null_win_rate:.0%} (p={p_value:.4f})"
        else:
            interpretation = f"Win rate ({observed_win_rate:.1%}) is significantly below {null_win_rate:.0%} (p={p_value:.4f})"
    else:
        interpretation = f"Win rate ({observed_win_rate:.1%}) is not significantly different from {null_win_rate:.0%} (p={p_value:.4f})"
    
    return StatisticalTestResult(
        test_name="Win Rate Significance Test",
        statistic=observed_win_rate,
        p_value=p_value,
        is_significant=is_significant,
        confidence_level=confidence_level,
        interpretation=interpretation
    )


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════

def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    block_size: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for Sharpe ratio.
    
    Uses block bootstrap to preserve autocorrelation structure.
    
    Args:
        returns: Series of returns
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        block_size: Size of blocks for block bootstrap (None = sqrt(n))
        
    Returns:
        Tuple of (lower_bound, point_estimate, upper_bound)
    """
    n = len(returns)
    returns_array = returns.values
    
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))
    
    # Point estimate
    point_estimate = compute_sharpe_ratio(returns)
    
    # Bootstrap
    bootstrap_sharpes = []
    
    for _ in range(n_bootstrap):
        # Block bootstrap
        n_blocks = int(np.ceil(n / block_size))
        block_starts = np.random.randint(0, n - block_size + 1, n_blocks)
        
        # Construct bootstrap sample
        bootstrap_sample = []
        for start in block_starts:
            bootstrap_sample.extend(returns_array[start:start + block_size])
        bootstrap_sample = np.array(bootstrap_sample[:n])
        
        # Compute Sharpe
        sr = np.sqrt(252) * np.mean(bootstrap_sample) / np.std(bootstrap_sample) if np.std(bootstrap_sample) > 0 else 0
        bootstrap_sharpes.append(sr)
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
    upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)
    
    return lower, point_estimate, upper


def bootstrap_metric_ci(
    returns: pd.Series,
    metric_func: callable,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Generic bootstrap CI for any metric.
    
    Args:
        returns: Series of returns
        metric_func: Function that takes returns and returns a scalar
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        
    Returns:
        Tuple of (lower_bound, point_estimate, upper_bound)
    """
    n = len(returns)
    returns_array = returns.values
    
    point_estimate = metric_func(returns)
    
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        # Simple bootstrap (with replacement)
        indices = np.random.randint(0, n, n)
        bootstrap_sample = pd.Series(returns_array[indices])
        bootstrap_metrics.append(metric_func(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
    
    return lower, point_estimate, upper


# ═══════════════════════════════════════════════════════════════
# SIGNAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def compute_signal_decay(
    returns: pd.Series,
    signals: pd.Series,
    signal_threshold: float = 2.0,
    max_horizon: int = 30
) -> pd.DataFrame:
    """
    Compute average cumulative returns after signal.
    
    This measures how quickly the "alpha" from a signal decays.
    
    Args:
        returns: Series of returns
        signals: Series of Z-scores
        signal_threshold: Threshold for signal activation
        max_horizon: Maximum days to track after signal
        
    Returns:
        DataFrame with columns ['horizon', 'cum_return_long', 'cum_return_short', 'n_signals']
    """
    results = []
    
    # Find signal dates
    long_signals = signals < -signal_threshold  # Negative Z → long
    short_signals = signals > signal_threshold   # Positive Z → short
    
    for horizon in range(1, max_horizon + 1):
        # Long signals: cumulative return after signal
        long_cum_returns = []
        for i, (date, is_signal) in enumerate(long_signals.items()):
            if is_signal and i + horizon < len(returns):
                future_returns = returns.iloc[i:i + horizon]
                long_cum_returns.append((1 + future_returns).prod() - 1)
        
        # Short signals
        short_cum_returns = []
        for i, (date, is_signal) in enumerate(short_signals.items()):
            if is_signal and i + horizon < len(returns):
                future_returns = returns.iloc[i:i + horizon]
                short_cum_returns.append(-((1 + future_returns).prod() - 1))  # Negative for short
        
        results.append({
            'horizon': horizon,
            'cum_return_long': np.mean(long_cum_returns) if long_cum_returns else 0,
            'cum_return_short': np.mean(short_cum_returns) if short_cum_returns else 0,
            'n_long_signals': len(long_cum_returns),
            'n_short_signals': len(short_cum_returns)
        })
    
    return pd.DataFrame(results)


def compute_factor_contribution(
    trades: List,
    strategy_returns: pd.Series,
    n_components: int
) -> Dict[int, float]:
    """
    Compute P&L contribution from each factor.
    
    Args:
        trades: List of TradeRecord objects
        strategy_returns: Strategy returns series
        n_components: Number of PCA components
        
    Returns:
        Dictionary mapping factor index to total P&L contribution
    """
    factor_pnl = {k: 0.0 for k in range(1, n_components + 1)}
    
    # This is an approximation - attribute returns to active signals
    # In reality, multiple factors may be active simultaneously
    
    for trade in trades:
        trade_date = trade.date
        if trade_date in strategy_returns.index:
            daily_return = strategy_returns[trade_date]
            
            # Attribute proportionally to active signals
            total_abs_z = sum(abs(z) for z in trade.active_signals.values())
            
            if total_abs_z > 0:
                for pc, z in trade.active_signals.items():
                    factor_pnl[pc] += daily_return * (abs(z) / total_abs_z)
    
    return factor_pnl


def compute_regime_performance(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    regimes: pd.Series
) -> pd.DataFrame:
    """
    Compute performance breakdown by regime.
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        regimes: Series of regime labels
        
    Returns:
        DataFrame with performance metrics by regime
    """
    results = []
    
    for regime in regimes.unique():
        mask = regimes == regime
        
        if mask.sum() == 0:
            continue
        
        strat_ret = strategy_returns[mask]
        bench_ret = benchmark_returns[mask]
        
        n_days = mask.sum()
        
        results.append({
            'regime': regime,
            'n_days': n_days,
            'pct_time': n_days / len(regimes) * 100,
            'strategy_return': strat_ret.mean() * 252,
            'benchmark_return': bench_ret.mean() * 252,
            'excess_return': (strat_ret.mean() - bench_ret.mean()) * 252,
            'strategy_vol': strat_ret.std() * np.sqrt(252),
            'strategy_sharpe': compute_sharpe_ratio(strat_ret),
            'benchmark_sharpe': compute_sharpe_ratio(bench_ret)
        })
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════

def compute_parameter_sensitivity(
    prices: pd.DataFrame,
    base_config: dict,
    param_name: str,
    param_values: List,
    metric: str = 'sharpe_strategy'
) -> pd.DataFrame:
    """
    Compute sensitivity of a metric to a parameter.
    
    Args:
        prices: Price DataFrame
        base_config: Base configuration dictionary
        param_name: Name of parameter to vary
        param_values: List of values to test
        metric: Metric to track
        
    Returns:
        DataFrame with parameter values and resulting metrics
    """
    from .engine import PCAOUBacktester, BacktestConfig
    
    results = []
    
    for value in param_values:
        config_dict = base_config.copy()
        config_dict[param_name] = value
        
        try:
            config = BacktestConfig(**config_dict)
            backtester = PCAOUBacktester(prices, config)
            bt_results = backtester.run()
            
            results.append({
                param_name: value,
                metric: bt_results.metrics.get(metric, np.nan),
                'n_trades': bt_results.metrics.get('n_trades', 0)
            })
        except Exception as e:
            results.append({
                param_name: value,
                metric: np.nan,
                'n_trades': 0
            })
    
    return pd.DataFrame(results)

