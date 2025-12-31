"""
Backtesting Module for Portfolio Analyzer

This module provides tools for backtesting PCA-OU mean reversion strategies
with HMM regime detection.

Main Components:
    - PCAOUBacktester: Walk-forward backtesting engine
    - BacktestConfig: Configuration dataclass
    - BacktestResults: Results dataclass
    - Performance metrics and statistical tests

Example:
    >>> from core.backtesting import PCAOUBacktester, BacktestConfig
    >>> 
    >>> config = BacktestConfig(
    ...     estimation_window=252,
    ...     z_entry=2.0,
    ...     use_hmm=True
    ... )
    >>> 
    >>> backtester = PCAOUBacktester(prices_df, config)
    >>> results = backtester.run()
    >>> 
    >>> print(f"Sharpe: {results.metrics['sharpe_strategy']:.2f}")
    >>> print(f"Alpha: {results.metrics['alpha']*100:.2f}%")
"""

from .engine import (
    PCAOUBacktester,
    BacktestConfig,
    BacktestResults,
    TradeRecord,
    Regime,
    run_backtest
)

from .metrics import (
    # Basic metrics
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_var_cvar,
    compute_information_ratio,
    compute_alpha_beta,
    
    # Statistical tests
    StatisticalTestResult,
    test_sharpe_significance,
    test_strategy_vs_benchmark,
    test_win_rate_significance,
    
    # Bootstrap
    bootstrap_sharpe_ci,
    bootstrap_metric_ci,
    
    # Signal analysis
    compute_signal_decay,
    compute_factor_contribution,
    compute_regime_performance,
    
    # Sensitivity
    compute_parameter_sensitivity
)


__all__ = [
    # Engine
    'PCAOUBacktester',
    'BacktestConfig', 
    'BacktestResults',
    'TradeRecord',
    'Regime',
    'run_backtest',
    
    # Metrics
    'compute_sharpe_ratio',
    'compute_sortino_ratio',
    'compute_max_drawdown',
    'compute_calmar_ratio',
    'compute_var_cvar',
    'compute_information_ratio',
    'compute_alpha_beta',
    
    # Statistical tests
    'StatisticalTestResult',
    'test_sharpe_significance',
    'test_strategy_vs_benchmark',
    'test_win_rate_significance',
    
    # Bootstrap
    'bootstrap_sharpe_ci',
    'bootstrap_metric_ci',
    
    # Analysis
    'compute_signal_decay',
    'compute_factor_contribution',
    'compute_regime_performance',
    'compute_parameter_sensitivity'
]

__version__ = '1.0.0'

