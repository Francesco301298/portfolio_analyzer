import numpy as np
import pandas as pd
from core.metrics import calculate_portfolio_metrics
from core.metrics import calculate_robust_metrics
from core.optimization import optimize_portfolio_weights
from core.optimization import cvar_optimization  # AGGIUNGI QUESTO IMPORT
from core.rebalancing import calculate_portfolio_with_rebalancing
from itertools import combinations

def combinatorial_purged_cv(dates, n_splits, n_test_splits, embargo_pct):
    """
    Creates train/test splits with embargo to prevent leakage.
    """
    fold_size = len(dates) // n_splits
    
    if fold_size < 20:
        raise ValueError(f"Fold size too small ({fold_size}). Reduce n_splits or use more data.")
    
    folds = []
    for i in range(n_splits):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(dates)
        folds.append(dates[start_idx:end_idx])

    splits = []
    embargo_size = int(len(dates) * embargo_pct)
    
    for test_fold_indices in combinations(range(n_splits), n_test_splits):
        test_dates = pd.Index([])
        for fold_idx in test_fold_indices:
            test_dates = test_dates.append(folds[fold_idx])
        test_dates = test_dates.sort_values()
        
        train_dates = dates.difference(test_dates)
        
        if embargo_size > 0 and len(test_dates) > 0:
            test_end = test_dates.max()
            embargo_dates = dates[(dates > test_end)][:embargo_size]
            train_dates = train_dates.difference(embargo_dates)
        
        if len(train_dates) < 50 or len(test_dates) < 10:
            continue
        
        splits.append((train_dates.sort_values(), test_dates))

    if len(splits) == 0:
        raise ValueError("No valid splits created. Adjust parameters.")
    
    return splits


def run_cpcv_backtest(
    returns_df,
    methods,
    rf_rate,
    n_splits=5,
    n_test_splits=2,
    embargo_pct=0.01
):
    """
    Run combinatorial purged cross-validation with multi-metric evaluation.
    """
    # Mapping da nomi UI a chiavi interne
    method_key_map = {
        "Equally Weighted": "equal",
        "Minimum Volatility": "min_vol",
        "Maximum Sharpe": "max_sharpe",
        "Maximum Return": "max_return",
        "Risk Parity": "risk_parity",
        "Hierarchical Risk Parity": "hrp",
        "CVaR (95%)": "cvar",
        "Your Portfolio": "custom"
    }
    
    splits = combinatorial_purged_cv(
        returns_df.index,
        n_splits,
        n_test_splits,
        embargo_pct
    )

    is_metrics_all = {m: [] for m in methods}
    oos_metrics_all = {m: [] for m in methods}
    oos_returns = {m: [] for m in methods}
    
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        
        train_returns = returns_df.loc[train_idx]
        test_returns = returns_df.loc[test_idx]

        for method in methods:
            try:
                # ===== USA IL MAPPING =====
                method_key = method_key_map.get(method, "equal")
                
                # ===== GESTISCI CVaR SEPARATAMENTE =====
                if method_key == "cvar":
                    result = cvar_optimization(train_returns, alpha=0.95)
                    weights = result['weights']
                elif method_key == "custom":
                    # Per il custom portfolio, usa i pesi definiti dall'utente
                    # (già calcolati e memorizzati in analyzer.portfolios['custom'])
                    if 'custom' in analyzer.portfolios:
                        weights = analyzer.portfolios['custom']['weights']
                    else:
                        # Fallback se custom non è disponibile
                        nan_metrics = {k: np.nan for k in ['sharpe', 'sortino', 'calmar', 'max_drawdown', 'cvar_95', 'win_rate']}
                        is_metrics_all[method].append(nan_metrics)
                        oos_metrics_all[method].append(nan_metrics)
                        continue
                else:
                    weights = optimize_portfolio_weights(
                        train_returns,
                        method=method_key,
                        rf_rate=rf_rate
                    )
                
                # Evaluate on IN-SAMPLE (training data)
                train_portfolio_returns = train_returns.dot(weights)
                is_metrics = calculate_robust_metrics(
                    train_portfolio_returns,
                    rf_rate=rf_rate
                )
                is_metrics_all[method].append(is_metrics)
                
                # Evaluate on OUT-OF-SAMPLE (test data)
                test_portfolio_returns = test_returns.dot(weights)
                oos_metrics = calculate_robust_metrics(
                    test_portfolio_returns,
                    rf_rate=rf_rate
                )
                
                oos_returns[method].append(test_portfolio_returns)
                oos_metrics_all[method].append(oos_metrics)
                
            except Exception as e:
                nan_metrics = {k: np.nan for k in ['sharpe', 'sortino', 'calmar', 'max_drawdown', 'cvar_95', 'win_rate']}
                is_metrics_all[method].append(nan_metrics)
                oos_metrics_all[method].append(nan_metrics)
                continue

    return is_metrics_all, oos_metrics_all, oos_returns


def compute_pbo(is_metrics, oos_metrics, metric_name='sharpe'):
    """
    Compute Probability of Backtest Overfitting for specified metric.
    """
    is_values = {method: [m[metric_name] for m in metrics] 
                for method, metrics in is_metrics.items()}
    oos_values = {method: [m[metric_name] for m in metrics] 
                for method, metrics in oos_metrics.items()}
    
    is_df = pd.DataFrame(is_values).dropna()
    oos_df = pd.DataFrame(oos_values).dropna()
    
    if len(is_df) == 0 or len(oos_df) == 0:
        return np.nan
    
    common_idx = is_df.index.intersection(oos_df.index)
    is_df = is_df.loc[common_idx]
    oos_df = oos_df.loc[common_idx]
    
    n_strategies = len(is_df.columns)
    
    is_ranks = is_df.rank(axis=1, ascending=False)
    best_is_strategy = is_ranks.idxmin(axis=1)
    
    oos_ranks = oos_df.rank(axis=1, ascending=False)
    oos_rank_of_best_is = np.array([
        oos_ranks.loc[idx, best_is_strategy[idx]] 
        for idx in common_idx
    ])
    
    median_rank = (n_strategies + 1) / 2
    pbo = np.mean(oos_rank_of_best_is > median_rank)
    
    return pbo
