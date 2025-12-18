import pandas as pd
import numpy as np

# ============ TRANSACTION COSTS & REBALANCING ============

def calculate_portfolio_with_rebalancing(
    returns_df,
    weights_target,
    rebalance_freq=None,
    rebalance_threshold=None,
    cost_bps=0,
    rf_rate=0.02
):
    """
    Calculate portfolio returns with rebalancing and transaction costs.
    
    Methodology based on:
    - DeMiguel, Garlappi, Uppal (2009) "Optimal Versus Naive Diversification"
      Review of Financial Studies
    - Kirby & Ostdiek (2012) "It's All in the Timing"
      Journal of Financial and Quantitative Analysis
    
    Transaction costs are applied as proportional costs that reduce portfolio value:
        V_t = V_{t-1} * (1 + r_gross) * (1 - TC_t)
    
    where TC_t = turnover_t * cost_rate (only on rebalancing days)
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Daily returns for each asset
    weights_target : array-like
        Target portfolio weights
    rebalance_freq : str, optional
        Pandas frequency string ('D', 'W', 'M', 'Q', 'A') for calendar rebalancing
    rebalance_threshold : float, optional
        Maximum drift allowed before triggering rebalancing (e.g., 0.05 = 5%)
    cost_bps : float
        Proportional transaction cost in basis points (one-way).
        Example: 10 bps = 0.10% cost per unit of turnover
    rf_rate : float
        Annual risk-free rate for Sharpe calculation
        
    Returns:
    --------
    dict with portfolio metrics, returns series, and rebalancing details
    """
    weights_target = np.array(weights_target)
    weights_current = weights_target.copy()
    n_assets = len(weights_target)
    cost_rate = cost_bps / 10000  # Convert bps to decimal
    
    # Track portfolio value explicitly (not just returns)
    # This is critical for correct compounding of costs
    portfolio_value = 1.0  # Start with $1
    portfolio_values = []
    
    # For gross comparison (same rebalancing, no costs)
    portfolio_value_gross = 1.0
    portfolio_values_gross = []
    weights_gross = weights_target.copy()
    
    # Tracking variables
    turnover_history = []
    rebalance_dates = []
    weights_history = []
    costs_paid = []
    
    # Build rebalance schedule if calendar-based
    if rebalance_freq:
        rebalance_schedule = set(returns_df.resample(rebalance_freq).last().index)
    else:
        rebalance_schedule = set()
    
    for date, daily_returns in returns_df.iterrows():
        
        # ===== NET PORTFOLIO (with costs) =====
        
        # 1. Calculate gross return for the day
        port_return_gross = np.dot(weights_current, daily_returns.values)
        
        # 2. Update portfolio value with gross return (before costs)
        portfolio_value_pre_cost = portfolio_value * (1 + port_return_gross)
        
        # 3. Update weights based on drift (before any rebalancing)
        weights_drifted = weights_current * (1 + daily_returns.values)
        total_weight = weights_drifted.sum()
        if total_weight > 0:
            weights_drifted = weights_drifted / total_weight
        else:
            weights_drifted = weights_target.copy()
        
        # 4. Calculate current drift
        current_drift = np.max(np.abs(weights_drifted - weights_target))
        
        # 5. Check if rebalancing needed
        needs_rebalance = False
        
        if rebalance_threshold is not None:
            if current_drift > rebalance_threshold:
                needs_rebalance = True
        elif rebalance_freq and date in rebalance_schedule:
            needs_rebalance = True
        
        # 6. Execute rebalancing with transaction costs
        if needs_rebalance:
            # One-way turnover: sum of absolute weight changes
            # This represents the total fraction of portfolio traded
            # Note: we do NOT divide by 2 (one-way convention per DeMiguel et al.)
            turnover = np.sum(np.abs(weights_drifted - weights_target))
            
            # Transaction cost reduces portfolio value multiplicatively
            # This correctly captures the compounding effect of costs
            transaction_cost = turnover * cost_rate
            portfolio_value_post_cost = portfolio_value_pre_cost * (1 - transaction_cost)
            
            # Reset weights to target
            weights_current = weights_target.copy()
            
            # Track
            turnover_history.append(turnover)
            rebalance_dates.append(date)
            costs_paid.append(transaction_cost * portfolio_value_pre_cost)
        else:
            portfolio_value_post_cost = portfolio_value_pre_cost
            weights_current = weights_drifted.copy()
        
        # Update portfolio value
        portfolio_value = portfolio_value_post_cost
        portfolio_values.append(portfolio_value)
        
        # ===== GROSS PORTFOLIO (no costs, same rebalancing schedule) =====
        
        port_return_gross_track = np.dot(weights_gross, daily_returns.values)
        portfolio_value_gross *= (1 + port_return_gross_track)
        portfolio_values_gross.append(portfolio_value_gross)
        
        # Drift weights for gross tracking
        weights_gross_drifted = weights_gross * (1 + daily_returns.values)
        weights_gross_drifted = weights_gross_drifted / weights_gross_drifted.sum()
        
        # Rebalance gross at same times (but no cost)
        if needs_rebalance:
            weights_gross = weights_target.copy()
        else:
            weights_gross = weights_gross_drifted.copy()
        
        # Track weights
        weights_history.append({
            'date': date,
            'weights': weights_current.copy(),
            'max_drift': current_drift,
            'rebalanced': needs_rebalance
        })
    
    # Convert to series
    portfolio_values_series = pd.Series(portfolio_values, index=returns_df.index)
    portfolio_values_gross_series = pd.Series(portfolio_values_gross, index=returns_df.index)
    
    # Calculate returns from values
    returns_net = portfolio_values_series.pct_change().dropna()
    returns_gross = portfolio_values_gross_series.pct_change().dropna()
    
    # ===== CALCULATE METRICS =====
    
    # Time period
    n_years = len(returns_df) / 252
    
    # Total returns
    total_return_net = portfolio_values_series.iloc[-1] - 1
    total_return_gross = portfolio_values_gross_series.iloc[-1] - 1
    
    # Annualized returns (geometric - correct for compounding)
    ann_return_net = (1 + total_return_net) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_return_gross = (1 + total_return_gross) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Volatility
    ann_vol_net = returns_net.std() * np.sqrt(252)
    ann_vol_gross = returns_gross.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_net = (ann_return_net - rf_rate) / ann_vol_net if ann_vol_net > 0 else 0
    sharpe_gross = (ann_return_gross - rf_rate) / ann_vol_gross if ann_vol_gross > 0 else 0
    
    # Sortino Ratio (net)
    downside = returns_net[returns_net < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol_net
    sortino_net = (ann_return_net - rf_rate) / downside_std if downside_std > 0 else 0
    
    # Max Drawdown (net)
    cumulative_net = portfolio_values_series
    rolling_max = cumulative_net.expanding().max()
    drawdown = (cumulative_net - rolling_max) / rolling_max
    max_dd_net = drawdown.min()
    
    # Calmar Ratio (net)
    calmar_net = ann_return_net / abs(max_dd_net) if max_dd_net != 0 else 0
    
    # ===== COST ANALYSIS =====
    
    # Total turnover over the period
    total_turnover = sum(turnover_history)
    
    # Total costs as percentage of final value
    # This measures cumulative drag from all transactions
    if portfolio_values_gross_series.iloc[-1] > 0:
        total_costs_pct = (1 - portfolio_values_series.iloc[-1] / portfolio_values_gross_series.iloc[-1]) * 100
    else:
        total_costs_pct = 0
    
    # Number of rebalancing events
    n_rebalances = len(rebalance_dates)
    
    # Average turnover per rebalancing
    avg_turnover = total_turnover / n_rebalances if n_rebalances > 0 else 0
    
    # Annual cost drag (difference in annualized returns)
    cost_drag = ann_return_gross - ann_return_net
    
    # Annualized turnover
    annual_turnover = total_turnover / n_years if n_years > 0 else 0
    
    # Create weights DataFrame for visualization
    weights_df = pd.DataFrame([
        {**{'date': w['date'], 'max_drift': w['max_drift'], 'rebalanced': w['rebalanced']}, 
         **{f'weight_{i}': w['weights'][i] for i in range(n_assets)}}
        for w in weights_history
    ])
    weights_df.set_index('date', inplace=True)
    
    return {
        # Returns series
        'returns_gross': returns_gross,
        'returns_net': returns_net,
        'cumulative_gross': portfolio_values_gross_series,
        'cumulative_net': portfolio_values_series,
        
        # Gross metrics
        'ann_return_gross': ann_return_gross,
        'ann_vol_gross': ann_vol_gross,
        'sharpe_gross': sharpe_gross,
        
        # Net metrics
        'ann_return_net': ann_return_net,
        'ann_vol_net': ann_vol_net,
        'sharpe_net': sharpe_net,
        'sortino_net': sortino_net,
        'max_dd_net': max_dd_net,
        'calmar_net': calmar_net,
        
        # Cost analysis
        'total_turnover': total_turnover,
        'annual_turnover': annual_turnover,
        'total_costs_pct': total_costs_pct,
        'n_rebalances': n_rebalances,
        'avg_turnover': avg_turnover,
        'cost_drag': cost_drag,
        'rebalance_dates': rebalance_dates,
        
        # Weight tracking
        'weights_history': weights_df,
        'weights_target': weights_target
    }

def calculate_all_portfolios_with_costs(analyzer, rebalance_freq, rebalance_threshold, cost_bps, rf_rate):
    """
    Recalculate all portfolios with transaction costs and rebalancing.
    
    Returns dict with enhanced metrics for each portfolio.
    """
    results = {}
    
    for key, portfolio in analyzer.portfolios.items():
        enhanced = calculate_portfolio_with_rebalancing(
            returns_df=analyzer.returns,
            weights_target=portfolio['weights'],
            rebalance_freq=rebalance_freq,
            rebalance_threshold=rebalance_threshold,
            cost_bps=cost_bps,
            rf_rate=rf_rate
        )
        
        # Merge with original portfolio data
        results[key] = {
            'name': portfolio['name'],
            'weights': portfolio['weights'],  # Original weights array
            **enhanced  # Enhanced metrics including weights_target, weights_history, etc.
        }
    
    return results

