import numpy as np
import pandas as pd

def calculate_portfolio_metrics(returns, rf_rate=0.0):
    """
    Calculate comprehensive risk-adjusted performance metrics.
    
    Returns dictionary with:
    - sharpe: Standard Sharpe Ratio
    - sortino: Sortino Ratio (downside deviation)
    - calmar: Calmar Ratio (return / max drawdown)
    - max_drawdown: Maximum peak-to-trough decline
    - cvar_95: Conditional Value at Risk (5% worst cases)
    - win_rate: Percentage of positive return periods
    """
    if len(returns) == 0:
        return {
            'sharpe': np.nan,
            'sortino': np.nan,
            'calmar': np.nan,
            'max_drawdown': np.nan,
            'cvar_95': np.nan,
            'win_rate': np.nan
        }
    
    returns_excess = returns - rf_rate
    
    # 1. SHARPE RATIO (volatility-adjusted)
    # Formula: (Annualized Return - Risk Free Rate) / Annualized Volatility
    # Note: rf_rate is already annualized, so we don't multiply by 252
    if returns.std() > 0:
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (annual_return - rf_rate) / annual_vol
    else:
        sharpe = np.nan
    
    # 2. SORTINO RATIO (downside risk only)
    # Formula: (Annualized Return - Risk Free Rate) / Annualized Downside Deviation
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        annual_return = returns.mean() * 252
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std > 0:
            sortino = (annual_return - rf_rate) / downside_std
        else:
            sortino = np.nan
    else:
        # No negative returns - strategy never loses
        sortino = np.inf if returns.mean() > 0 else np.nan
    
    # 3. MAXIMUM DRAWDOWN
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())
    
    # 4. CALMAR RATIO (return / max drawdown)
    annual_return = returns.mean() * 252
    if max_dd > 0:
        calmar = annual_return / max_dd
    else:
        calmar = np.inf if annual_return > 0 else np.nan
    
    # 5. CONDITIONAL VALUE AT RISK (CVaR at 95%)
    if len(returns) > 0:
        cvar_95 = returns.quantile(0.05)
    else:
        cvar_95 = np.nan
    
    # 6. WIN RATE
    win_rate = (returns > 0).mean()
    
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_dd,
        'cvar_95': cvar_95,
        'win_rate': win_rate
    }
