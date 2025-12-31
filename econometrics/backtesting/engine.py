"""
PCA-OU Mean Reversion Backtesting Engine

This module implements a walk-forward backtesting framework for evaluating
PCA-based mean reversion strategies with O-U modeling and HMM regime detection.

References:
    - Avellaneda, M. & Lee, J.H. (2010). "Statistical Arbitrage in the US Equities Market"
    - López de Prado, M. (2018). "Advances in Financial Machine Learning"
    - Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio"

Author: Portfolio Analyzer
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import warnings

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Regime(Enum):
    """Market regime enumeration."""
    CALM = 0
    NORMAL = 1
    PANIC = 2


@dataclass
class BacktestConfig:
    """
    Configuration parameters for the backtester.
    
    Attributes:
        estimation_window: Days of data for PCA/OU parameter estimation
        reestimation_freq: How often to re-estimate PCA/OU (in trading days)
        rebalance_freq: How often to rebalance portfolio (in trading days)
        n_components: Number of principal components to use
        z_entry: Z-score threshold to enter a position
        z_exit: Z-score threshold to exit a position
        transaction_cost_bps: Transaction cost in basis points (round-trip)
        min_weight: Minimum weight per asset (allows controlled shorting)
        max_weight: Maximum weight per asset (prevents concentration)
        use_hmm: Whether to use HMM regime filtering
        n_regimes: Number of HMM regimes (2 or 3)
        regime_panic_factor: Position reduction factor in Panic regime
        regime_normal_factor: Position reduction factor in Normal regime
        max_tilt: Maximum deviation from equal weight per asset
    """
    estimation_window: int = 252
    reestimation_freq: int = 21  # Monthly re-estimation
    rebalance_freq: int = 5  # Weekly rebalancing
    n_components: int = 5
    z_entry: float = 2.0
    z_exit: float = 0.5
    transaction_cost_bps: float = 10.0
    min_weight: float = 0.05
    max_weight: float = 0.40
    use_hmm: bool = True
    n_regimes: int = 2
    regime_panic_factor: float = 0.3
    regime_normal_factor: float = 0.7
    max_tilt: float = 0.15  # Max 15% deviation from equal weight


@dataclass
class TradeRecord:
    """Record of a single rebalancing event."""
    date: pd.Timestamp
    weights_before: np.ndarray
    weights_after: np.ndarray
    turnover: float
    transaction_cost: float
    regime: str
    active_signals: Dict[int, float]  # PC index -> Z-score
    

@dataclass
class BacktestResults:
    """
    Complete results from a backtest run.
    
    Attributes:
        config: Configuration used for this backtest
        dates: Index of all trading dates
        benchmark_returns: Daily returns of benchmark (equal-weight buy&hold)
        strategy_returns: Daily returns of the strategy
        benchmark_equity: Cumulative equity curve for benchmark
        strategy_equity: Cumulative equity curve for strategy
        weights_history: DataFrame of portfolio weights over time
        trades: List of all trade records
        regimes: Series of detected regimes over time
        signals_history: DataFrame of Z-scores over time
        metrics: Dictionary of performance metrics
    """
    config: BacktestConfig
    dates: pd.DatetimeIndex
    benchmark_returns: pd.Series
    strategy_returns: pd.Series
    benchmark_equity: pd.Series
    strategy_equity: pd.Series
    weights_history: pd.DataFrame
    trades: List[TradeRecord]
    regimes: pd.Series
    signals_history: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)


class PCAOUBacktester:
    """
    Walk-forward backtester for PCA-OU mean reversion strategy.
    
    The strategy:
    1. Starts with equal-weight portfolio
    2. Uses PCA to identify factor structure
    3. Fits O-U models to cumulative factor scores
    4. Tilts weights based on Z-score signals
    5. Optionally reduces tilt intensity in Panic regime (HMM)
    
    Example:
        >>> prices = pd.DataFrame(...)  # Price data
        >>> config = BacktestConfig(estimation_window=252, z_entry=2.0)
        >>> backtester = PCAOUBacktester(prices, config)
        >>> results = backtester.run()
        >>> print(f"Sharpe: {results.metrics['sharpe_ratio']:.2f}")
    """
    
    def __init__(
        self, 
        prices: pd.DataFrame, 
        config: Optional[BacktestConfig] = None,
        asset_names: Optional[List[str]] = None
    ):
        """
        Initialize the backtester.
        
        Args:
            prices: DataFrame of asset prices (index=dates, columns=assets)
            config: Backtest configuration (uses defaults if None)
            asset_names: Optional display names for assets
        """
        self.prices = prices.copy()
        self.config = config or BacktestConfig()
        self.asset_names = asset_names or list(prices.columns)
        self.n_assets = len(prices.columns)
        
        # Compute log returns
        self.returns = np.log(prices / prices.shift(1)).dropna()
        
        # Validate configuration
        self._validate_config()
        
        # State variables (populated during backtest)
        self._current_pca = None
        self._current_loadings = None
        self._current_ou_params = None
        self._current_hmm = None
        self._current_scaler = None
        
    def _validate_config(self):
        """Validate configuration parameters."""
        cfg = self.config
        
        if cfg.estimation_window >= len(self.returns):
            raise ValueError(
                f"Estimation window ({cfg.estimation_window}) must be smaller "
                f"than available data ({len(self.returns)} days)"
            )
        
        if cfg.n_components > self.n_assets:
            self.config.n_components = self.n_assets
            warnings.warn(
                f"n_components reduced to {self.n_assets} (number of assets)"
            )
        
        if cfg.min_weight < 0:
            raise ValueError("min_weight cannot be negative")
        
        if cfg.max_weight > 1:
            raise ValueError("max_weight cannot exceed 1")
        
        if cfg.min_weight * self.n_assets > 1:
            raise ValueError(
                f"min_weight ({cfg.min_weight}) too high for {self.n_assets} assets"
            )
    
    def run(self) -> BacktestResults:
        """
        Execute the full walk-forward backtest.
        
        Returns:
            BacktestResults object containing all results and metrics
        """
        cfg = self.config
        returns = self.returns
        
        # Determine backtest start (need estimation_window of data first)
        start_idx = cfg.estimation_window
        backtest_dates = returns.index[start_idx:]
        
        # Initialize tracking arrays
        n_days = len(backtest_dates)
        
        strategy_returns = np.zeros(n_days)
        benchmark_returns = np.zeros(n_days)
        weights_history = np.zeros((n_days, self.n_assets))
        signals_history = np.zeros((n_days, cfg.n_components))
        regimes = []
        trades = []
        
        # Initial weights: equal weight
        current_weights = np.ones(self.n_assets) / self.n_assets
        
        # Track when to re-estimate and rebalance
        days_since_estimation = cfg.reestimation_freq  # Force estimation on first day
        days_since_rebalance = cfg.rebalance_freq  # Force rebalance on first day
        
        # Main backtest loop
        for i, date in enumerate(backtest_dates):
            
            # Get index in original returns DataFrame
            t = start_idx + i
            
            # ─────────────────────────────────────────────────────────
            # STEP 1: RE-ESTIMATE MODELS (if needed)
            # ─────────────────────────────────────────────────────────
            if days_since_estimation >= cfg.reestimation_freq:
                estimation_returns = returns.iloc[t - cfg.estimation_window:t]
                self._estimate_models(estimation_returns)
                days_since_estimation = 0
            else:
                days_since_estimation += 1
            
            # ─────────────────────────────────────────────────────────
            # STEP 2: COMPUTE CURRENT SIGNALS
            # ─────────────────────────────────────────────────────────
            # Use recent data for Z-score calculation
            recent_returns = returns.iloc[max(0, t - cfg.estimation_window):t]
            z_scores, regime = self._compute_signals(recent_returns)
            
            signals_history[i, :] = z_scores
            regimes.append(regime)
            
            # ─────────────────────────────────────────────────────────
            # STEP 3: REBALANCE (if needed)
            # ─────────────────────────────────────────────────────────
            if days_since_rebalance >= cfg.rebalance_freq:
                target_weights, active_signals = self._compute_target_weights(
                    z_scores, regime
                )
                
                # Calculate turnover and costs
                turnover = np.sum(np.abs(target_weights - current_weights))
                tx_cost = turnover * (cfg.transaction_cost_bps / 10000)
                
                # Record trade
                if turnover > 0.001:  # Only record meaningful trades
                    trades.append(TradeRecord(
                        date=date,
                        weights_before=current_weights.copy(),
                        weights_after=target_weights.copy(),
                        turnover=turnover,
                        transaction_cost=tx_cost,
                        regime=regime,
                        active_signals=active_signals
                    ))
                
                current_weights = target_weights
                days_since_rebalance = 0
            else:
                tx_cost = 0.0
                days_since_rebalance += 1
            
            # ─────────────────────────────────────────────────────────
            # STEP 4: COMPUTE RETURNS
            # ─────────────────────────────────────────────────────────
            # Get today's actual returns
            daily_returns = returns.iloc[t].values
            
            # Strategy return (weighted)
            strategy_returns[i] = np.dot(current_weights, daily_returns) - tx_cost
            
            # Benchmark return (equal weight, no rebalancing cost)
            benchmark_returns[i] = np.mean(daily_returns)
            
            # Update weights for drift (prices changed)
            # w_new = w_old * (1 + r) / sum(w_old * (1 + r))
            gross_returns = np.exp(daily_returns)
            drifted_weights = current_weights * gross_returns
            current_weights = drifted_weights / np.sum(drifted_weights)
            
            # Store weights
            weights_history[i, :] = current_weights
        
        # ─────────────────────────────────────────────────────────
        # BUILD RESULTS
        # ─────────────────────────────────────────────────────────
        strategy_returns_series = pd.Series(
            strategy_returns, index=backtest_dates, name='strategy'
        )
        benchmark_returns_series = pd.Series(
            benchmark_returns, index=backtest_dates, name='benchmark'
        )
        
        # Compute equity curves (cumulative returns)
        strategy_equity = (1 + strategy_returns_series).cumprod()
        benchmark_equity = (1 + benchmark_returns_series).cumprod()
        
        # Build DataFrames
        weights_df = pd.DataFrame(
            weights_history, 
            index=backtest_dates, 
            columns=self.prices.columns
        )
        signals_df = pd.DataFrame(
            signals_history,
            index=backtest_dates,
            columns=[f'PC{i+1}' for i in range(cfg.n_components)]
        )
        regimes_series = pd.Series(regimes, index=backtest_dates, name='regime')
        
        # Create results object
        results = BacktestResults(
            config=cfg,
            dates=backtest_dates,
            benchmark_returns=benchmark_returns_series,
            strategy_returns=strategy_returns_series,
            benchmark_equity=benchmark_equity,
            strategy_equity=strategy_equity,
            weights_history=weights_df,
            trades=trades,
            regimes=regimes_series,
            signals_history=signals_df
        )
        
        # Compute performance metrics
        results.metrics = self._compute_metrics(results)
        
        return results
    
    def _estimate_models(self, returns: pd.DataFrame):
        """
        Estimate PCA and O-U models on the given data window.
        
        Args:
            returns: DataFrame of returns for estimation
        """
        cfg = self.config
        returns_matrix = returns.values
        
        # ─────────────────────────────────────────────────────────
        # PCA
        # ─────────────────────────────────────────────────────────
        self._current_scaler = StandardScaler()
        returns_standardized = self._current_scaler.fit_transform(returns_matrix)
        
        self._current_pca = PCA(n_components=cfg.n_components)
        factor_scores = self._current_pca.fit_transform(returns_standardized)
        self._current_loadings = self._current_pca.components_.T  # N x K
        
        # ─────────────────────────────────────────────────────────
        # O-U Parameters for each factor
        # ─────────────────────────────────────────────────────────
        cumulative_scores = np.cumsum(factor_scores, axis=0)
        
        self._current_ou_params = []
        
        for k in range(cfg.n_components):
            f_k = cumulative_scores[:, k]
            
            # AR(1) regression: f_t = alpha + beta * f_{t-1} + eps
            f_lag = f_k[:-1]
            f_current = f_k[1:]
            
            # OLS
            X = np.column_stack([np.ones(len(f_lag)), f_lag])
            try:
                beta_ols = np.linalg.lstsq(X, f_current, rcond=None)[0]
                alpha, beta = beta_ols[0], beta_ols[1]
            except:
                alpha, beta = 0, 0.99
            
            # O-U parameters
            if 0 < beta < 1:
                kappa = -np.log(beta)
                half_life = np.log(2) / kappa if kappa > 0 else np.inf
                mu = alpha / (1 - beta)
            else:
                kappa = 0
                half_life = np.inf
                mu = 0
            
            # Residual volatility
            residuals = f_current - (alpha + beta * f_lag)
            sigma = np.std(residuals) if len(residuals) > 0 else 1.0
            
            self._current_ou_params.append({
                'kappa': kappa,
                'half_life': half_life,
                'mu': mu,
                'sigma': sigma,
                'beta': beta,
                'is_mean_reverting': 0 < beta < 0.99 and half_life < 60
            })
        
        # ─────────────────────────────────────────────────────────
        # HMM (optional)
        # ─────────────────────────────────────────────────────────
        if cfg.use_hmm:
            try:
                from hmmlearn.hmm import GaussianHMM
                
                portfolio_returns = np.mean(returns_matrix, axis=1).reshape(-1, 1)
                
                self._current_hmm = GaussianHMM(
                    n_components=cfg.n_regimes,
                    covariance_type='full',
                    n_iter=100,
                    random_state=42
                )
                self._current_hmm.fit(portfolio_returns)
                
                # Determine which state is "Panic" (highest volatility)
                state_vols = []
                hidden_states = self._current_hmm.predict(portfolio_returns)
                for s in range(cfg.n_regimes):
                    state_returns = portfolio_returns[hidden_states == s]
                    state_vols.append(np.std(state_returns) if len(state_returns) > 0 else 0)
                
                self._hmm_vol_order = np.argsort(state_vols)  # Low to high volatility
                
            except ImportError:
                self._current_hmm = None
            except Exception:
                self._current_hmm = None
    
    def _compute_signals(
        self, 
        recent_returns: pd.DataFrame
    ) -> Tuple[np.ndarray, str]:
        """
        Compute current Z-scores and regime.
        
        Args:
            recent_returns: Recent returns for signal computation
            
        Returns:
            Tuple of (z_scores array, regime string)
        """
        cfg = self.config
        
        # Transform returns through current PCA
        returns_standardized = self._current_scaler.transform(recent_returns.values)
        factor_scores = self._current_pca.transform(returns_standardized)
        cumulative_scores = np.cumsum(factor_scores, axis=0)
        
        # Compute Z-scores (rolling, using last portion of data)
        trading_window = min(60, len(cumulative_scores))
        z_scores = np.zeros(cfg.n_components)
        
        for k in range(cfg.n_components):
            f_k = cumulative_scores[:, k]
            
            if len(f_k) >= trading_window:
                recent = f_k[-trading_window:]
                current_value = f_k[-1]
                rolling_mean = np.mean(recent)
                rolling_std = np.std(recent)
                
                if rolling_std > 1e-8:
                    z_scores[k] = (current_value - rolling_mean) / rolling_std
                else:
                    z_scores[k] = 0
            else:
                z_scores[k] = 0
        
        # Determine regime
        regime = 'Calm'
        
        if cfg.use_hmm and self._current_hmm is not None:
            try:
                portfolio_returns = np.mean(recent_returns.values, axis=1).reshape(-1, 1)
                current_state = self._current_hmm.predict(portfolio_returns)[-1]
                
                # Map to ordered regime
                ordered_state = np.where(self._hmm_vol_order == current_state)[0][0]
                
                if cfg.n_regimes == 2:
                    regime = 'Calm' if ordered_state == 0 else 'Panic'
                else:
                    regime = ['Calm', 'Normal', 'Panic'][ordered_state]
                    
            except Exception:
                regime = 'Calm'
        
        return z_scores, regime
    
    def _compute_target_weights(
        self, 
        z_scores: np.ndarray, 
        regime: str
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Compute target portfolio weights based on signals and regime.
        
        The approach:
        1. Start with equal weights
        2. For each mean-reverting factor with |Z| > entry threshold:
           - Compute tilt based on Z-score and loadings
           - Direction: opposite to Z-score (mean reversion)
        3. Apply regime adjustment (reduce tilt in Panic)
        4. Apply constraints (min/max weight)
        
        Args:
            z_scores: Current Z-scores for each factor
            regime: Current market regime
            
        Returns:
            Tuple of (target weights array, active signals dict)
        """
        cfg = self.config
        n_assets = self.n_assets
        
        # Start with equal weight
        equal_weight = 1.0 / n_assets
        weights = np.ones(n_assets) * equal_weight
        
        # Track active signals
        active_signals = {}
        
        # Compute tilts from each factor (skip PC1 = market factor)
        total_tilt = np.zeros(n_assets)
        
        for k in range(1, cfg.n_components):  # Skip PC1
            z = z_scores[k]
            ou_params = self._current_ou_params[k]
            
            # Only trade mean-reverting factors with significant Z-score
            if not ou_params['is_mean_reverting']:
                continue
            
            # Check entry/exit thresholds
            if abs(z) < cfg.z_entry:
                continue
            
            active_signals[k] = z
            
            # Tilt direction: opposite to Z-score (mean reversion)
            # If Z > 0, factor is high, expect it to fall
            # → Short positive loadings, Long negative loadings
            # Tilt = -sign(Z) * |Z-scaled| * loadings
            
            # Scale Z-score to [0, 1] range for tilt intensity
            z_intensity = min((abs(z) - cfg.z_entry) / cfg.z_entry, 1.0)
            
            # Get loadings for this factor
            loadings_k = self._current_loadings[:, k]
            
            # Compute tilt: opposite direction to Z * loadings
            tilt_k = -np.sign(z) * z_intensity * loadings_k * cfg.max_tilt
            
            total_tilt += tilt_k
        
        # Apply regime adjustment
        if regime == 'Panic':
            total_tilt *= cfg.regime_panic_factor
        elif regime == 'Normal':
            total_tilt *= cfg.regime_normal_factor
        # Calm: full tilt (factor = 1.0)
        
        # Apply tilt to weights
        weights = weights + total_tilt
        
        # Apply constraints
        weights = np.clip(weights, cfg.min_weight, cfg.max_weight)
        
        # Renormalize to sum to 1
        weights = weights / np.sum(weights)
        
        return weights, active_signals
    
    def _compute_metrics(self, results: BacktestResults) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.
        
        Args:
            results: BacktestResults object
            
        Returns:
            Dictionary of performance metrics
        """
        strat_ret = results.strategy_returns
        bench_ret = results.benchmark_returns
        
        # Basic returns
        total_return_strategy = results.strategy_equity.iloc[-1] - 1
        total_return_benchmark = results.benchmark_equity.iloc[-1] - 1
        
        # Annualized metrics (assuming 252 trading days)
        n_years = len(strat_ret) / 252
        
        ann_return_strategy = (1 + total_return_strategy) ** (1 / n_years) - 1
        ann_return_benchmark = (1 + total_return_benchmark) ** (1 / n_years) - 1
        
        ann_vol_strategy = strat_ret.std() * np.sqrt(252)
        ann_vol_benchmark = bench_ret.std() * np.sqrt(252)
        
        # Risk-adjusted returns
        rf_rate = 0.0  # Assume 0 for simplicity
        
        sharpe_strategy = (ann_return_strategy - rf_rate) / ann_vol_strategy if ann_vol_strategy > 0 else 0
        sharpe_benchmark = (ann_return_benchmark - rf_rate) / ann_vol_benchmark if ann_vol_benchmark > 0 else 0
        
        # Sortino (downside deviation)
        downside_returns_strat = strat_ret[strat_ret < 0]
        downside_vol_strat = downside_returns_strat.std() * np.sqrt(252) if len(downside_returns_strat) > 0 else ann_vol_strategy
        sortino_strategy = (ann_return_strategy - rf_rate) / downside_vol_strat if downside_vol_strat > 0 else 0
        
        # Maximum Drawdown
        def max_drawdown(equity_curve):
            rolling_max = equity_curve.expanding().max()
            drawdowns = equity_curve / rolling_max - 1
            return drawdowns.min()
        
        max_dd_strategy = max_drawdown(results.strategy_equity)
        max_dd_benchmark = max_drawdown(results.benchmark_equity)
        
        # Calmar Ratio
        calmar_strategy = ann_return_strategy / abs(max_dd_strategy) if max_dd_strategy != 0 else 0
        calmar_benchmark = ann_return_benchmark / abs(max_dd_benchmark) if max_dd_benchmark != 0 else 0
        
        # Alpha and Beta (vs benchmark)
        excess_returns = strat_ret - bench_ret
        
        # Regression: R_strat = alpha + beta * R_bench + eps
        X = np.column_stack([np.ones(len(bench_ret)), bench_ret.values])
        y = strat_ret.values
        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha_daily = coef[0]
            beta = coef[1]
            alpha_annual = alpha_daily * 252
        except:
            alpha_annual = 0
            beta = 1
        
        # Information Ratio
        tracking_error = excess_returns.std() * np.sqrt(252)
        info_ratio = (ann_return_strategy - ann_return_benchmark) / tracking_error if tracking_error > 0 else 0
        
        # VaR and CVaR (95%)
        var_95_strategy = np.percentile(strat_ret, 5)
        cvar_95_strategy = strat_ret[strat_ret <= var_95_strategy].mean()
        
        var_95_benchmark = np.percentile(bench_ret, 5)
        cvar_95_benchmark = bench_ret[bench_ret <= var_95_benchmark].mean()
        
        # Trade statistics
        n_trades = len(results.trades)
        total_turnover = sum(t.turnover for t in results.trades)
        total_tx_costs = sum(t.transaction_cost for t in results.trades)
        avg_turnover = total_turnover / n_trades if n_trades > 0 else 0
        
        # Win rate (trades that improved vs benchmark)
        wins = 0
        for trade in results.trades:
            trade_date = trade.date
            if trade_date in results.strategy_returns.index:
                if results.strategy_returns[trade_date] > results.benchmark_returns[trade_date]:
                    wins += 1
        win_rate = wins / n_trades if n_trades > 0 else 0.5
        
        # Regime breakdown
        regime_returns = {}
        for regime in ['Calm', 'Normal', 'Panic']:
            regime_mask = results.regimes == regime
            if regime_mask.any():
                regime_ret = strat_ret[regime_mask]
                regime_returns[f'return_{regime.lower()}'] = regime_ret.mean() * 252
                regime_returns[f'vol_{regime.lower()}'] = regime_ret.std() * np.sqrt(252)
        
        metrics = {
            # Returns
            'total_return_strategy': total_return_strategy,
            'total_return_benchmark': total_return_benchmark,
            'excess_return': total_return_strategy - total_return_benchmark,
            'ann_return_strategy': ann_return_strategy,
            'ann_return_benchmark': ann_return_benchmark,
            
            # Volatility
            'ann_vol_strategy': ann_vol_strategy,
            'ann_vol_benchmark': ann_vol_benchmark,
            
            # Risk-adjusted
            'sharpe_strategy': sharpe_strategy,
            'sharpe_benchmark': sharpe_benchmark,
            'sortino_strategy': sortino_strategy,
            'calmar_strategy': calmar_strategy,
            'calmar_benchmark': calmar_benchmark,
            
            # Alpha/Beta
            'alpha': alpha_annual,
            'beta': beta,
            'information_ratio': info_ratio,
            
            # Drawdown
            'max_drawdown_strategy': max_dd_strategy,
            'max_drawdown_benchmark': max_dd_benchmark,
            
            # VaR/CVaR
            'var_95_strategy': var_95_strategy,
            'cvar_95_strategy': cvar_95_strategy,
            'var_95_benchmark': var_95_benchmark,
            'cvar_95_benchmark': cvar_95_benchmark,
            
            # Trade stats
            'n_trades': n_trades,
            'total_turnover': total_turnover,
            'total_tx_costs': total_tx_costs,
            'avg_turnover': avg_turnover,
            'win_rate': win_rate,
            
            # Other
            'n_days': len(strat_ret),
            'n_years': n_years,
            
            **regime_returns
        }
        
        return metrics


def run_backtest(
    prices: pd.DataFrame,
    estimation_window: int = 252,
    rebalance_freq: int = 5,
    n_components: int = 5,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    transaction_cost_bps: float = 10.0,
    use_hmm: bool = True,
    **kwargs
) -> BacktestResults:
    """
    Convenience function to run a backtest with specified parameters.
    
    Args:
        prices: DataFrame of asset prices
        estimation_window: Days for model estimation
        rebalance_freq: Days between rebalancing
        n_components: Number of PCA components
        z_entry: Z-score entry threshold
        z_exit: Z-score exit threshold
        transaction_cost_bps: Transaction costs in basis points
        use_hmm: Whether to use HMM regime filtering
        **kwargs: Additional config parameters
        
    Returns:
        BacktestResults object
    """
    config = BacktestConfig(
        estimation_window=estimation_window,
        rebalance_freq=rebalance_freq,
        n_components=n_components,
        z_entry=z_entry,
        z_exit=z_exit,
        transaction_cost_bps=transaction_cost_bps,
        use_hmm=use_hmm,
        **kwargs
    )
    
    backtester = PCAOUBacktester(prices, config)
    return backtester.run()


