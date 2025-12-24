"""
Portfolio API Routes
These endpoints handle portfolio analysis operations.
"""

from fastapi import APIRouter, HTTPException
from api.schemas.portfolio import (
    MarketDataRequest, MarketDataResponse,
    PortfolioAnalysisRequest, PortfolioAnalysisResponse, PortfolioMetrics,
    OverviewDataResponse, AssetStats, AssetTimeSeries, AssetPricePoint,
    PerformanceDataResponse, PortfolioTimeSeries, RollingMetrics, DrawdownStats,
    FrontierDataResponse, RandomPortfolio, StrategyPoint, FrontierPoint
)

# Import your existing code!
import sys
import os

# Add parent directory to path so we can import portfolio_analyzer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from portfolio_analyzer import AdvancedPortfolioAnalyzer

# Create router
router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


@router.post("/market-data", response_model=MarketDataResponse)
async def get_market_data(request: MarketDataRequest):
    """
    Download market data for specified symbols and date range.
    This endpoint validates that data can be retrieved for the given parameters.
    """
    try:
        # Use your existing AdvancedPortfolioAnalyzer class!
        analyzer = AdvancedPortfolioAnalyzer(
            symbols=request.symbols,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat()
        )
        
        # Download data
        analyzer.download_data()
        
        # Check if data was downloaded successfully
        if analyzer.data is None or len(analyzer.data) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Failed to download market data. Check symbols and date range."
            )
        
        # Calculate returns
        analyzer.calculate_returns()
        
        return MarketDataResponse(
            success=True,
            symbols=analyzer.symbols,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            trading_days=len(analyzer.returns),
            message=f"Successfully downloaded data for {len(analyzer.symbols)} symbols"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=PortfolioAnalysisResponse)
async def analyze_portfolio(request: PortfolioAnalysisRequest):
    """
    Perform full portfolio analysis with multiple optimization strategies.
    Returns metrics for all portfolio strategies.
    """
    try:
        # Initialize analyzer with your existing class
        analyzer = AdvancedPortfolioAnalyzer(
            symbols=request.symbols,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            risk_free_rate=request.risk_free_rate
        )
        
        # Download data
        analyzer.download_data()
        
        # Check if data was downloaded successfully
        if analyzer.data is None or len(analyzer.data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to download market data"
            )
        
        # Calculate returns
        analyzer.calculate_returns()
        
        # Build all portfolios (this uses your existing optimization code!)
        analyzer.build_all_portfolios()
        
        # Prepare response
        portfolios_data = {}
        
        for key, p in analyzer.portfolios.items():
            # Convert numpy array weights to dictionary
            weights_dict = {
                symbol: float(weight) 
                for symbol, weight in zip(analyzer.symbols, p['weights'])
            }
            
            portfolios_data[key] = PortfolioMetrics(
                name=p['name'],
                annualized_return=float(p['annualized_return']),
                annualized_volatility=float(p['annualized_volatility']),
                sharpe_ratio=float(p['sharpe_ratio']),
                sortino_ratio=float(p['sortino_ratio']),
                max_drawdown=float(p['max_drawdown']),
                calmar_ratio=float(p['calmar_ratio']),
                var_95=float(p.get('var_95', 0)),
                cvar_95=float(p.get('cvar_95', 0)),
                weights=weights_dict
            )
        
        # Correlation matrix as nested dict
        corr_matrix = analyzer.returns.corr()
        corr_dict = {
            col: {row: float(corr_matrix.loc[row, col]) for row in corr_matrix.index}
            for col in corr_matrix.columns
        }
        
        return PortfolioAnalysisResponse(
            success=True,
            portfolios=portfolios_data,
            correlation_matrix=corr_dict,
            symbols=analyzer.symbols,
            period={
                "start": request.start_date.isoformat(),
                "end": request.end_date.isoformat()
            },
            trading_days=len(analyzer.returns)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/overview-data", response_model=OverviewDataResponse)
async def get_overview_data(request: PortfolioAnalysisRequest):
    """
    Get all data needed for the Overview tab:
    - Normalized price series (base 100)
    - Individual asset statistics
    - Correlation matrix
    - Top/Bottom performers
    """
    import numpy as np
    
    # Initialize analyzer
    analyzer = AdvancedPortfolioAnalyzer(
        symbols=request.symbols,
        start_date=request.start_date.isoformat(),
        end_date=request.end_date.isoformat(),
        risk_free_rate=request.risk_free_rate
    )
    
    # Download data
    analyzer.download_data()
    if analyzer.data is None or len(analyzer.data) == 0:
        raise HTTPException(status_code=400, detail="Failed to download market data")
    
    analyzer.calculate_returns()
    
    rf_rate = request.risk_free_rate
    
    # 1. Normalized prices (base 100)
    normalized = (analyzer.data / analyzer.data.iloc[0]) * 100
    price_series = []
    for ticker in analyzer.symbols:
        prices = [
            AssetPricePoint(
                date=date.strftime('%Y-%m-%d'),
                value=float(normalized[ticker].loc[date])
            )
            for date in normalized.index
        ]
        price_series.append(AssetTimeSeries(ticker=ticker, prices=prices))
    
    # 2. Individual asset statistics
    asset_stats = []
    final_values = {}
    
    for ticker in analyzer.symbols:
        asset_returns = analyzer.returns[ticker]
        
        # Annualized metrics
        ann_return = float(asset_returns.mean() * 252)
        ann_vol = float(asset_returns.std() * np.sqrt(252))
        sharpe = float((ann_return - rf_rate) / ann_vol) if ann_vol > 0 else 0
        
        # Sortino
        downside = asset_returns[asset_returns < 0]
        downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else ann_vol
        sortino = float((ann_return - rf_rate) / downside_std) if downside_std > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + asset_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = float(drawdown.min())
        
        # Calmar
        calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0
        
        # Total return for ranking
        final_values[ticker] = float((normalized[ticker].iloc[-1] / 100) - 1)
        
        asset_stats.append(AssetStats(
            ticker=ticker,
            name=ticker,  # Could be enhanced with a name lookup
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar
        ))
    
    # Sort by Sharpe ratio
    asset_stats.sort(key=lambda x: x.sharpe_ratio, reverse=True)
    
    # 3. Correlation matrix
    corr_matrix = analyzer.returns.corr()
    corr_dict = {
        col: {row: float(corr_matrix.loc[row, col]) for row in corr_matrix.index}
        for col in corr_matrix.columns
    }
    
    # 4. Top/Bottom performers
    sorted_by_return = sorted(final_values.items(), key=lambda x: x[1], reverse=True)
    top_performers = [{"ticker": t, "return": r * 100} for t, r in sorted_by_return[:3]]
    bottom_performers = [{"ticker": t, "return": r * 100} for t, r in sorted_by_return[-3:]]
    
    return OverviewDataResponse(
        success=True,
        symbols=analyzer.symbols,
        trading_days=len(analyzer.returns),
        period={"start": request.start_date.isoformat(), "end": request.end_date.isoformat()},
        price_series=price_series,
        asset_stats=asset_stats,
        correlation_matrix=corr_dict,
        top_performers=top_performers,
        bottom_performers=bottom_performers
    )

@router.post("/performance-data", response_model=PerformanceDataResponse)
async def get_performance_data(request: PortfolioAnalysisRequest, window_years: int = 3):
    """
    Get all data needed for the Performance tab:
    - Cumulative returns for each portfolio (Base 100)
    - Drawdown series
    - Rolling returns and volatility
    - Drawdown statistics
    """
    import numpy as np
    
    # Initialize analyzer
    analyzer = AdvancedPortfolioAnalyzer(
        symbols=request.symbols,
        start_date=request.start_date.isoformat(),
        end_date=request.end_date.isoformat(),
        risk_free_rate=request.risk_free_rate
    )
    
    # Download and process data
    analyzer.download_data()
    if analyzer.data is None or len(analyzer.data) == 0:
        raise HTTPException(status_code=400, detail="Failed to download market data")
    
    analyzer.calculate_returns()
    analyzer.build_all_portfolios()
    
    # Rolling window in days
    window_days = window_years * 252
    
    portfolio_series = []
    rolling_metrics_list = []
    drawdown_stats_list = []
    rolling_summary = []
    
    for key, portfolio in analyzer.portfolios.items():
        returns = portfolio['returns']
        dates = [d.strftime('%Y-%m-%d') for d in returns.index]
        
        # 1. Cumulative returns (Base 100)
        cumulative = (1 + returns).cumprod() * 100
        
        # 2. Drawdown calculation
        rolling_max = cumulative.expanding().max()
        drawdown = ((cumulative - rolling_max) / rolling_max) * 100  # In percentage
        
        # 3. Drawdown statistics
        max_dd_idx = drawdown.idxmin()
        max_dd_val = float(drawdown.min())
        
        # Find recovery
        post_dd = cumulative[cumulative.index >= max_dd_idx]
        peak_before_dd = rolling_max[max_dd_idx]
        recovered = post_dd[post_dd >= peak_before_dd]
        
        if len(recovered) > 0:
            recovery_date = recovered.index[0]
            recovery_days = (recovery_date - max_dd_idx).days
            recovery_str = f"{recovery_days} days"
        else:
            recovery_str = "Not recovered"
        
        # Add to portfolio series
        portfolio_series.append(PortfolioTimeSeries(
            key=key,
            name=portfolio['name'],
            dates=dates,
            cumulative_returns=[float(v) for v in cumulative.values],
            drawdown=[float(v) for v in drawdown.values]
        ))
        
        # Add drawdown stats
        drawdown_stats_list.append(DrawdownStats(
            key=key,
            name=portfolio['name'],
            max_drawdown=max_dd_val,
            max_drawdown_date=max_dd_idx.strftime('%Y-%m-%d'),
            recovery_time=recovery_str
        ))
        
        # 4. Rolling metrics (only if enough data)
        if len(returns) >= window_days:
            # Rolling annualized return
            roll_ret = returns.rolling(window=window_days).apply(
                lambda x: (1 + x).prod() ** (252 / len(x)) - 1 if len(x) == window_days else np.nan
            )
            
            # Rolling annualized volatility
            roll_vol = returns.rolling(window=window_days).std() * np.sqrt(252)
            
            rolling_metrics_list.append(RollingMetrics(
                key=key,
                name=portfolio['name'],
                dates=dates,
                rolling_return=[float(v) * 100 if not np.isnan(v) else None for v in roll_ret.values],
                rolling_volatility=[float(v) * 100 if not np.isnan(v) else None for v in roll_vol.values]
            ))
            
            # Rolling summary statistics
            valid_returns = roll_ret.dropna()
            valid_vol = roll_vol.dropna()
            
            if len(valid_returns) > 0:
                rolling_summary.append({
                    'key': key,
                    'name': portfolio['name'],
                    'avg_return': float(valid_returns.mean() * 100),
                    'min_return': float(valid_returns.min() * 100),
                    'max_return': float(valid_returns.max() * 100),
                    'avg_vol': float(valid_vol.mean() * 100),
                    'min_vol': float(valid_vol.min() * 100),
                    'max_vol': float(valid_vol.max() * 100)
                })
        else:
            # Not enough data for rolling analysis
            rolling_metrics_list.append(RollingMetrics(
                key=key,
                name=portfolio['name'],
                dates=dates,
                rolling_return=[None] * len(dates),
                rolling_volatility=[None] * len(dates)
            ))
    
    return PerformanceDataResponse(
        success=True,
        symbols=analyzer.symbols,
        trading_days=len(analyzer.returns),
        period={"start": request.start_date.isoformat(), "end": request.end_date.isoformat()},
        window_days=window_days,
        portfolio_series=portfolio_series,
        rolling_metrics=rolling_metrics_list,
        drawdown_stats=drawdown_stats_list,
        rolling_summary=rolling_summary
    )

@router.post("/frontier-data", response_model=FrontierDataResponse)
async def get_frontier_data(
    request: PortfolioAnalysisRequest,
    frontier_type: str = "mean_variance",
    n_portfolios: int = 5000,
    allow_short: bool = False,
    cvar_alpha: float = 0.95
):
    """
    Get all data needed for the Efficient Frontier tab:
    - Random portfolio cloud
    - Optimized strategy positions
    - Approximate efficient frontier
    - Statistics
    
    frontier_type: "mean_variance" or "mean_cvar"
    """
    import numpy as np
    
    # Initialize analyzer
    analyzer = AdvancedPortfolioAnalyzer(
        symbols=request.symbols,
        start_date=request.start_date.isoformat(),
        end_date=request.end_date.isoformat(),
        risk_free_rate=request.risk_free_rate
    )
    
    # Download and process data
    analyzer.download_data()
    if analyzer.data is None or len(analyzer.data) == 0:
        raise HTTPException(status_code=400, detail="Failed to download market data")
    
    analyzer.calculate_returns()
    analyzer.build_all_portfolios()
    
    rf_rate = request.risk_free_rate
    n_assets = len(analyzer.symbols)
    
    # Prepare return data
    returns_matrix = analyzer.returns.values
    cov_matrix = analyzer.returns.cov().values * 252
    mean_returns = analyzer.returns.mean().values * 252
    
    # Generate random portfolios
    # Use different seed based on frontier_type for variety
    seed_val = 42 if frontier_type == "mean_variance" else 43
    np.random.seed(seed_val)
    random_portfolios = []
    
    for _ in range(n_portfolios):
        # Generate weights
        if allow_short:
            w = np.random.normal(1, 0.3, n_assets)
            w = w / np.sum(w)
        else:
            w = np.random.dirichlet(np.ones(n_assets) * 2.0)
        
        # Calculate return
        port_return = np.dot(mean_returns, w)
        
        # Calculate volatility
        port_variance = np.dot(w, np.dot(cov_matrix, w))
        port_volatility = np.sqrt(port_variance)
        
        # Calculate risk based on frontier type
        if frontier_type == "mean_variance":
            port_risk = port_volatility
        else:
            # CVaR calculation
            daily_returns = returns_matrix @ w
            var_threshold = np.percentile(daily_returns, 100 * (1 - cvar_alpha))
            tail_returns = daily_returns[daily_returns <= var_threshold]
            if len(tail_returns) > 0:
                port_risk = -tail_returns.mean()
            else:
                port_risk = -var_threshold
        
        # Sharpe ratio
        port_sharpe = (port_return - rf_rate) / port_volatility if port_volatility > 0 else 0
        
        random_portfolios.append(RandomPortfolio(
            return_pct=float(port_return * 100),
            risk_pct=float(port_risk * 100),
            sharpe=float(port_sharpe),
            weights=[float(x) for x in w]
        ))
    
    # Get optimized strategy points
    strategy_points = []
    for key, portfolio in analyzer.portfolios.items():
        strat_return = portfolio['annualized_return']
        strat_volatility = portfolio['annualized_volatility']
        strat_sharpe = portfolio['sharpe_ratio']
        
        # Calculate risk based on frontier type
        if frontier_type == "mean_variance":
            strat_risk = strat_volatility
        else:
            # CVaR for this portfolio
            port_returns = portfolio['returns'].values
            var_threshold = np.percentile(port_returns, 100 * (1 - cvar_alpha))
            tail = port_returns[port_returns <= var_threshold]
            if len(tail) > 0:
                strat_risk = -tail.mean()
            else:
                strat_risk = -var_threshold
        
        strategy_points.append(StrategyPoint(
            key=key,
            name=portfolio['name'],
            return_pct=float(strat_return * 100),
            risk_pct=float(strat_risk * 100),
            volatility_pct=float(strat_volatility * 100),
            sharpe=float(strat_sharpe)
        ))
    
    # Calculate approximate efficient frontier
    n_buckets = 50
    risk_values = [p.risk_pct for p in random_portfolios]
    risk_min, risk_max = min(risk_values), max(risk_values)
    bucket_size = (risk_max - risk_min) / n_buckets
    
    frontier_points = []
    for i in range(n_buckets):
        bucket_start = risk_min + i * bucket_size
        bucket_end = bucket_start + bucket_size
        
        bucket_portfolios = [
            p for p in random_portfolios
            if bucket_start <= p.risk_pct < bucket_end
        ]
        
        if bucket_portfolios:
            best = max(bucket_portfolios, key=lambda p: p.return_pct)
            # Find volatility for this portfolio
            idx = random_portfolios.index(best)
            w = best.weights
            vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            
            frontier_points.append(FrontierPoint(
                risk_pct=best.risk_pct,
                return_pct=best.return_pct,
                volatility_pct=float(vol * 100),
                sharpe=best.sharpe,
                weights=best.weights
            ))
    
    # Sort frontier by risk
    frontier_points.sort(key=lambda p: p.risk_pct)
    
    # Calculate statistics
    all_sharpes = [p.sharpe for p in random_portfolios]
    all_risks = [p.risk_pct for p in random_portfolios]
    all_returns = [p.return_pct for p in random_portfolios]
    
    best_sharpe_idx = np.argmax(all_sharpes)
    min_risk_idx = np.argmin(all_risks)
    max_return_idx = np.argmax(all_returns)
    
    stats = {
        "best_random_sharpe": float(all_sharpes[best_sharpe_idx]),
        "best_sharpe_return": float(all_returns[best_sharpe_idx]),
        "best_sharpe_risk": float(all_risks[best_sharpe_idx]),
        "min_risk": float(all_risks[min_risk_idx]),
        "min_risk_return": float(all_returns[min_risk_idx]),
        "max_return": float(all_returns[max_return_idx]),
        "max_return_risk": float(all_risks[max_return_idx])
    }
    
    # Calculate cumulative returns for each strategy (for historical comparison)
    dates = [d.strftime('%Y-%m-%d') for d in analyzer.returns.index]
    strategy_cumulative = {}
    
    for key, portfolio in analyzer.portfolios.items():
        cumulative = (1 + portfolio['returns']).cumprod() * 100
        strategy_cumulative[key] = [float(v) for v in cumulative.values]
    
    # Also store daily returns matrix for frontier portfolio calculation
    # (will be used client-side to calculate cumulative for any weight combination)
    
    return FrontierDataResponse(
        success=True,
        symbols=analyzer.symbols,
        frontier_type=frontier_type,
        cvar_alpha=cvar_alpha if frontier_type == "mean_cvar" else None,
        n_portfolios=n_portfolios,
        allow_short=allow_short,
        random_portfolios=random_portfolios,
        strategy_points=strategy_points,
        frontier_points=frontier_points,
        stats=stats,
        dates=dates,
        strategy_cumulative=strategy_cumulative
    )