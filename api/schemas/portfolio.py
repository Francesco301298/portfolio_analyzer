"""
Pydantic schemas for request/response validation.
These define the structure of data sent to and received from the API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date


# ============ REQUEST SCHEMAS ============

class MarketDataRequest(BaseModel):
    """Request schema for downloading market data."""
    symbols: List[str] = Field(
        ..., 
        min_length=1,
        max_length=20,
        description="List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])",
        json_schema_extra={"example": ["AAPL", "MSFT", "GOOGL"]}
    )
    start_date: date = Field(
        ..., 
        description="Start date for historical data",
        json_schema_extra={"example": "2020-01-01"}
    )
    end_date: date = Field(
        ..., 
        description="End date for historical data",
        json_schema_extra={"example": "2024-12-01"}
    )


class PortfolioAnalysisRequest(BaseModel):
    """Request schema for full portfolio analysis."""
    symbols: List[str] = Field(
        ..., 
        min_length=2,
        max_length=20,
        description="List of ticker symbols"
    )
    start_date: date = Field(..., description="Start date")
    end_date: date = Field(..., description="End date")
    risk_free_rate: float = Field(
        default=0.02, 
        ge=0, 
        le=0.2,
        description="Annual risk-free rate (e.g., 0.02 = 2%)"
    )
    custom_weights: Optional[List[float]] = Field(
        default=None,
        description="Custom portfolio weights (must sum to 1)"
    )


# ============ RESPONSE SCHEMAS ============

class MarketDataResponse(BaseModel):
    """Response schema for market data."""
    success: bool
    symbols: List[str]
    start_date: str
    end_date: str
    trading_days: int
    message: str


class PortfolioMetrics(BaseModel):
    """Metrics for a single portfolio strategy."""
    name: str
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    weights: Dict[str, float]


class PortfolioAnalysisResponse(BaseModel):
    """Response schema for portfolio analysis."""
    success: bool
    portfolios: Dict[str, PortfolioMetrics]
    correlation_matrix: Dict[str, Dict[str, float]]
    symbols: List[str]
    period: Dict[str, str]
    trading_days: int

class AssetStats(BaseModel):
    """Statistics for a single asset"""
    ticker: str
    name: str
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

class AssetPricePoint(BaseModel):
    """Single price point"""
    date: str
    value: float

class AssetTimeSeries(BaseModel):
    """Time series for one asset"""
    ticker: str
    prices: List[AssetPricePoint]

class OverviewDataResponse(BaseModel):
    """Response with all data needed for Overview tab"""
    success: bool
    symbols: List[str]
    trading_days: int
    period: dict
    # Normalized prices (base 100)
    price_series: List[AssetTimeSeries]
    # Individual asset statistics
    asset_stats: List[AssetStats]
    # Correlation matrix
    correlation_matrix: dict
    # Top/Bottom performers
    top_performers: List[dict]
    bottom_performers: List[dict]

class PortfolioTimeSeries(BaseModel):
    """Time series data for a portfolio strategy"""
    key: str
    name: str
    dates: List[str]
    cumulative_returns: List[float]  # Base 100
    drawdown: List[float]  # Percentage drawdown
    
class RollingMetrics(BaseModel):
    """Rolling metrics for a portfolio"""
    key: str
    name: str
    dates: List[str]
    rolling_return: List[Optional[float]]  # Annualized rolling return
    rolling_volatility: List[Optional[float]]  # Annualized rolling volatility

class DrawdownStats(BaseModel):
    """Drawdown statistics for a portfolio"""
    key: str
    name: str
    max_drawdown: float
    max_drawdown_date: str
    recovery_time: Optional[str]  # "X days" or "Not recovered"

class PerformanceDataResponse(BaseModel):
    """Response with all data needed for Performance tab"""
    success: bool
    symbols: List[str]
    trading_days: int
    period: dict
    window_days: int  # Rolling window in days
    # Time series for each portfolio
    portfolio_series: List[PortfolioTimeSeries]
    # Rolling metrics
    rolling_metrics: List[RollingMetrics]
    # Drawdown statistics
    drawdown_stats: List[DrawdownStats]
    # Rolling statistics summary
    rolling_summary: List[dict]

class RandomPortfolio(BaseModel):
    """A single random portfolio point"""
    return_pct: float  # Annualized return %
    risk_pct: float  # Risk measure % (volatility or CVaR)
    sharpe: float
    weights: List[float]

class StrategyPoint(BaseModel):
    """An optimized strategy point on the frontier"""
    key: str
    name: str
    return_pct: float
    risk_pct: float  # Volatility for MV, CVaR for MC
    volatility_pct: float  # Always store volatility
    sharpe: float

class FrontierPoint(BaseModel):
    """A point on the efficient frontier"""
    risk_pct: float
    return_pct: float
    volatility_pct: float
    sharpe: float
    weights: List[float]

class FrontierDataResponse(BaseModel):
    """Response with all data needed for Frontier tab"""
    success: bool
    symbols: List[str]
    frontier_type: str  # "mean_variance" or "mean_cvar"
    cvar_alpha: Optional[float]
    n_portfolios: int
    allow_short: bool
    # Random portfolios cloud
    random_portfolios: List[RandomPortfolio]
    # Optimized strategies
    strategy_points: List[StrategyPoint]
    # Approximate efficient frontier points
    frontier_points: List[FrontierPoint]
    # Statistics
    stats: dict
    # Historical data for performance comparison
    dates: List[str]  # Trading dates
    strategy_cumulative: dict  # {strategy_key: [cumulative values]}