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