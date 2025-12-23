"""
Portfolio API Routes
These endpoints handle portfolio analysis operations.
"""

from fastapi import APIRouter, HTTPException
from api.schemas.portfolio import (
    MarketDataRequest,
    MarketDataResponse,
    PortfolioAnalysisRequest,
    PortfolioAnalysisResponse,
    PortfolioMetrics
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