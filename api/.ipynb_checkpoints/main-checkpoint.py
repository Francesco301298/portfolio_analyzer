"""
Portfolio Analyzer Pro - FastAPI Backend
This API exposes the portfolio analysis functionality as REST endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Portfolio Analyzer Pro API",
    description="Advanced quantitative portfolio analysis API",
    version="1.0.0"
)

# CORS middleware - allows frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
        "https://advancedportfolioanalyzer.streamlit.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API welcome message."""
    return {
        "message": "Portfolio Analyzer Pro API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - useful for monitoring."""
    return {"status": "healthy"}


@app.get("/api/info")
async def api_info():
    """Returns information about available endpoints."""
    return {
        "endpoints": {
            "/": "API welcome message",
            "/health": "Health check",
            "/api/info": "This endpoint - lists available endpoints",
            "/api/portfolio/analyze": "(Coming soon) Analyze portfolio",
            "/api/portfolio/efficient-frontier": "(Coming soon) Generate efficient frontier",
            "/api/backtest/cpcv": "(Coming soon) Run CPCV backtest"
        }
    }