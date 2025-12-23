"""
Portfolio Analyzer Pro - FastAPI Backend
This API exposes the portfolio analysis functionality as REST endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routes
from api.routes.portfolio import router as portfolio_router

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

# Include routers
app.include_router(portfolio_router)


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