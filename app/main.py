"""
Minimal Working FastAPI Main Application
Replace your current app/main.py with this
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="DeFi Liquidation Risk System",
    description="Early-warning system for AAVE lending protocol",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include the API router
from .api import router as api_router

app.include_router(api_router, prefix="/api", tags=["API"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DeFi Liquidation Risk System",
        "status": "running",
        "documentation": "/docs",
        "api_base": "/api"
    }

@app.on_event("startup")
async def startup():
    print("Server starting...")
    print("Documentation: http://127.0.0.1:8000/docs")
    print("API endpoints: http://127.0.0.1:8000/api")