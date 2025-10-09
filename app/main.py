"""
Enhanced FastAPI Application for DeFi Liquidation Risk System
Fixed for Railway deployment
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi.responses import RedirectResponse
import warnings
import logging

# Setup logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', module='eth_utils')

# Load environment variables
load_dotenv()

# Import with error handling
try:
    from app import api
    logger.info("✅ API module imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import API module: {e}")
    try:
        import api
        logger.info("✅ API module imported (fallback)")
    except ImportError as e2:
        logger.error(f"❌ Fallback import failed: {e2}")
        raise

# ------------------------------------------------------
# 🔹 Application Lifespan (startup/shutdown events)
# ------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"[{datetime.now(timezone.utc)}] 🚀 Starting FastAPI app...")
    
    # Start scheduler with error handling
    try:
        from app.scheduler import start_scheduler
        start_scheduler()
        logger.info("✅ Scheduler initialized")
    except Exception as e:
        logger.error(f"⚠️ Scheduler failed to start (non-critical): {e}")
        # Don't crash the app if scheduler fails
    
    yield
    
    logger.info(f"[{datetime.now(timezone.utc)}] 🛑 Shutting down FastAPI app...")
    try:
        from app.scheduler import scheduler
        if scheduler.running:
            scheduler.shutdown()
            logger.info("✅ Scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")


# ------------------------------------------------------
# 🔹 Initialize FastAPI App
# ------------------------------------------------------
app = FastAPI(
    title="DeFi Risk Early-Warning System",
    description="Real-time liquidation risk monitoring for DeFi protocols",
    version="2.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------
# 🔹 CORS Configuration
# ------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://perspectively-slaty-sheilah.ngrok-free.dev",
    "https://your-vercel-dashboard.vercel.app",
    "http://localhost:3000",
    "https://easygoing-charm-production-707b.up.railway.app",  # Your Railway domain
    "*"  # Allow all for testing (remove in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------
# 🔹 Include API Router
# ------------------------------------------------------
try:
    app.include_router(api.router, prefix="/api")
    logger.info("✅ API router included")
except Exception as e:
    logger.error(f"❌ Failed to include API router: {e}")
    raise


# ------------------------------------------------------
# 🔹 Root & Health Endpoints
# ------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation"""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "DeFi Risk API",
        "version": "2.0.0"
    }

@app.get("/startup-status")
def startup_status():
    """Debug endpoint to check what loaded"""
    try:
        from app.scheduler import scheduler
        scheduler_running = scheduler.running
    except:
        scheduler_running = False
    
    return {
        "api_loaded": True,
        "scheduler_running": scheduler_running,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ------------------------------------------------------
# 🔹 Run (for local testing only)
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Disable reload in production
    )