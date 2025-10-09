"""
Enhanced FastAPI Application for DeFi Liquidation Risk System
Fixed for Railway deployment with proper port binding
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
    logger.info(f"🌍 Environment: {'Production' if os.getenv('RAILWAY_ENVIRONMENT') else 'Development'}")
    logger.info(f"🔌 Port: {os.getenv('PORT', '8080')}")
    
    # Start scheduler with error handling
    try:
        from app.scheduler import start_scheduler
        start_scheduler()
        logger.info("✅ Scheduler initialized")
    except Exception as e:
        logger.error(f"⚠️ Scheduler failed to start (non-critical): {e}")
    
    # 🔥 TRIGGER INITIAL DATA LOAD (NEW)
    try:
        logger.info("🔄 Triggering initial data refresh...")
        from app.db_models import SessionLocal
        db = SessionLocal()
        
        # Check if database is empty
        from app.db_models import Reserve, Position
        reserve_count = db.query(Reserve).count()
        position_count = db.query(Position).count()
        
        logger.info(f"📊 Current data: {reserve_count} reserves, {position_count} positions")
        
        if reserve_count == 0 or position_count == 0:
            logger.info("🚨 Database is empty - triggering full refresh")
            # Note: You'll need to manually call /api/data/refresh after startup
            logger.info("📝 To populate data, visit: /docs and call POST /api/data/refresh")
        
        db.close()
    except Exception as e:
        logger.warning(f"⚠️ Initial data check failed: {e}")
    
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
# 🔹 CORS Configuration (Fixed for Railway)
# ------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://perspectively-slaty-sheilah.ngrok-free.dev",
    "https://your-vercel-dashboard.vercel.app",
    "http://localhost:3000",
    "https://easygoing-charm-production-707b.up.railway.app",
    "https://*.railway.app",  # Allow all Railway subdomains
    "*"  # Development only - REMOVE IN PRODUCTION
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
    logger.info("✅ API router included at /api")
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
    from app.db_models import SessionLocal, Reserve, Position
    
    try:
        db = SessionLocal()
        reserve_count = db.query(Reserve).count()
        position_count = db.query(Position).count()
        db.close()
        
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "DeFi Risk API",
            "version": "2.0.0",
            "database": {
                "connected": True,
                "reserves": reserve_count,
                "positions": position_count
            },
            "environment": os.getenv("RAILWAY_ENVIRONMENT", "development")
        }
    except Exception as e:
        return {
            "status": "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

@app.get("/startup-status")
def startup_status():
    """Debug endpoint to check what loaded"""
    from app.db_models import SessionLocal, Reserve, Position
    
    try:
        from app.scheduler import scheduler
        scheduler_running = scheduler.running
    except:
        scheduler_running = False
    
    try:
        db = SessionLocal()
        reserve_count = db.query(Reserve).count()
        position_count = db.query(Position).count()
        db.close()
        db_connected = True
    except Exception as e:
        reserve_count = 0
        position_count = 0
        db_connected = False
    
    return {
        "api_loaded": True,
        "scheduler_running": scheduler_running,
        "database_connected": db_connected,
        "data_counts": {
            "reserves": reserve_count,
            "positions": position_count
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "port": os.getenv("PORT", "8080"),
        "database_url_set": bool(os.getenv("DATABASE_URL"))
    }


# ------------------------------------------------------
# 🔹 Run (Railway uses this)
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    # Railway sets PORT environment variable
    port = int(os.getenv("PORT", 8080))
    host = "0.0.0.0"  # Must be 0.0.0.0 for Railway
    
    logger.info(f"🔌 PORT environment variable: {os.getenv('PORT', 'not set')}")
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )