"""
Enhanced FastAPI Application - FIXED
Handles missing PostgreSQL gracefully
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
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', module='eth_utils')

# Load environment variables
load_dotenv()

# ==================== DATABASE CHECK ====================
def check_database():
    """Check if database is accessible"""
    try:
        from app.db_models import SessionLocal, engine
        
        # Try to connect
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        logger.info("✅ Database connection successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error("=" * 70)
        logger.error("⚠️  POSTGRESQL NOT RUNNING OR NOT CONFIGURED")
        logger.error("=" * 70)
        logger.error("Solutions:")
        logger.error("1. Start PostgreSQL server: pg_ctl -D /path/to/data start")
        logger.error("2. Or use Railway/cloud PostgreSQL (set DATABASE_URL)")
        logger.error("3. App will run but database features disabled")
        logger.error("=" * 70)
        return False

# Check database on startup
DB_AVAILABLE = check_database()

# Import API with proper error handling
try:
    from app import api
    logger.info("✅ API module imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import API module: {e}")
    raise

# ==================== LIFESPAN ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"[{datetime.now(timezone.utc)}] 🚀 Starting FastAPI app...")
    logger.info(f"🌍 Environment: {'Production' if os.getenv('RAILWAY_ENVIRONMENT') else 'Development'}")
    logger.info(f"🔌 Port: {os.getenv('PORT', '8080')}")
    
    if not DB_AVAILABLE:
        logger.warning("⚠️  Starting without database connection")
        yield
        return
    
    # Initialize services only if DB available
    try:
        api.init_services()
        logger.info("✅ Portfolio and Alert services initialized")
    except Exception as e:
        logger.error(f"⚠️ Service initialization failed: {e}")
    
    # Start scheduler
    try:
        from app.scheduler import unified_scheduler
        unified_scheduler.start_scheduler()
        logger.info("✅ Scheduler initialized")
    except Exception as e:
        logger.error(f"⚠️ Scheduler failed to start: {e}")
    
    # Check database
    if DB_AVAILABLE:
        try:
            from app.db_models import SessionLocal, Reserve, Position
            db = SessionLocal()
            reserve_count = db.query(Reserve).count()
            position_count = db.query(Position).count()
            logger.info(f"📊 Current data: {reserve_count} reserves, {position_count} positions")
            
            if reserve_count == 0 or position_count == 0:
                logger.info("🚨 Database is empty - visit /docs and call POST /api/data/refresh")
            
            db.close()
        except Exception as e:
            logger.warning(f"⚠️ Initial data check failed: {e}")
    
    yield
    
    logger.info(f"[{datetime.now(timezone.utc)}] 🛑 Shutting down...")
    try:
        from app.scheduler import unified_scheduler
        if unified_scheduler.scheduler.running:
            unified_scheduler.stop_scheduler()
            logger.info("✅ Scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")

# ==================== APP INITIALIZATION ====================

app = FastAPI(
    title="DeFi Risk Early-Warning System",
    description="Real-time liquidation risk monitoring",
    version="2.0.0",
    lifespan=lifespan,
)

# ==================== CORS ====================

ALLOWED_ORIGINS = [
    "https://perspectively-slaty-sheilah.ngrok-free.dev",
    "http://localhost:3000",
    "https://easygoing-charm-production-707b.up.railway.app",
    "https://*.railway.app",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== INCLUDE ROUTERS ====================

try:
    app.include_router(api.router_v1)
    app.include_router(api.router_v2)
    logger.info("✅ Both API routers included")
except Exception as e:
    logger.error(f"❌ Failed to include routers: {e}")
    raise

# ==================== ROOT ENDPOINTS ====================

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    if not DB_AVAILABLE:
        return {
            "status": "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "Database not available",
            "message": "PostgreSQL connection required"
        }
    
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
    if not DB_AVAILABLE:
        return {
            "api_loaded": True,
            "database_connected": False,
            "error": "PostgreSQL not running",
            "message": "Start PostgreSQL or set DATABASE_URL"
        }
    
    from app.db_models import SessionLocal, Reserve, Position
    
    try:
        from app.scheduler import unified_scheduler
        scheduler_running = unified_scheduler
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
        "database_url_set": bool(os.getenv("DATABASE_URL")),
        "services_initialized": api.portfolio_service is not None
    }