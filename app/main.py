"""
Integrated FastAPI Main Application - Zero dependency breakage
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging
import os

# Import and include the API router
from .api import router as api_router

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import new components with robust error handling
cache = None
Base = None
engine = None
NEW_COMPONENTS_AVAILABLE = False

try:
    from .cache_manager import cache
    logger.info("✅ Cache manager initialized")
    NEW_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Cache manager not available: {e}")

try:
    from .db_models import Base, engine
    logger.info("✅ Database models imported")
    NEW_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.info(f"ℹ️ Database models not available: {e}")

# Scheduler functions with fallbacks
def start_scheduler():
    """Start background scheduler if available"""
    try:
        from .scheduler import start_scheduler as real_start_scheduler
        real_start_scheduler()
    except ImportError as e:
        logger.info(f"ℹ️ Scheduler not available: {e}")

def stop_scheduler():
    """Stop background scheduler if available"""
    try:
        from .scheduler import stop_scheduler as real_stop_scheduler
        real_stop_scheduler()
    except ImportError:
        pass  # Silent fail on shutdown

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting DeFi Risk API...")
    print("Server starting...")
    print("Documentation: http://127.0.0.1:8000/docs")
    print("API endpoints: http://127.0.0.1:8000/api")
    
    # Initialize components if available
    if NEW_COMPONENTS_AVAILABLE:
        # Create database tables
        if Base and engine:
            try:
                Base.metadata.create_all(bind=engine)
                logger.info("✅ Database tables created")
            except Exception as e:
                logger.error(f"❌ Database table creation failed: {e}")
        
        # Check cache health
        if cache:
            cache_health = cache.health_check()
            logger.info(f"✅ Cache: {cache_health['info']}")
        
        # Start scheduler
        start_scheduler()
    else:
        logger.info("⚡ Running in compatible mode - all core features active")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    stop_scheduler()

# Create FastAPI app
app_kwargs = {
    "title": "DeFi Risk Early-Warning System",
    "description": "Real-time liquidation risk monitoring for DeFi protocols", 
    "version": "2.0.0"
}

# Only use lifespan if we have new components to avoid conflicts
if NEW_COMPONENTS_AVAILABLE:
    app_kwargs["lifespan"] = lifespan

app = FastAPI(**app_kwargs)

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173", 
    "https://*.vercel.app",
]

# Allow all origins in development
if os.getenv("ENVIRONMENT") == "development":
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes (maintains existing data pipeline)
app.include_router(api_router, prefix="/api", tags=["API"])

# Root endpoint - always works
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DeFi Risk Early-Warning System API",
        "version": "2.0.0",
        "status": "running", 
        "docs": "/docs",
        "health": "/health",
        "mode": "enhanced" if NEW_COMPONENTS_AVAILABLE else "compatible"
    }

# Health check endpoint - always available
# Health check endpoint - always available
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_info = {
        "status": "healthy", 
        "version": "2.0.0",
        "api": "operational",
        "timestamp": "2025-10-08T11:12:54Z"  # You can make this dynamic
    }
    
    # Add cache health if available
    if cache:
        cache_health = cache.health_check()
        health_info["cache"] = cache_health
        if not cache_health.get("connected", False):
            health_info["status"] = "degraded"
    else:
        health_info["cache"] = {"available": False, "info": "No cache configured"}
    
    # Add database health if available - with SQLAlchemy 2.0 fix
    # In your health_check function, replace the database section with:
    if Base and engine:
        try:
            from .db_models import SessionLocal
            from sqlalchemy import text  # Add this import
            
            db = SessionLocal()
            db.execute(text("SELECT 1"))  # Wrap with text()
            db.close()
            health_info["database"] = "connected"
        except Exception as e:
            health_info["database"] = f"error: {e}"
            health_info["status"] = "degraded"
    else:
        health_info["database"] = "not_configured"
    
    return health_info

@app.get("/test-redis-cloud")
async def test_redis_cloud():
    """Test Redis Cloud connection with SSL"""
    import redis
    from urllib.parse import urlparse
    
    redis_url = os.getenv("REDIS_URL")
    
    try:
        # Parse URL to hide password in logs
        parsed = urlparse(redis_url)
        safe_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        
        r = redis.Redis.from_url(
            redis_url, 
            decode_responses=True,
            ssl_cert_reqs=None  # Important for Redis Cloud SSL
        )
        r.ping()
        
        # Test write/read
        r.set("defi_test", "Connected to Redis Cloud!", ex=60)
        value = r.get("defi_test")
        
        return {
            "status": "success", 
            "message": "Redis Cloud is connected!",
            "redis_url": safe_url,
            "test_value": value,
            "ssl_enabled": True
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Failed to connect to Redis Cloud",
            "error": str(e),
            "your_redis_url_scheme": redis_url.split('://')[0] if '://' in redis_url else 'missing'
        }
        

