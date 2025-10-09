"""
Database models with fixed Railway PostgreSQL connection
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import os
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

# ==================== DATABASE CONNECTION ====================

def get_database_url():
    """Get database URL with proper error handling"""
    
    # Railway provides DATABASE_URL automatically
    db_url = os.getenv("DATABASE_URL")
    
    if db_url:
        # Railway PostgreSQL URLs start with postgres://
        # SQLAlchemy 1.4+ requires postgresql://
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        
        logger.info("✅ Using PostgreSQL (Railway)")
        return db_url
    
    # Fallback to SQLite for local development
    logger.info("⚠️ No DATABASE_URL found, using SQLite (local dev)")
    return "sqlite:///./aave_risk.db"

try:
    DATABASE_URL = get_database_url()
    
    # Create engine with appropriate settings
    if DATABASE_URL.startswith("postgresql://"):
        # PostgreSQL settings
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections after 1 hour
            connect_args={
                "connect_timeout": 10,
                "options": "-c timezone=utc"
            }
        )
    else:
        # SQLite settings
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False}
        )
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    logger.info("✅ Database engine created successfully")
    
except Exception as e:
    logger.error(f"❌ Database engine creation failed: {e}")
    # Fallback to SQLite if PostgreSQL fails
    logger.info("🔄 Falling back to SQLite...")
    DATABASE_URL = "sqlite:///./aave_risk.db"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==================== MODELS ====================

class Reserve(Base):
    """RPC-sourced reserve data"""
    __tablename__ = "reserves"
    
    id = Column(Integer, primary_key=True, index=True)
    chain = Column(String, index=True)
    token_address = Column(String, index=True)
    token_symbol = Column(String, index=True)
    token_name = Column(String, nullable=True)
    decimals = Column(Integer)
    
    # Interest rates (raw values from contract)
    liquidity_rate = Column(Float, nullable=True)
    variable_borrow_rate = Column(Float, nullable=True)
    stable_borrow_rate = Column(Float, nullable=True)
    
    # APYs (calculated)
    supply_apy = Column(Float, nullable=True)
    borrow_apy = Column(Float, nullable=True)
    
    # Risk parameters
    ltv = Column(Float, nullable=True)
    liquidation_threshold = Column(Float, nullable=True)
    liquidation_bonus = Column(Float, nullable=True)
    
    # Status flags
    is_active = Column(Boolean, default=True)
    is_frozen = Column(Boolean, default=False)
    borrowing_enabled = Column(Boolean, default=True)
    stable_borrowing_enabled = Column(Boolean, default=False)
    
    # Indices
    liquidity_index = Column(Float, nullable=True)
    variable_borrow_index = Column(Float, nullable=True)
    
    # Token addresses
    atoken_address = Column(String, nullable=True)
    variable_debt_token_address = Column(String, nullable=True)
    
    # Price data
    price_usd = Column(Float, nullable=True)
    price_available = Column(Boolean, default=False)
    
    # Timestamps
    last_update_timestamp = Column(Integer, nullable=True)
    query_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Position(Base):
    """User positions (from Dune)"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    borrower_address = Column(String, index=True)
    chain = Column(String, index=True)
    token_symbol = Column(String, index=True)
    token_address = Column(String, nullable=True)
    
    collateral_amount = Column(Float, nullable=True)
    debt_amount = Column(Float, nullable=True)
    health_factor = Column(Float, nullable=True)
    
    total_collateral_usd = Column(Float, nullable=True)
    total_debt_usd = Column(Float, nullable=True)
    
    enhanced_health_factor = Column(Float, nullable=True)
    risk_category = Column(String, nullable=True)
    liquidation_threshold = Column(Float, nullable=True)
    
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    query_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class LiquidationHistory(Base):
    """Liquidation events (from Dune)"""
    __tablename__ = "liquidation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    liquidation_date = Column(DateTime, index=True)
    chain = Column(String, index=True)
    
    borrower = Column(String, nullable=True)
    liquidator = Column(String, nullable=True)
    
    collateral_symbol = Column(String, nullable=True)
    debt_symbol = Column(String, nullable=True)
    collateral_asset = Column(String, nullable=True)
    debt_asset = Column(String, nullable=True)
    
    total_collateral_seized = Column(Float, nullable=True)
    total_debt_normalized = Column(Float, nullable=True)
    
    # USD values
    liquidated_collateral_usd = Column(Float, nullable=True)
    liquidated_debt_usd = Column(Float, nullable=True)
    
    # Metadata
    liquidation_count = Column(Integer, nullable=True)
    avg_debt_per_event = Column(Float, nullable=True)
    unique_liquidators = Column(Integer, nullable=True)
    health_factor_before = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    query_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AnalysisSnapshot(Base):
    """Periodic analysis snapshots"""
    __tablename__ = "analysis_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    snapshot_time = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    total_positions = Column(Integer)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    avg_health_factor = Column(Float, nullable=True)
    
    critical_positions = Column(Integer, nullable=True)
    high_risk_positions = Column(Integer, nullable=True)
    
    scenario_data = Column(Text, nullable=True)  # JSON string
    alert_summary = Column(Text, nullable=True)  # JSON string


# ==================== INITIALIZATION ====================

def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

# Auto-initialize on import
init_db()