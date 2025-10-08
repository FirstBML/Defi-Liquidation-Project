"""
Database models - SQLAlchemy 1.4 compatible
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime, timezone
import os
import logging

logger = logging.getLogger(__name__)

def get_database_url():
    """Get database URL - works for both SQLite and PostgreSQL"""
    database_url = os.getenv("DATABASE_URL")
    
    if database_url and database_url.startswith("postgres"):
        # PostgreSQL on Railway
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        logger.info("✅ Using PostgreSQL (Railway)")
        return database_url
    else:
        # SQLite for local development
        logger.info("✅ Using SQLite (local development)")
        return "sqlite:///./aave_risk.db"

def create_database_engine():
    """Create database engine"""
    database_url = get_database_url()
    
    try:
        if database_url.startswith("postgresql://"):
            engine = create_engine(database_url)
            logger.info("✅ PostgreSQL engine created")
        else:
            # SQLite for local development
            engine = create_engine(database_url)
            logger.info("✅ SQLite engine created")
        return engine
    except Exception as e:
        logger.error(f"❌ Database engine creation failed: {e}")
        # Ultimate fallback to SQLite
        return create_engine("sqlite:///./aave_risk.db")

# Create engine and session
engine = create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Reserve(Base):
    """Reserve model aligned with RPC blockchain data"""
    __tablename__ = "reserves"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Identification
    chain = Column(String(50), index=True, nullable=False)
    token_address = Column(String(100), index=True, nullable=False)
    token_symbol = Column(String(50), index=True)
    token_name = Column(String(200))
    decimals = Column(Integer)
    
    # Interest rates (decimal format, not percentages)
    liquidity_rate = Column(Float)
    variable_borrow_rate = Column(Float)
    stable_borrow_rate = Column(Float)
    
    # Percentage formats for display
    supply_apy = Column(Float)
    borrow_apy = Column(Float)
    
    # Risk parameters (decimal format: 0.0 to 1.0)
    ltv = Column(Float)
    liquidation_threshold = Column(Float)
    liquidation_bonus = Column(Float)
    
    # Reserve status flags
    is_active = Column(Boolean, default=True)
    is_frozen = Column(Boolean, default=False)
    borrowing_enabled = Column(Boolean, default=True)
    stable_borrowing_enabled = Column(Boolean, default=False)
    
    # Indices
    liquidity_index = Column(Float)
    variable_borrow_index = Column(Float)
    
    # Contract addresses
    atoken_address = Column(String(100))
    variable_debt_token_address = Column(String(100))
    
    # Pricing
    price_usd = Column(Float)
    price_available = Column(Boolean, default=False)
    
    # Timestamps
    last_update_timestamp = Column(Integer)
    query_time = Column(DateTime, default=datetime.now(timezone.utc))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

class Position(Base):
    """Position model"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    borrower_address = Column(String(100), nullable=False, index=True)
    chain = Column(String(50))
    
    token_symbol = Column(String(50))
    token_address = Column(String(100))
    
    collateral_amount = Column(Float)
    debt_amount = Column(Float)
    health_factor = Column(Float)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    
    enhanced_health_factor = Column(Float)
    risk_category = Column(String(50))
    current_ltv = Column(Float)
    liquidation_price = Column(Float)
    price_drop_to_liquidation_pct = Column(Float)
    position_size_category = Column(String(50))
    
    current_collateral_usd = Column(Float)
    current_price = Column(Float)
    price_available = Column(Boolean, default=False)
    
    last_updated = Column(DateTime)
    query_time = Column(DateTime)

class LiquidationHistory(Base):
    """Liquidation history"""
    __tablename__ = "liquidation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    tx_hash = Column(String(100), index=True)
    block_number = Column(Integer)
    liquidation_date = Column(DateTime, index=True)
    chain = Column(String(50))
    
    liquidator = Column(String(100), index=True)
    borrower = Column(String(100), index=True)
    
    collateral_asset = Column(String(100))
    collateral_symbol = Column(String(50))
    debt_asset = Column(String(100))
    debt_symbol = Column(String(50))
    
    liquidated_collateral_amount = Column(Float)
    liquidated_debt_amount = Column(Float)
    liquidated_collateral_usd = Column(Float)
    liquidated_debt_usd = Column(Float)
    
    total_collateral_seized = Column(Float)
    total_debt_normalized = Column(Float)
    liquidation_count = Column(Integer)
    avg_debt_per_event = Column(Float)
    unique_liquidators = Column(Integer)
    
    health_factor_before = Column(Float)
    total_collateral_before = Column(Float)
    total_debt_before = Column(Float)
    health_factor_after = Column(Float)
    
    gas_used = Column(Float)
    gas_price = Column(Float)
    gas_cost_usd = Column(Float)
    
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    query_time = Column(DateTime, default=datetime.now(timezone.utc))

class AnalysisSnapshot(Base):
    """Analysis snapshots"""
    __tablename__ = "analysis_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    summary = Column(JSON, nullable=False)
    
    total_positions = Column(Integer)
    critical_positions = Column(Integer)
    protocol_ltv = Column(Float)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    analysis_duration_seconds = Column(Float)
    price_data_coverage = Column(Float)
    positions_analyzed = Column(Integer)

def initialize_database():
    """Initialize database"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

# Initialize database
initialize_database()

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()