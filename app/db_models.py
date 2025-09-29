"""
Complete database models for DeFi Liquidation System
Includes models for all 3 Dune tables: reserves, positions, and liquidation history
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aave_risk.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Reserve(Base):
    """Model for df_reserve data from Dune - current-reserve endpoint"""
    __tablename__ = "reserves"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Reserve identification
    token_symbol = Column(String, index=True)
    token_address = Column(String, index=True)
    chain = Column(String)
    
    # Reserve metrics from Dune
    total_liquidity = Column(Float)
    available_liquidity = Column(Float)
    total_stable_borrows = Column(Float)
    total_variable_borrows = Column(Float)
    liquidity_rate = Column(Float)
    variable_borrow_rate = Column(Float)
    stable_borrow_rate = Column(Float)
    
    # Risk parameters
    ltv = Column(Float)  # Loan-to-Value ratio
    liquidation_threshold = Column(Float)
    liquidation_bonus = Column(Float)
    
    # Utilization metrics
    utilization_rate = Column(Float)
    
    # Pricing information
    price_usd = Column(Float)
    
    # Additional Dune fields
    supply_cap = Column(Float)
    borrow_cap = Column(Float)
    debt_ceiling = Column(Float)
    is_active = Column(Boolean)
    is_frozen = Column(Boolean)
    is_paused = Column(Boolean)
    is_borrowing_enabled = Column(Boolean)
    is_stable_rate_enabled = Column(Boolean)
    
    # Timestamps
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    query_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Position(Base):
    """Model for df_position data from Dune - current-position endpoint"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Position identification
    borrower_address = Column(String, nullable=False, index=True)
    chain = Column(String)
    
    # Token information
    token_symbol = Column(String)
    token_address = Column(String)
    
    # Core position metrics from Dune
    collateral_amount = Column(Float)
    debt_amount = Column(Float)
    health_factor = Column(Float)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    
    # Enhanced metrics
    enhanced_health_factor = Column(Float)
    risk_category = Column(String)
    current_ltv = Column(Float)
    liquidation_threshold = Column(Float)
    liquidation_price = Column(Float)
    price_drop_to_liquidation_pct = Column(Float)
    position_size_category = Column(String)
    
    # Current pricing
    current_collateral_usd = Column(Float)
    current_price = Column(Float)
    price_available = Column(Boolean, default=False)
    
    # Timestamps
    last_updated = Column(DateTime)
    query_time = Column(DateTime)

class LiquidationHistory(Base):
    """Model for df_liquidation_history data from Dune"""
    __tablename__ = "liquidation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Transaction identification
    tx_hash = Column(String, index=True)
    block_number = Column(Integer)
    liquidation_date = Column(DateTime, index=True)
    chain = Column(String)
    
    # Liquidation participants
    liquidator = Column(String, index=True)
    borrower = Column(String, index=True)
    
    # Asset information
    collateral_asset = Column(String)
    collateral_symbol = Column(String)
    debt_asset = Column(String)
    debt_symbol = Column(String)
    
    # Liquidation amounts
    liquidated_collateral_amount = Column(Float)
    liquidated_debt_amount = Column(Float)
    liquidated_collateral_usd = Column(Float)
    liquidated_debt_usd = Column(Float)
    
    # Aggregated metrics
    total_collateral_seized = Column(Float)
    total_debt_normalized = Column(Float)
    liquidation_count = Column(Integer)
    avg_debt_per_event = Column(Float)
    unique_liquidators = Column(Integer)
    
    # Health factor information
    health_factor_before = Column(Float)
    total_collateral_before = Column(Float)
    total_debt_before = Column(Float)
    health_factor_after = Column(Float)
    
    # Transaction costs
    gas_used = Column(Float)
    gas_price = Column(Float)
    gas_cost_usd = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    query_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class AnalysisSnapshot(Base):
    """Model for analysis snapshots"""
    __tablename__ = "analysis_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    summary = Column(JSON, nullable=False)
    
    # Key metrics
    total_positions = Column(Integer)
    critical_positions = Column(Integer)
    protocol_ltv = Column(Float)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    analysis_duration_seconds = Column(Float)
    price_data_coverage = Column(Float)
    positions_analyzed = Column(Integer)

# Database setup
Base.metadata.create_all(bind=engine)

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
