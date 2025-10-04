"""
Updated database models - Reserve model aligned with RPC output
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aave_risk.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Reserve(Base):
    """Reserve model aligned with RPC blockchain data"""
    __tablename__ = "reserves"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Identification
    chain = Column(String, index=True, nullable=False)
    token_address = Column(String, index=True, nullable=False)
    token_symbol = Column(String, index=True)
    token_name = Column(String)
    decimals = Column(Integer)
    
    # Interest rates (decimal format, not percentages)
    liquidity_rate = Column(Float)  # Supply APY in decimal
    variable_borrow_rate = Column(Float)  # Borrow APY in decimal
    stable_borrow_rate = Column(Float)  # Stable borrow APY in decimal
    
    # Percentage formats for display
    supply_apy = Column(Float)  # liquidity_rate * 100
    borrow_apy = Column(Float)  # variable_borrow_rate * 100
    
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
    atoken_address = Column(String)
    variable_debt_token_address = Column(String)
    
    # Pricing
    price_usd = Column(Float)
    price_available = Column(Boolean, default=False)
    
    # Timestamps
    last_update_timestamp = Column(Integer)  # From blockchain
    query_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Position(Base):
    """Position model - unchanged"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    borrower_address = Column(String, nullable=False, index=True)
    chain = Column(String)
    
    token_symbol = Column(String)
    token_address = Column(String)
    
    collateral_amount = Column(Float)
    debt_amount = Column(Float)
    health_factor = Column(Float)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    
    enhanced_health_factor = Column(Float)
    risk_category = Column(String)
    current_ltv = Column(Float)
    liquidation_threshold = Column(Float)
    liquidation_price = Column(Float)
    price_drop_to_liquidation_pct = Column(Float)
    position_size_category = Column(String)
    
    current_collateral_usd = Column(Float)
    current_price = Column(Float)
    price_available = Column(Boolean, default=False)
    
    last_updated = Column(DateTime)
    query_time = Column(DateTime)

class LiquidationHistory(Base):
    """Liquidation history - unchanged"""
    __tablename__ = "liquidation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    tx_hash = Column(String, index=True)
    block_number = Column(Integer)
    liquidation_date = Column(DateTime, index=True)
    chain = Column(String)
    
    liquidator = Column(String, index=True)
    borrower = Column(String, index=True)
    
    collateral_asset = Column(String)
    collateral_symbol = Column(String)
    debt_asset = Column(String)
    debt_symbol = Column(String)
    
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
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    query_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class AnalysisSnapshot(Base):
    """Analysis snapshots - unchanged"""
    __tablename__ = "analysis_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    summary = Column(JSON, nullable=False)
    
    total_positions = Column(Integer)
    critical_positions = Column(Integer)
    protocol_ltv = Column(Float)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    analysis_duration_seconds = Column(Float)
    price_data_coverage = Column(Float)
    positions_analyzed = Column(Integer)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()