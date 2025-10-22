# ============================================================
# Enhanced Database Models with SQLite Thread Safety Fix
# ============================================================

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, JSON, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import StaticPool, NullPool
from datetime import datetime, timezone
from app.config import get_settings

import os
import logging

# ============================================================
# LOAD SETTINGS EARLY
# ============================================================
settings = get_settings()

DATABASE_URL = settings.DATABASE_URL
logger = logging.getLogger(__name__)

Base = declarative_base()

# ------------------------------------------------------------
# DATABASE CONNECTION SETUP (WITH FIX)
# ------------------------------------------------------------
def get_database_url():
    """Get database URL with proper error handling"""
    db_url = os.getenv("DATABASE_URL")

    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        logger.info("✅ Using PostgreSQL (Railway or remote)")
        return db_url

    logger.info("⚠️ No DATABASE_URL found, using local SQLite")
    return "sqlite:///./aave_risk.db"


try:
    DATABASE_URL = get_database_url()

    if DATABASE_URL.startswith("postgresql://"):
        # PostgreSQL engine configuration
        engine = create_engine(
            DATABASE_URL,
            poolclass=NullPool,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "connect_timeout": 10,
                "options": "-c timezone=utc"
            },
            echo=False
        )
    else:
        # ✅ FIXED: SQLite with StaticPool and check_same_thread=False
        engine = create_engine(
            DATABASE_URL,
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            },
            poolclass=StaticPool,  # Single shared connection
            pool_pre_ping=True,
            echo=False
        )

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("✅ Database engine created successfully")

except Exception as e:
    logger.error(f"❌ Database engine creation failed: {e}")
    logger.info("🔄 Falling back to SQLite...")
    DATABASE_URL = "sqlite:///./aave_risk.db"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ------------------------------------------------------------
# CORE MODELS
# ------------------------------------------------------------
class Reserve(Base):
    __tablename__ = "reserves"

    id = Column(Integer, primary_key=True, index=True)
    chain = Column(String, index=True)
    token_address = Column(String, index=True)
    token_symbol = Column(String, index=True)
    token_name = Column(String, nullable=True)
    decimals = Column(Integer)

    liquidity_rate = Column(Float, nullable=True)
    variable_borrow_rate = Column(Float, nullable=True)
    stable_borrow_rate = Column(Float, nullable=True)

    supply_apy = Column(Float, nullable=True)
    borrow_apy = Column(Float, nullable=True)

    ltv = Column(Float, nullable=True)
    liquidation_threshold = Column(Float, nullable=True)
    liquidation_bonus = Column(Float, nullable=True)

    is_active = Column(Boolean, default=True)
    is_frozen = Column(Boolean, default=False)
    borrowing_enabled = Column(Boolean, default=True)
    stable_borrowing_enabled = Column(Boolean, default=False)

    liquidity_index = Column(Float, nullable=True)
    variable_borrow_index = Column(Float, nullable=True)

    atoken_address = Column(String, nullable=True)
    variable_debt_token_address = Column(String, nullable=True)

    price_usd = Column(Float, nullable=True)
    price_available = Column(Boolean, default=False)

    last_update_timestamp = Column(Integer, nullable=True)
    query_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Position(Base):
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

    liquidated_collateral_usd = Column(Float, nullable=True)
    liquidated_debt_usd = Column(Float, nullable=True)

    liquidation_count = Column(Integer, nullable=True)
    avg_debt_per_event = Column(Float, nullable=True)
    unique_liquidators = Column(Integer, nullable=True)
    health_factor_before = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    query_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AnalysisSnapshot(Base):
    __tablename__ = "analysis_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_time = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    total_positions = Column(Integer)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    avg_health_factor = Column(Float, nullable=True)

    critical_positions = Column(Integer, nullable=True)
    high_risk_positions = Column(Integer, nullable=True)

    scenario_data = Column(Text, nullable=True)
    alert_summary = Column(Text, nullable=True)


# ------------------------------------------------------------
# ALERT SYSTEM MODELS
# ------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    telegram_chat_id = Column(String, unique=True, index=True, nullable=True)
    slack_webhook_url = Column(String, nullable=True)
    username = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)

    alert_subscriptions = relationship("AlertSubscription", back_populates="user")
    monitored_addresses = relationship("MonitoredAddress", back_populates="user")
    alert_history = relationship("AlertHistory", back_populates="user")


class AlertSubscription(Base):
    __tablename__ = "alert_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    channel_email = Column(Boolean, default=False)
    channel_telegram = Column(Boolean, default=False)
    channel_slack = Column(Boolean, default=False)
    health_factor_threshold = Column(Float, default=1.5)
    ltv_threshold = Column(Float, default=0.75)
    alert_on_liquidation = Column(Boolean, default=True)
    alert_on_new_risky_borrowers = Column(Boolean, default=True)
    alert_frequency_hours = Column(Integer, default=1)
    monitored_chains = Column(JSON, default=list)
    minimum_risk_level = Column(String, default="MEDIUM_RISK")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="alert_subscriptions")


class MonitoredAddress(Base):
    __tablename__ = "monitored_addresses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    wallet_address = Column(String, nullable=False, index=True)
    label = Column(String, nullable=True)
    custom_hf_threshold = Column(Float, nullable=True)
    notify_on_all_changes = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_checked = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="monitored_addresses")
    position_snapshots = relationship("PositionSnapshot", back_populates="monitored_address")


class PositionSnapshot(Base):
    __tablename__ = "position_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    monitored_address_id = Column(Integer, ForeignKey("monitored_addresses.id"))
    wallet_address = Column(String, nullable=False, index=True)
    chain = Column(String, nullable=False)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    health_factor = Column(Float)
    ltv_ratio = Column(Float)
    risk_category = Column(String)
    assets_breakdown = Column(JSON)
    snapshot_time = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    monitored_address = relationship("MonitoredAddress", back_populates="position_snapshots")


class AlertHistory(Base):
    __tablename__ = "alert_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    wallet_address = Column(String, nullable=True)
    chain = Column(String, nullable=True)
    message = Column(String, nullable=False)
    details = Column(JSON, nullable=True)
    sent_via_email = Column(Boolean, default=False)
    sent_via_telegram = Column(Boolean, default=False)
    sent_via_slack = Column(Boolean, default=False)
    delivery_status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    delivered_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="alert_history")


class RiskMetricHistory(Base):
    __tablename__ = "risk_metric_history"

    id = Column(Integer, primary_key=True, index=True)
    chain = Column(String, nullable=False, index=True)
    total_positions = Column(Integer)
    risky_positions = Column(Integer)
    critical_positions = Column(Integer)
    total_collateral_usd = Column(Float)
    total_debt_usd = Column(Float)
    protocol_ltv = Column(Float)
    average_health_factor = Column(Float)
    liquidations_24h = Column(Integer)
    liquidation_volume_24h_usd = Column(Float)
    recorded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class PriceMovement(Base):
    __tablename__ = "price_movements"

    id = Column(Integer, primary_key=True, index=True)
    token_symbol = Column(String, nullable=False, index=True)
    token_address = Column(String, nullable=False)
    chain = Column(String, nullable=False)
    price_usd = Column(Float, nullable=False)
    previous_price_usd = Column(Float, nullable=False)
    change_percent = Column(Float, nullable=False)
    positions_affected = Column(Integer, default=0)
    positions_at_risk = Column(Integer, default=0)
    recorded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


# ------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------
def init_db():
    """Initialize database - creates all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


if __name__ == "__main__":
    init_db()
