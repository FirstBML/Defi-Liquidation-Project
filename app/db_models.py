# ============================================================
# Enhanced Database Models with SQLite Thread Safety Fix
# ============================================================

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, JSON, Text, Index, CheckConstraint, func, event, exc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, validates
from sqlalchemy.pool import StaticPool, NullPool
from datetime import datetime, timezone
from app.config import get_settings
from enum import Enum as PyEnum
from sqlalchemy import Enum
import os
import logging

# ============================================================
# DATABASE CONFIGURATION
# ============================================================

class DatabaseConfig:
    """Database configuration constants"""
    POOL_RECYCLE_SECONDS = 3600  # 1 hour
    CONNECTION_TIMEOUT_SECONDS = 30
    POOL_SIZE = 5
    MAX_OVERFLOW = 10
    POOL_PRE_PING = True

# ============================================================
# ENUM TYPES
# ============================================================

class RiskCategory(PyEnum):
    SAFE = "SAFE"
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"

class DeliveryStatus(PyEnum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"

# ============================================================
# DATABASE SETUP
# ============================================================

Base = declarative_base()
logger = logging.getLogger(__name__)

def get_database_url() -> str:
    """Get database URL with proper error handling"""
    db_url = os.getenv("DATABASE_URL")

    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        logger.info("✅ Using PostgreSQL (Railway or remote)")
        return db_url

    logger.info("⚠️ No DATABASE_URL found, using local SQLite")
    return "sqlite:///./aave_risk.db"

def create_database_engine():
    """Create database engine with proper configuration"""
    DATABASE_URL = get_database_url()

    if DATABASE_URL.startswith("postgresql://"):
        # PostgreSQL engine configuration
        engine = create_engine(
            DATABASE_URL,
            poolclass=NullPool,
            pool_pre_ping=DatabaseConfig.POOL_PRE_PING,
            pool_recycle=DatabaseConfig.POOL_RECYCLE_SECONDS,
            connect_args={
                "connect_timeout": DatabaseConfig.CONNECTION_TIMEOUT_SECONDS,
                "options": "-c timezone=utc"
            },
            echo=False
        )
        
        # Add connection pool health checks for PostgreSQL
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            connection_record.info['pid'] = os.getpid()

        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            pid = os.getpid()
            if connection_record.info['pid'] != pid:
                connection_record.connection = connection_proxy.connection = None
                raise exc.DisconnectionError(
                    "Connection record belongs to pid %s, "
                    "attempting to check out in pid %s" %
                    (connection_record.info['pid'], pid)
                )
                
    else:
        # SQLite with StaticPool and check_same_thread=False
        engine = create_engine(
            DATABASE_URL,
            connect_args={
                "check_same_thread": False,
                "timeout": DatabaseConfig.CONNECTION_TIMEOUT_SECONDS
            },
            poolclass=StaticPool,  # Single shared connection
            pool_pre_ping=DatabaseConfig.POOL_PRE_PING,
            echo=False
        )

    return engine

try:
    engine = create_database_engine()
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

# ============================================================
# CORE MODELS
# ============================================================

class Reserve(Base):
    """
    Represents a lending pool reserve asset in AAVE protocol.
    
    Tracks interest rates, collateral parameters, and asset state.
    Updated periodically from on-chain data.
    """
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

    ltv = Column(Float, CheckConstraint('ltv >= 0 AND ltv <= 1'), nullable=True)
    liquidation_threshold = Column(Float, CheckConstraint('liquidation_threshold >= 0 AND liquidation_threshold <= 1'), nullable=True)
    liquidation_bonus = Column(Float, CheckConstraint('liquidation_bonus >= 0'), nullable=True)

    is_active = Column(Boolean, default=True)
    is_frozen = Column(Boolean, default=False)
    borrowing_enabled = Column(Boolean, default=True)
    stable_borrowing_enabled = Column(Boolean, default=False)

    liquidity_index = Column(Float, CheckConstraint('liquidity_index >= 0'), nullable=True)
    variable_borrow_index = Column(Float, CheckConstraint('variable_borrow_index >= 0'), nullable=True)

    atoken_address = Column(String, nullable=True)
    variable_debt_token_address = Column(String, nullable=True)

    price_usd = Column(Float, CheckConstraint('price_usd >= 0'), nullable=True)
    price_available = Column(Boolean, default=False)

    last_update_timestamp = Column(Integer, nullable=True)
    query_time = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, server_default=func.now())

    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_reserve_chain_token', 'chain', 'token_address'),
        Index('idx_reserve_chain_active', 'chain', 'is_active'),
        Index('idx_reserve_symbol_chain', 'token_symbol', 'chain'),
    )

    def __repr__(self):
        return f"<Reserve(symbol={self.token_symbol}, chain={self.chain}, active={self.is_active})>"

    @validates('decimals')
    def validate_decimals(self, key, value):
        if value is not None and (value < 0 or value > 36):
            raise ValueError("Decimals must be between 0 and 36")
        return value


class Position(Base):
    """
    Represents a user's borrowing position in AAVE protocol.
    
    Tracks collateral, debt, and health factor for risk monitoring.
    """
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    borrower_address = Column(String, index=True)
    chain = Column(String, index=True)
    token_symbol = Column(String, index=True)
    token_address = Column(String, nullable=True)

    collateral_amount = Column(Float, CheckConstraint('collateral_amount >= 0'), nullable=True)
    debt_amount = Column(Float, CheckConstraint('debt_amount >= 0'), nullable=True)
    health_factor = Column(Float, CheckConstraint('health_factor >= 0'), nullable=True)

    total_collateral_usd = Column(Float, CheckConstraint('total_collateral_usd >= 0'), nullable=True)
    total_debt_usd = Column(Float, CheckConstraint('total_debt_usd >= 0'), nullable=True)

    enhanced_health_factor = Column(Float, CheckConstraint('enhanced_health_factor >= 0'), nullable=True)
    risk_category = Column(Enum(RiskCategory), nullable=True)
    liquidation_threshold = Column(Float, CheckConstraint('liquidation_threshold >= 0 AND liquidation_threshold <= 1'), nullable=True)

    last_updated = Column(DateTime, server_default=func.now())
    query_time = Column(DateTime, server_default=func.now())

    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_position_borrower_chain', 'borrower_address', 'chain'),
        Index('idx_position_health_risk', 'health_factor', 'risk_category'),
        Index('idx_position_risk_chain', 'risk_category', 'chain'),
        Index('idx_position_updated_chain', 'last_updated', 'chain'),
    )

    def __repr__(self):
        return f"<Position(address={self.borrower_address[:8]}, chain={self.chain}, hf={self.health_factor})>"

    @validates('health_factor', 'enhanced_health_factor')
    def validate_health_factor(self, key, value):
        if value is not None and value < 0:
            raise ValueError("Health factor cannot be negative")
        return value


class LiquidationHistory(Base):
    """
    Historical record of liquidation events in AAVE protocol.
    
    Used for analyzing liquidation patterns and risk assessment.
    """
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

    total_collateral_seized = Column(Float, CheckConstraint('total_collateral_seized >= 0'), nullable=True)
    total_debt_normalized = Column(Float, CheckConstraint('total_debt_normalized >= 0'), nullable=True)

    liquidated_collateral_usd = Column(Float, CheckConstraint('liquidated_collateral_usd >= 0'), nullable=True)
    liquidated_debt_usd = Column(Float, CheckConstraint('liquidated_debt_usd >= 0'), nullable=True)

    liquidation_count = Column(Integer, CheckConstraint('liquidation_count >= 0'), nullable=True)
    avg_debt_per_event = Column(Float, CheckConstraint('avg_debt_per_event >= 0'), nullable=True)
    unique_liquidators = Column(Integer, CheckConstraint('unique_liquidators >= 0'), nullable=True)
    health_factor_before = Column(Float, CheckConstraint('health_factor_before >= 0'), nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    query_time = Column(DateTime, server_default=func.now())

    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_liquidation_date_chain', 'liquidation_date', 'chain'),
        Index('idx_liquidation_debt_usd', 'liquidated_debt_usd'),
        Index('idx_liquidation_borrower', 'borrower', 'liquidation_date'),
    )

    def __repr__(self):
        return f"<LiquidationHistory(date={self.liquidation_date}, chain={self.chain}, debt_usd={self.liquidated_debt_usd})>"


class AnalysisSnapshot(Base):
    """
    Snapshot of system-wide risk analysis at a point in time.
    
    Used for historical trend analysis and reporting.
    """
    __tablename__ = "analysis_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_time = Column(DateTime, server_default=func.now(), index=True)

    total_positions = Column(Integer, CheckConstraint('total_positions >= 0'))
    total_collateral_usd = Column(Float, CheckConstraint('total_collateral_usd >= 0'))
    total_debt_usd = Column(Float, CheckConstraint('total_debt_usd >= 0'))
    avg_health_factor = Column(Float, CheckConstraint('avg_health_factor >= 0'), nullable=True)

    critical_positions = Column(Integer, CheckConstraint('critical_positions >= 0'), nullable=True)
    high_risk_positions = Column(Integer, CheckConstraint('high_risk_positions >= 0'), nullable=True)

    scenario_data = Column(Text, nullable=True)
    alert_summary = Column(Text, nullable=True)

    def __repr__(self):
        return f"<AnalysisSnapshot(time={self.snapshot_time}, positions={self.total_positions})>"


# ============================================================
# ALERT SYSTEM MODELS
# ============================================================

class User(Base):
    """
    Represents a user of the risk monitoring system.
    
    Users can configure alert preferences and monitor addresses.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    telegram_chat_id = Column(String, unique=True, index=True, nullable=True)
    slack_webhook_url = Column(String, nullable=True)
    username = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime, nullable=True)

    alert_subscriptions = relationship(
        "AlertSubscription", 
        back_populates="user",
        cascade="all, delete-orphan"
    )
    monitored_addresses = relationship(
        "MonitoredAddress", 
        back_populates="user",
        cascade="all, delete-orphan"
    )
    alert_history = relationship(
        "AlertHistory", 
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"


class AlertSubscription(Base):
    """
    User's alert configuration and preferences.
    
    Defines what types of alerts to receive and through which channels.
    """
    __tablename__ = "alert_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    channel_email = Column(Boolean, default=False)
    channel_telegram = Column(Boolean, default=False)
    channel_slack = Column(Boolean, default=False)
    health_factor_threshold = Column(Float, CheckConstraint('health_factor_threshold >= 0'), default=1.5)
    ltv_threshold = Column(Float, CheckConstraint('ltv_threshold >= 0 AND ltv_threshold <= 1'), default=0.75)
    alert_on_liquidation = Column(Boolean, default=True)
    alert_on_new_risky_borrowers = Column(Boolean, default=True)
    alert_frequency_hours = Column(Integer, CheckConstraint('alert_frequency_hours >= 0'), default=1)
    monitored_chains = Column(JSON, default=list)
    minimum_risk_level = Column(Enum(RiskCategory), default=RiskCategory.MEDIUM_RISK)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="alert_subscriptions")

    __table_args__ = (
        Index('idx_subscription_user_active', 'user_id', 'is_active'),
    )

    def __repr__(self):
        return f"<AlertSubscription(user_id={self.user_id}, active={self.is_active})>"

    @validates('alert_frequency_hours')
    def validate_alert_frequency(self, key, value):
        if value < 0:
            raise ValueError("Alert frequency cannot be negative")
        return value


class MonitoredAddress(Base):
    """
    Wallet addresses monitored by users for risk alerts.
    """
    __tablename__ = "monitored_addresses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    wallet_address = Column(String, nullable=False, index=True)
    label = Column(String, nullable=True)
    custom_hf_threshold = Column(Float, CheckConstraint('custom_hf_threshold >= 0'), nullable=True)
    notify_on_all_changes = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    added_at = Column(DateTime, server_default=func.now())
    last_checked = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="monitored_addresses")
    position_snapshots = relationship(
        "PositionSnapshot", 
        back_populates="monitored_address",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index('idx_monitored_user_address', 'user_id', 'wallet_address'),
        Index('idx_monitored_active', 'user_id', 'is_active'),
    )

    def __repr__(self):
        return f"<MonitoredAddress(address={self.wallet_address[:8]}, user={self.user_id})>"


class PositionSnapshot(Base):
    """
    Historical snapshot of a monitored address's position.
    
    Used for tracking changes and generating alerts.
    """
    __tablename__ = "position_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    monitored_address_id = Column(Integer, ForeignKey("monitored_addresses.id"))
    wallet_address = Column(String, nullable=False, index=True)
    chain = Column(String, nullable=False)
    total_collateral_usd = Column(Float, CheckConstraint('total_collateral_usd >= 0'))
    total_debt_usd = Column(Float, CheckConstraint('total_debt_usd >= 0'))
    health_factor = Column(Float, CheckConstraint('health_factor >= 0'))
    ltv_ratio = Column(Float, CheckConstraint('ltv_ratio >= 0 AND ltv_ratio <= 1'))
    risk_category = Column(Enum(RiskCategory))
    assets_breakdown = Column(JSON)
    snapshot_time = Column(DateTime, server_default=func.now(), index=True)

    monitored_address = relationship("MonitoredAddress", back_populates="position_snapshots")

    __table_args__ = (
        Index('idx_snapshot_address_time', 'wallet_address', 'snapshot_time'),
        Index('idx_snapshot_risk_time', 'risk_category', 'snapshot_time'),
    )

    def __repr__(self):
        return f"<PositionSnapshot(address={self.wallet_address[:8]}, hf={self.health_factor:.2f})>"


class AlertHistory(Base):
    """
    Historical record of all alerts sent to users.
    
    Used for delivery tracking and analytics.
    """
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
    delivery_status = Column(Enum(DeliveryStatus), default=DeliveryStatus.PENDING)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    delivered_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="alert_history")

    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_alert_user_created', 'user_id', 'created_at'),
        Index('idx_alert_delivery_status', 'user_id', 'delivery_status'),
        Index('idx_alert_delivered_at', 'delivered_at'),
        Index('idx_alert_type_created', 'alert_type', 'created_at'),
    )

    def __repr__(self):
        return f"<AlertHistory(user={self.user_id}, type={self.alert_type}, status={self.delivery_status})>"


class RiskMetricHistory(Base):
    """
    System-wide risk metrics tracked over time.
    
    Used for protocol-level risk analysis and dashboards.
    """
    __tablename__ = "risk_metric_history"

    id = Column(Integer, primary_key=True, index=True)
    chain = Column(String, nullable=False, index=True)
    total_positions = Column(Integer, CheckConstraint('total_positions >= 0'))
    risky_positions = Column(Integer, CheckConstraint('risky_positions >= 0'))
    critical_positions = Column(Integer, CheckConstraint('critical_positions >= 0'))
    total_collateral_usd = Column(Float, CheckConstraint('total_collateral_usd >= 0'))
    total_debt_usd = Column(Float, CheckConstraint('total_debt_usd >= 0'))
    protocol_ltv = Column(Float, CheckConstraint('protocol_ltv >= 0 AND protocol_ltv <= 1'))
    average_health_factor = Column(Float, CheckConstraint('average_health_factor >= 0'))
    liquidations_24h = Column(Integer, CheckConstraint('liquidations_24h >= 0'))
    liquidation_volume_24h_usd = Column(Float, CheckConstraint('liquidation_volume_24h_usd >= 0'))
    recorded_at = Column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index('idx_risk_chain_recorded', 'chain', 'recorded_at'),
        Index('idx_risk_recorded_chain', 'recorded_at', 'chain'),
    )

    def __repr__(self):
        return f"<RiskMetricHistory(chain={self.chain}, recorded={self.recorded_at})>"


class PriceMovement(Base):
    """
    Tracks price movements and their impact on positions.
    
    Used for assessing market risk and potential liquidations.
    """
    __tablename__ = "price_movements"

    id = Column(Integer, primary_key=True, index=True)
    token_symbol = Column(String, nullable=False, index=True)
    token_address = Column(String, nullable=False)
    chain = Column(String, nullable=False)
    price_usd = Column(Float, CheckConstraint('price_usd >= 0'), nullable=False)
    previous_price_usd = Column(Float, CheckConstraint('previous_price_usd >= 0'), nullable=False)
    change_percent = Column(Float, nullable=False)
    positions_affected = Column(Integer, CheckConstraint('positions_affected >= 0'), default=0)
    positions_at_risk = Column(Integer, CheckConstraint('positions_at_risk >= 0'), default=0)
    recorded_at = Column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index('idx_price_token_recorded', 'token_symbol', 'recorded_at'),
        Index('idx_price_chain_recorded', 'chain', 'recorded_at'),
        Index('idx_price_change_recorded', 'change_percent', 'recorded_at'),
    )

    def __repr__(self):
        return f"<PriceMovement(symbol={self.token_symbol}, change={self.change_percent:.2f}%)>"

    @validates('change_percent')
    def validate_change_percent(self, key, value):
        # Allow both positive and negative changes
        if value is None:
            raise ValueError("Change percent cannot be None")
        return value


# ============================================================
# DATABASE UTILITIES
# ============================================================

def init_db() -> None:
    """Initialize database - creates all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")

def get_db():
    """Dependency for FastAPI - yields database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()