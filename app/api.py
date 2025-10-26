"""
Consolidated API with All Endpoints - FIXED VERSION
All critical fixes applied + request not defined errors fixed
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy import func, desc, case, and_, text
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from collections import defaultdict, Counter
from sqlalchemy.pool import StaticPool
import os
import pandas as pd
import logging
from collections import defaultdict
from fastapi import HTTPException
from pydantic import BaseModel
import json
import time
from web3 import Web3
# Import database models
from .db_models import SessionLocal, Reserve, Position, LiquidationHistory, AnalysisSnapshot
from .db_models import User, AlertSubscription, MonitoredAddress, AlertHistory
from .db_models import PositionSnapshot, RiskMetricHistory, PriceMovement

# Import services
from .rpc_reserve_fetcher import AaveRPCReserveFetcher
from .price_fetcher import EnhancedPriceFetcher
from .portfolio_tracker_service import PortfolioTrackerService
from .alert_service import AlertService
from .config import get_settings

from fastapi import Query, Depends

SUPPORTED_CHAINS = ['ethereum', 'avalanche', 'polygon', 'arbitrum', 'optimism', 'base']


def verify_admin_password(password: str = Query(..., description="Admin password")):
    """
    Dependency to verify admin password
    Use with Depends() in any endpoint that needs protection
    """
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change_me_in_production")
    
    if not ADMIN_PASSWORD or ADMIN_PASSWORD == "change_me_in_production":
        raise HTTPException(
            status_code=500, 
            detail="Admin password not configured. Set ADMIN_PASSWORD environment variable."
        )
    
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    
    return True

logger = logging.getLogger(__name__)

# Create two routers - one for v1, one for v2
router_v1 = APIRouter(prefix="/api", tags=["API v1"])
router_v2 = APIRouter(prefix="/api/v2", tags=["User Alerts & Portfolio"])

# Fallback liquidation thresholds
FALLBACK_LIQUIDATION_THRESHOLDS = {
    'WETH': 0.825, 'WBTC': 0.70, 'USDC': 0.875, 'DAI': 0.77,
    'USDT': 0.80, 'LINK': 0.70, 'AAVE': 0.66, 'UNI': 0.70,
    'WMATIC': 0.70, 'stETH': 0.825, 'wstETH': 0.825,
    'MATIC': 0.70, 'MaticX': 0.70, 'WPOL': 0.70,
    'ARB': 0.70, 'OP': 0.70,
    'AVAX': 0.70, 'WAVAX': 0.70,
    'CELO': 0.70, 'cUSD': 0.85, 'cEUR': 0.85, 'cREAL': 0.85,
    'USDC.e': 0.875,
    'PT-sUSDE-27NOV2025': 0.75,
    'DEFAULT': 0.75
}

# Initialize services (will be done in main app startup)
portfolio_service = None
alert_service = None

def init_services():
    """Initialize services - call this from main app startup"""
    global portfolio_service, alert_service
    portfolio_service = PortfolioTrackerService()
    alert_service = AlertService()

# FIX 1: Clean token symbols helper
def clean_token_symbol(symbol):
    """Clean garbled token symbols"""
    if not symbol:
        return "UNKNOWN"
    # Remove null bytes and non-printable characters
    cleaned = ''.join(char for char in symbol if char.isprintable() and char != '\x00')
    return cleaned.strip() or "UNKNOWN"

# FIX 9: Round all metrics helper
def round_metrics(data, exclude_keys=None):
    """Recursively round float values to 2 decimals"""
    exclude_keys = exclude_keys or ['token_amount', 'collateral_amount', 'debt_amount', 
                                     'total_collateral_seized', 'total_debt_normalized']
    
    if isinstance(data, dict):
        return {
            k: round_metrics(v, exclude_keys) if k not in exclude_keys else v
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [round_metrics(item, exclude_keys) for item in data]
    elif isinstance(data, float):
        return round(data, 2)
    return data

# FIX 2: Risk category calculation helper
def get_risk_category(health_factor):
    """Calculate risk category from health factor"""
    if not health_factor or health_factor >= 999:
        return "SAFE"
    elif health_factor >= 2.0:
        return "SAFE"
    elif health_factor >= 1.5:
        return "LOW_RISK"
    elif health_factor >= 1.3:
        return "MEDIUM_RISK"
    elif health_factor >= 1.1:
        return "HIGH_RISK"
    elif health_factor >= 1.0:
        return "CRITICAL"
    else:
        return "LIQUIDATION_IMMINENT"

# FIX 3: Universal null handler
def safe_display(value, field_type="string", unavailable_text="Avail Soon"):
    """
    Universal null/zero handler for API responses
    
    Args:
        value: The value to check
        field_type: 'string', 'number', 'array'
        unavailable_text: Text to display for unavailable data
    """
    if value is None:
        return unavailable_text if field_type == "string" else None
    
    if field_type == "number" and value == 0:
        return unavailable_text
    
    if field_type == "array" and not value:
        return []
    
    return value
# Add this after the existing helper functions (e.g., after get_risk_category around line 150-200)

def fix_null_fields_in_portfolio_response(portfolio_data):
    """Fix null fields in portfolio response to prevent frontend errors"""
    
    # Fix chains_with_positions
    if 'total_metrics' in portfolio_data:
        if not portfolio_data['total_metrics'].get('chains_with_positions'):
            portfolio_data['total_metrics']['chains_with_positions'] = list(portfolio_data.get('portfolio_data', {}).keys())
    
    # Fix safest_chain in risk_assessment
    if 'cross_chain_risk' in portfolio_data:
        if not portfolio_data['cross_chain_risk'].get('safest_chain'):
            # Find the chain with highest health factor
            chain_hfs = {}
            for chain_name, chain_data in portfolio_data.get('portfolio_data', {}).items():
                hf = chain_data.get('account_data', {}).get('health_factor')
                if hf and hf < 999:  # Exclude infinite HF (collateral-only)
                    chain_hfs[chain_name] = hf
            
            if chain_hfs:
                portfolio_data['cross_chain_risk']['safest_chain'] = max(chain_hfs, key=chain_hfs.get)
            elif portfolio_data.get('portfolio_data'):
                # Fallback: First active chain if no valid HF
                portfolio_data['cross_chain_risk']['safest_chain'] = list(portfolio_data['portfolio_data'].keys())[0]
            else:
                portfolio_data['cross_chain_risk']['safest_chain'] = "N/A"
    
    return portfolio_data

def get_db():
    """
    Database session dependency - FIXED for SQLite thread safety
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        # Remove the try-except around close() - let SQLAlchemy handle it
        db.close()

def get_liquidation_threshold(db: Session, chain: str, token_symbol: str) -> tuple:
    """Get liquidation threshold with fallback logic"""
    reserve = db.query(Reserve).filter(
        Reserve.chain == chain,
        Reserve.token_symbol == token_symbol,
        Reserve.is_active == True
    ).order_by(Reserve.query_time.desc()).first()
    
    if reserve and reserve.liquidation_threshold:
        return (reserve.liquidation_threshold, 'reserve')
    
    if token_symbol in FALLBACK_LIQUIDATION_THRESHOLDS:
        return (FALLBACK_LIQUIDATION_THRESHOLDS[token_symbol], 'fallback')
    
    logger.warning(f"No LT found for {token_symbol} on {chain}, using default 0.75")
    return (FALLBACK_LIQUIDATION_THRESHOLDS['DEFAULT'], 'default')

# ==================== PYDANTIC MODELS ====================

class RefreshRequest(BaseModel):
    chains: Optional[List[str]] = None
    refresh_reserves: bool = True
    refresh_positions: bool = True
    refresh_liquidations: bool = True
    prices_only: bool = False

class UserCreate(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    telegram_chat_id: Optional[str] = None
    slack_webhook_url: Optional[str] = None

class AlertSubscriptionCreate(BaseModel):
    channel_email: bool = False
    channel_telegram: bool = False
    channel_slack: bool = False
    health_factor_threshold: float = 1.5
    ltv_threshold: float = 0.75
    alert_frequency_hours: int = 1
    monitored_chains: List[str] = []
    minimum_risk_level: str = "MEDIUM_RISK"

class MonitoredAddressCreate(BaseModel):
    wallet_address: str
    label: Optional[str] = None
    custom_hf_threshold: Optional[float] = None
    notify_on_all_changes: bool = False

class PortfolioRequest(BaseModel):
    wallet_address: str
    chains: Optional[List[str]] = None

# ==================== V1 ENDPOINTS ====================

@router_v1.get("/")
async def root():
    return {
        "message": "DeFi Liquidation Risk API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "v1": "/api",
            "v2": "/api/v2"
        }
    }

@router_v1.get("/health")
async def health_check():
    return {"status": "healthy"}

# FIX 4: Fixed chain_filter display
@router_v1.get("/reserves/rpc")
async def get_rpc_reserves(
    chains: Optional[str] = None,
    active_only: bool = True,
    limit: Optional[int] = 100,
    latest_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get RPC-sourced reserve data"""
    try:
        query = db.query(Reserve)
        
        chain_list = None
        if chains:
            chain_list = [c.strip().lower() for c in chains.split(',')]
            query = query.filter(Reserve.chain.in_(chain_list))
        
        if active_only:
            query = query.filter(Reserve.is_active == True)
        
        if latest_only:
            subq = db.query(
                Reserve.chain,
                Reserve.token_address,
                func.max(Reserve.query_time).label('max_time')
            ).group_by(Reserve.chain, Reserve.token_address).subquery()
            
            query = query.join(
                subq,
                and_(
                    Reserve.chain == subq.c.chain,
                    Reserve.token_address == subq.c.token_address,
                    Reserve.query_time == subq.c.max_time
                )
            )
        else:
            query = query.order_by(desc(Reserve.query_time))
        
        if limit:
            query = query.limit(limit)
        
        reserves = query.all()
        
        return {
            "count": len(reserves),
            "chain_filter": chain_list if chains else None,  # FIXED
            "reserves": [
                {
                    "chain": r.chain,
                    "token_symbol": r.token_symbol,
                    "token_address": r.token_address,
                     "ltv": round(r.ltv or 0, 4),
                    "liquidation_threshold": round(r.liquidation_threshold or 0, 4),
                    "liquidation_bonus": round(r.liquidation_bonus or 0, 4),
                    "is_active": r.is_active,
                    "is_frozen": r.is_frozen,
                    "borrowing_enabled": r.borrowing_enabled,
                    "price_usd": round(r.price_usd or 0, 2),
                    "query_time": r.query_time
                }
                for r in reserves
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reserves: {str(e)}")

# Create a global executor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=3)

async def _fetch_reserves_background(db_session, chains, price_fetcher):
    """Background task to fetch reserves (runs in thread pool) - FIXED with clean_token_symbol"""
    try:
        logger.info(f"🔄 Background: Fetching reserves for chains: {chains or 'all'}")
        
        loop = asyncio.get_event_loop()
        
        def fetch_sync():
            from .rpc_reserve_fetcher import AaveRPCReserveFetcher
            rpc_fetcher = AaveRPCReserveFetcher(price_fetcher=price_fetcher)
            return rpc_fetcher.fetch_all_chains(chains=chains)
        
        all_data = await loop.run_in_executor(executor, fetch_sync)
        
        if not all_data:
            logger.warning("❌ Background: No reserve data returned")
            return 0
        
        stored_count = 0
        for chain, df in all_data.items():
            if df is None or df.empty:
                continue
                
            logger.info(f"Background: Storing {len(df)} reserves from {chain}")
            
            for _, row in df.iterrows():
                try:
                    reserve = Reserve(
                        chain=row.get('chain'),
                        token_address=row.get('token_address'),
                        token_symbol=clean_token_symbol(row.get('token_symbol')),  # FIX 1: CLEANED
                        token_name=row.get('token_name'),
                        decimals=row.get('decimals'),
                        liquidity_rate=row.get('liquidity_rate'),
                        variable_borrow_rate=row.get('variable_borrow_rate'),
                        stable_borrow_rate=row.get('stable_borrow_rate'),
                        supply_apy=row.get('supply_apy'),
                        borrow_apy=row.get('borrow_apy'),
                        ltv=row.get('ltv'),
                        liquidation_threshold=row.get('liquidation_threshold'),
                        liquidation_bonus=row.get('liquidation_bonus'),
                        is_active=row.get('is_active', True),
                        is_frozen=row.get('is_frozen', False),
                        borrowing_enabled=row.get('borrowing_enabled', True),
                        stable_borrowing_enabled=row.get('stable_borrowing_enabled', False),
                        liquidity_index=row.get('liquidity_index'),
                        variable_borrow_index=row.get('variable_borrow_index'),
                        atoken_address=row.get('atoken_address'),
                        variable_debt_token_address=row.get('variable_debt_token_address'),
                        price_usd=row.get('price_usd', 0.0),
                        price_available=row.get('price_available', False),
                        last_update_timestamp=row.get('last_update_timestamp'),
                        query_time=row.get('query_time', datetime.now(timezone.utc))
                    )
                    db_session.add(reserve)
                    stored_count += 1
                except Exception as row_error:
                    logger.error(f"Failed to store reserve: {row_error}")
                    continue
        
        if stored_count > 0:
            db_session.commit()
            logger.info(f"✅ Background: Stored {stored_count} reserves")
        
        return stored_count
        
    except Exception as e:
        logger.error(f"❌ Background reserve fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return 0

@router_v1.post("/data/refresh")
async def unified_data_refresh(
    password: str = Query(..., description="Admin password"),
    _verified: bool = Depends(verify_admin_password),
    refresh_reserves: bool = Query(False, description="✅ Refresh reserve data from blockchain (background, ~20 min)"),
    refresh_positions: bool = Query(False, description="✅ Refresh position data from Dune Analytics"),
    refresh_liquidations: bool = Query(False, description="✅ Refresh liquidation history from Dune Analytics"),
    prices_only: bool = Query(False, description="🔄 Only update prices for existing reserves"),
    chains: Optional[List[str]] = Query(None, description="Specific chains to refresh (leave empty for all)"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    ⚠️ Requires admin password
    🔄 CHECKBOX-BASED DATA REFRESH ENDPOINT
    
    Check the boxes for what you want to refresh:
    - Reserve data runs in background (~20 minutes)
    - Positions and liquidations are immediate
    - Prices only updates existing reserve prices quickly
    
    Examples:
    - Check "refresh_reserves" + select chains = Fetch reserve data for specific chains
    - Check "refresh_positions" + "refresh_liquidations" = Quick update of positions & liquidations
    - Check "prices_only" = Fast price update for existing reserves
    """
    try:
        from dune_client.client import DuneClient
        
        settings = get_settings()
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chains": chains or "all",
            "reserves": None,
            "positions": None,
            "liquidations": None,
            "prices": None,
            "errors": []
        }
        
        # Validate at least one option selected
        if not any([refresh_reserves, refresh_positions, refresh_liquidations, prices_only]):
            return {
                "status": "error",
                "message": "Please select at least one refresh option",
                "options": {
                    "refresh_reserves": False,
                    "refresh_positions": False,
                    "refresh_liquidations": False,
                    "prices_only": False
                }
            }
        
        logger.info(f"🔄 Refresh request: reserves={refresh_reserves}, positions={refresh_positions}, liquidations={refresh_liquidations}, prices_only={prices_only}")
        logger.info("Initializing price fetcher...")
        price_fetcher = EnhancedPriceFetcher(
            api_key=getattr(settings, 'COINGECKO_API_KEY', None)
        )
        
        # ========== 1. RESERVES - START IN BACKGROUND ==========
        if refresh_reserves and not prices_only:
            logger.info("🔄 Starting reserve refresh in background...")
            
            chains_to_fetch = None
            if chains:
                chains_to_fetch = [c.lower() for c in chains if c and c.lower() not in ['string', 'all', 'none']]
                if not chains_to_fetch:
                    chains_to_fetch = None
            
            background_tasks.add_task(
                _fetch_reserves_background,
                db,
                chains_to_fetch,
                price_fetcher
            )
                        
            results["reserves"] = {
                "status": "processing",
                "message": "Reserve fetch started in background (~20 min). Check /api/reserves/rpc later.",
                "chains_requested": chains_to_fetch or "all"
            }
                    
        # ========== 2. POSITIONS - IMMEDIATE ==========
        if refresh_positions and not prices_only:
            logger.info("🔄 Refreshing position data from Dune...")
            try:
                dune_key = os.getenv("DUNE_API_KEY_CURRENT_POSITION")
                if not dune_key:
                    results["positions"] = {
                        "status": "skipped",
                        "reason": "DUNE_API_KEY_CURRENT_POSITION not set"
                    }
                else:
                    dune = DuneClient(api_key=dune_key)
                    query_id = 5780129
                    
                    logger.info(f"Fetching Dune query {query_id}...")
                    
                    rows = None
                    try:
                        result = dune.get_latest_result(query_id)
                        if hasattr(result, 'get_rows'):
                            rows = result.get_rows()
                        elif hasattr(result, 'result') and hasattr(result.result, 'rows'):
                            rows = result.result.rows
                        elif isinstance(result, dict):
                            rows = result.get('result', {}).get('rows', [])
                    except AttributeError:
                        logger.info("Trying dataframe method...")
                        result_df = dune.get_latest_result_dataframe(query_id)
                        if result_df is not None and not result_df.empty:
                            rows = result_df.to_dict('records')
                    
                    if rows and len(rows) > 0:
                        db.query(Position).delete()
                        db.commit()
                        
                        position_count = 0
                        for row in rows:
                            try:
                                # FIX 5: Calculate risk category during position refresh
                                health_factor = row.get('enhanced_health_factor') or row.get('health_factor')
                                risk_category = get_risk_category(health_factor)
                                
                                position = Position(
                                    borrower_address=row.get('borrower_address'),
                                    chain=row.get('chain'),
                                    token_symbol=row.get('token_symbol'),
                                    token_address=row.get('token_address'),
                                    collateral_amount=row.get('collateral_amount'),
                                    debt_amount=row.get('debt_amount'),
                                    health_factor=row.get('health_factor'),
                                    total_collateral_usd=row.get('total_collateral_usd'),
                                    total_debt_usd=row.get('total_debt_usd'),
                                    enhanced_health_factor=row.get('enhanced_health_factor'),
                                    risk_category=risk_category,  # FIXED: Now populated
                                    last_updated=datetime.now(timezone.utc)
                                )
                                db.add(position)
                                position_count += 1
                            except Exception as pos_error:
                                logger.error(f"Failed to add position: {pos_error}")
                                continue
                        
                        db.commit()
                        results["positions"] = {
                            "status": "success",
                            "count": position_count
                        }
                        logger.info(f"✅ Stored {position_count} positions")
                    else:
                        results["positions"] = {
                            "status": "success",
                            "count": 0,
                            "note": "No rows returned from Dune"
                        }
                        
            except Exception as e:
                import traceback
                logger.error(f"Position refresh failed: {e}")
                logger.error(traceback.format_exc())
                results["positions"] = {"status": "error", "error": str(e)}
                results["errors"].append(f"Positions: {str(e)}")
        
        # ========== 3. LIQUIDATIONS - IMMEDIATE (WITH WORKING USD VALUE) ==========
        if refresh_liquidations and not prices_only:
            logger.info("🔄 Refreshing liquidation data from Dune...")
            try:
                dune_key = os.getenv("DUNE_API_KEY_LIQUIDATION_HISTORY")
                if not dune_key:
                    results["liquidations"] = {
                        "status": "skipped",
                        "reason": "DUNE_API_KEY_LIQUIDATION_HISTORY not set"
                    }
                else:
                    dune = DuneClient(api_key=dune_key)
                    query_id = 5774891
                    
                    rows = None
                    try:
                        result = dune.get_latest_result(query_id)
                        if hasattr(result, 'get_rows'):
                            rows = result.get_rows()
                        elif hasattr(result, 'result') and hasattr(result.result, 'rows'):
                            rows = result.result.rows
                        elif isinstance(result, dict):
                            rows = result.get('result', {}).get('rows', [])
                    except AttributeError:
                        result_df = dune.get_latest_result_dataframe(query_id)
                        if result_df is not None and not result_df.empty:
                            rows = result_df.to_dict('records')
                    
                    if rows and len(rows) > 0:
                        db.query(LiquidationHistory).delete()
                        db.commit()
                        
                        liq_count = 0
                        liquidations_to_price = []
                        
                        for row in rows:
                            try:
                                liquidation = LiquidationHistory(
                                    liquidation_date=pd.to_datetime(row.get('liquidation_date')) if row.get('liquidation_date') else datetime.now(timezone.utc),
                                    chain=row.get('chain'),
                                    borrower=row.get('borrower'),
                                    collateral_symbol=row.get('collateral_symbol'),
                                    debt_symbol=row.get('debt_symbol'),
                                    collateral_asset=row.get('collateral_token'),
                                    debt_asset=row.get('debt_token'),
                                    total_collateral_seized=row.get('total_collateral_seized', 0) or 0,
                                    total_debt_normalized=row.get('total_debt_normalized'),
                                    liquidated_collateral_usd=0,  # Will be calculated below
                                    liquidated_debt_usd=0,  # FIX 6: Initialize debt USD field
                                    liquidation_count=row.get('liquidation_count'),
                                    query_time=datetime.now(timezone.utc)
                                )
                                db.add(liquidation)
                                liq_count += 1
                                liquidations_to_price.append(liquidation)
                            except Exception as liq_error:
                                logger.error(f"Failed to add liquidation: {liq_error}")
                                continue
                        
                        if liq_count > 0:
                            db.commit()
                            logger.info(f"✅ Stored {liq_count} liquidations, now calculating USD values...")
                            
                            # ========== USD VALUE CALCULATION (FIXED VERSION) ==========
                            # Get liquidations needing prices
                            liquidations_to_price = db.query(LiquidationHistory).filter(
                                LiquidationHistory.liquidated_collateral_usd == 0,
                                LiquidationHistory.total_collateral_seized > 0
                            ).all()
                            
                            # Group by (symbol, chain, address) - same format as price_fetcher uses
                            collateral_symbol_groups = defaultdict(list)
                            debt_symbol_groups = defaultdict(list)
                            
                            for liq in liquidations_to_price:
                                if liq.collateral_symbol:
                                    key = (liq.collateral_symbol, liq.chain, liq.collateral_asset or "")
                                    collateral_symbol_groups[key].append(liq)
                                
                                if liq.debt_symbol:
                                    key = (liq.debt_symbol, liq.chain, liq.debt_asset or "")
                                    debt_symbol_groups[key].append(liq)
                            
                            logger.info(f"Need prices for {len(collateral_symbol_groups)} collateral tokens and {len(debt_symbol_groups)} debt tokens")
                            
                            # Build token list for batch pricing
                            token_list = []
                            key_to_lookup = {}  # maps price_fetcher key back to our groups
                            
                            # Add collateral tokens
                            for (symbol, chain, address) in collateral_symbol_groups.keys():
                                token_dict = {
                                    'symbol': symbol,
                                    'address': address if address else None,
                                    'chain': chain
                                }
                                token_list.append(token_dict)
                                
                                # Key format that price_fetcher returns: symbol|address|chain
                                price_key = f"{symbol}|{address or ''}|{chain or ''}"
                                key_to_lookup[price_key] = ('collateral', symbol, chain, address)
                            
                            # Add debt tokens
                            for (symbol, chain, address) in debt_symbol_groups.keys():
                                token_dict = {
                                    'symbol': symbol,
                                    'address': address if address else None,
                                    'chain': chain
                                }
                                token_list.append(token_dict)
                                
                                price_key = f"{symbol}|{address or ''}|{chain or ''}"
                                key_to_lookup[price_key] = ('debt', symbol, chain, address)
                            
                            try:
                                # Fetch all prices at once
                                price_data = price_fetcher.get_batch_prices(token_list, progress=None)
                                logger.info(f"Received prices for {len(price_data)} tokens")
                                
                                # Apply prices to liquidations
                                updated_collateral_count = 0
                                updated_debt_count = 0
                                
                                for price_key, price_value in price_data.items():
                                    # Extract price
                                    if isinstance(price_value, dict):
                                        price = price_value.get('price', 0)
                                    elif price_value is not None:
                                        price = float(price_value)
                                    else:
                                        price = 0
                                    
                                    if price and price > 0:
                                        # Find matching liquidations
                                        if price_key in key_to_lookup:
                                            asset_type, symbol, chain, address = key_to_lookup[price_key]
                                            
                                            if asset_type == 'collateral':
                                                group_key = (symbol, chain, address)
                                                if group_key in collateral_symbol_groups:
                                                    for liq in collateral_symbol_groups[group_key]:
                                                        liq.liquidated_collateral_usd = liq.total_collateral_seized * price
                                                        updated_collateral_count += 1
                                            else:  # debt
                                                group_key = (symbol, chain, address)
                                                if group_key in debt_symbol_groups:
                                                    for liq in debt_symbol_groups[group_key]:
                                                        if liq.total_debt_normalized:
                                                            liq.liquidated_debt_usd = liq.total_debt_normalized * price
                                                            updated_debt_count += 1
                                
                                db.commit()
                                logger.info(f"✅ Updated {updated_collateral_count} collateral USD values and {updated_debt_count} debt USD values")
                                
                                results["liquidations"] = {
                                    "status": "success",
                                    "count": liq_count,
                                    "collateral_usd_calculated": updated_collateral_count,
                                    "debt_usd_calculated": updated_debt_count
                                }
                                
                            except Exception as price_error:
                                logger.error(f"Price calculation failed: {price_error}")
                                # Continue without prices rather than failing completely
                                results["liquidations"] = {
                                    "status": "partial_success",
                                    "count": liq_count,
                                    "collateral_usd_calculated": 0,
                                    "debt_usd_calculated": 0,
                                    "price_error": str(price_error)
                                }
                    else:
                        results["liquidations"] = {
                            "status": "success",
                            "count": 0,
                            "note": "No rows returned from Dune"
                        }
                        
            except Exception as e:
                import traceback
                logger.error(f"Liquidation refresh failed: {e}")
                logger.error(traceback.format_exc())
                results["liquidations"] = {"status": "error", "error": str(e)}
                results["errors"].append(f"Liquidations: {str(e)}")
        
        # ========== 4. PRICES ONLY ==========
        if prices_only:
            logger.info("🔄 Refreshing prices only...")
            try:
                reserves = db.query(Reserve).filter(Reserve.is_active == True).all()
                
                reserves_by_chain = defaultdict(list)
                for r in reserves:
                    reserves_by_chain[r.chain].append(r)
                
                updated_count = 0
                for chain, chain_reserves in reserves_by_chain.items():
                    tokens = [
                        {'symbol': r.token_symbol, 'address': r.token_address, 'chain': chain}
                        for r in chain_reserves
                    ]
                    
                    prices = price_fetcher.get_batch_prices(tokens, progress=None)
                    
                    for reserve in chain_reserves:
                        price_data = prices.get(reserve.token_symbol, {})
                        price = price_data.get('price', 0) if isinstance(price_data, dict) else float(price_data or 0)
                        
                        if price > 0:
                            reserve.price_usd = price
                            reserve.price_available = True
                            updated_count += 1
                
                db.commit()
                results["prices"] = {
                    "status": "success",
                    "updated": updated_count,
                    "total": len(reserves)
                }
                logger.info(f"✅ Updated {updated_count} prices")
                
            except Exception as e:
                logger.error(f"Price refresh failed: {e}")
                results["prices"] = {"status": "error", "error": str(e)}
                results["errors"].append(f"Prices: {str(e)}")
        
        summary = {
            "reserves_status": results["reserves"]["status"] if results["reserves"] else "skipped",
            "positions_updated": results["positions"]["count"] if results["positions"] and results["positions"].get("status") == "success" else 0,
            "liquidations_updated": results["liquidations"]["count"] if results["liquidations"] and results["liquidations"].get("status") == "success" else 0,
            "prices_updated": results["prices"]["updated"] if results["prices"] and results["prices"].get("status") == "success" else 0,
            "has_errors": len(results["errors"]) > 0
        }
        
        return {
            "status": "completed_with_errors" if summary["has_errors"] else "completed",
            "message": "Reserves processing in background. Other data refreshed immediately.",
            "results": results,
            "summary": summary
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Unified refresh failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")

@router_v1.get("/reserves/rpc/summary")
async def get_rpc_reserve_summary(db: Session = Depends(get_db)):
    """Get summary statistics of RPC reserve data"""
    try:
        chain_stats = db.query(
            Reserve.chain,
            func.count(Reserve.id).label('total_reserves'),
            func.sum(case((Reserve.is_active == True, 1), else_=0)).label('active_reserves'),
            func.avg(Reserve.supply_apy).label('avg_supply_apy'),
            func.avg(Reserve.borrow_apy).label('avg_borrow_apy'),
            func.max(Reserve.query_time).label('last_update')
        ).group_by(Reserve.chain).all()
        
        return {
            "chains": [
                {
                    "chain": stat.chain,
                    "total_reserves": stat.total_reserves,
                    "active_reserves": stat.active_reserves,
                    "avg_supply_apy": round(float(stat.avg_supply_apy or 0), 2),
                    "avg_borrow_apy": round(float(stat.avg_borrow_apy or 0), 2),
                    "last_update": stat.last_update
                }
                for stat in chain_stats
            ],
            "total_chains": len(chain_stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@router_v1.get("/positions")
async def get_positions(
    limit: Optional[int] = 100,
    risk_category: Optional[str] = None,
    group_by_borrower: bool = False,
    db: Session = Depends(get_db)
):
    """Get all positions with optional grouping by borrower - FIXED risk_category"""
    
    def calculate_risk_category(health_factor: Optional[float]) -> str:
        """Calculate risk category based on health factor"""
        if health_factor is None:
            return "SAFE"
        elif health_factor < 1.0:
            return "CRITICAL"
        elif health_factor < 1.1:
            return "HIGH"
        elif health_factor < 1.5:
            return "MEDIUM"
        else:
            return "SAFE"
    
    try:
        query = db.query(Position)
        
        if risk_category:
            query = query.filter(Position.risk_category == risk_category)
        
        positions = query.limit(limit * 2).all()
        
        if group_by_borrower:
            borrower_groups = defaultdict(list)
            
            for p in positions:
                key = f"{p.borrower_address}_{p.chain}"
                borrower_groups[key].append(p)
            
            result = []
            for key, pos_list in list(borrower_groups.items())[:limit]:
                total_collateral = sum(p.total_collateral_usd or 0 for p in pos_list)
                total_debt = sum(p.total_debt_usd or 0 for p in pos_list)
                
                # Calculate health factors and risk categories
                health_factors = []
                for p in pos_list:
                    if p.enhanced_health_factor and p.enhanced_health_factor < 999:
                        health_factors.append(p.enhanced_health_factor)
                
                min_hf = min(health_factors) if health_factors else None
                
                # FIXED: Calculate risk_category properly
                if min_hf is not None:
                    risk_cat = calculate_risk_category(min_hf)
                elif total_debt > 0:
                    # If we have debt but no health factor, assume medium risk
                    risk_cat = "MEDIUM"
                else:
                    risk_cat = "SAFE"
                
                result.append({
                    "borrower_address": pos_list[0].borrower_address,
                    "chain": pos_list[0].chain,
                    "position_count": len(pos_list),
                    "tokens": list(set([p.token_symbol for p in pos_list if p.token_symbol])),
                    "total_collateral_usd": round(total_collateral, 2),
                    "total_debt_usd": round(total_debt, 2),
                    "lowest_health_factor": round(min_hf, 2) if min_hf else None,
                    "risk_category": risk_cat,  # FIXED: Use calculated risk category
                    "position_type": "collateral_only" if total_debt == 0 else "active_borrowing",
                    "last_updated": max((p.last_updated for p in pos_list if p.last_updated), default=None)
                })
            
            return result
        else:
            # Individual positions
            return [
                {
                    "borrower_address": p.borrower_address,
                    "chain": p.chain,
                    "token_symbol": p.token_symbol,
                    "collateral_amount": p.collateral_amount,
                    "debt_amount": p.debt_amount if p.debt_amount and p.debt_amount > 0 else None,
                    "collateral_usd": round(p.total_collateral_usd or 0, 2),
                    "debt_usd": round(p.total_debt_usd or 0, 2),
                    "enhanced_health_factor": p.enhanced_health_factor,
                    "risk_category": p.risk_category or calculate_risk_category(p.enhanced_health_factor),  # FIXED
                    "last_updated": p.last_updated
                }
                for p in positions[:limit]
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch positions: {str(e)}")

# FIX 8: Fixed risky positions with pagination and proper filtering
@router_v1.get("/positions/risky")
async def get_risky_positions(
    threshold_hf: Optional[float] = 1.5,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get positions with low health factors - FIXED with pagination"""
    try:
        # Get total count first
        total_query = db.query(func.count(Position.id)).filter(
            Position.enhanced_health_factor < threshold_hf,
            Position.enhanced_health_factor > 0,
            Position.total_debt_usd > 0
        )
        total_count = total_query.scalar()
        
        # Get paginated results
        offset = (page - 1) * page_size
        positions = db.query(Position).filter(
            Position.enhanced_health_factor < threshold_hf,
            Position.enhanced_health_factor > 0,
            Position.total_debt_usd > 0
        ).order_by(Position.enhanced_health_factor).offset(offset).limit(page_size).all()
        
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "threshold": threshold_hf,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "count": len(positions),
            "positions": [
                {
                    "borrower_address": p.borrower_address,
                    "token_symbol": p.token_symbol,
                    "health_factor": round(p.enhanced_health_factor, 2),
                    "collateral_usd": round(p.total_collateral_usd or 0, 2),
                    "debt_usd": round(p.total_debt_usd or 0, 2),
                    "risk_category": safe_display(p.risk_category, "string", get_risk_category(p.enhanced_health_factor))  # FIX 3: Safe display with fallback
                }
                for p in positions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

@router_v1.get("/positions_summary")
async def get_positions_summary(
    page: int = 1,
    page_size: int = 100,
    db: Session = Depends(get_db)
):
    """Get summarized view - PAGINATED"""
    try:
        total_count = db.query(Position).count()
        
        if total_count == 0:
            raise HTTPException(status_code=404, detail="No positions found")
        
        offset = (page - 1) * page_size
        positions = db.query(Position).offset(offset).limit(page_size).all()
        
        summary = [
            {
                "borrower": p.borrower_address,
                "chain": p.chain,
                "token_symbol": p.token_symbol,
                "total_collateral_usd": round(p.total_collateral_usd or 0.0, 2),
                "total_debt_usd": round(p.total_debt_usd or 0.0, 2),
                "health_factor": round(p.enhanced_health_factor or 0.0, 4) if p.enhanced_health_factor and p.enhanced_health_factor < 999 else "∞",
            }
            for p in positions
        ]
        
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "page": page,
            "page_size": page_size,
            "total_positions": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "positions": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
    
# ==================== LIQUIDATION HISTORY ====================

@router_v1.get("/liquidation-history")
async def get_liquidation_history(
    limit: Optional[int] = 100,
    chain: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get liquidation history with USD values - FIXED"""
    try:
        query = db.query(LiquidationHistory)
        
        if chain:
            query = query.filter(LiquidationHistory.chain == chain)
        
        liquidations = query.order_by(
            LiquidationHistory.liquidation_date.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": liq.id,
                "liquidation_date": liq.liquidation_date,
                "chain": liq.chain,
                "collateral_symbol": liq.collateral_symbol,
                "debt_symbol": liq.debt_symbol,
                "total_collateral_seized": round(liq.total_collateral_seized or 0, 6),
                "total_debt_normalized": round(liq.total_debt_normalized or 0, 6),
                "collateral_seized_usd": round(liq.liquidated_collateral_usd or 0, 2),
                "debt_repaid_usd": "Avail Soon" if not liq.liquidated_debt_usd or liq.liquidated_debt_usd == 0 else round(liq.liquidated_debt_usd, 2)  # FIXED
            }
            for liq in liquidations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
    
# FIX 9: Fixed average_health_factor calculation
@router_v1.get("/protocol_risk_summary")
async def get_protocol_risk_summary(
    chains: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Comprehensive protocol-level risk metrics - FIXED"""
    try:
        chain_filter = [c.strip().lower() for c in chains.split(',')] if chains else None
        
        position_query = db.query(Position)
        if chain_filter:
            position_query = position_query.filter(Position.chain.in_(chain_filter))
        positions = position_query.all()
        
        if not positions:
            return {
                "total_collateral_usd": 0,
                "total_debt_usd": 0,
                "average_health_factor": None,
                "at_risk_value_usd": 0,
                "chains_analyzed": chain_filter or [],
                "lt_coverage": {"reserve": 0, "fallback": 0, "default": 0}
            }
        
        lt_sources = {"reserve": 0, "fallback": 0, "default": 0}
        
        for pos in positions:
            _, lt_source = get_liquidation_threshold(db, pos.chain, pos.token_symbol)
            lt_sources[lt_source] += 1
        
        total_collateral = sum(p.total_collateral_usd or 0 for p in positions)
        total_debt = sum(p.total_debt_usd or 0 for p in positions)
        
        # FIX 9: Fixed average HF calculation
        valid_hfs = [p.enhanced_health_factor for p in positions 
                     if p.enhanced_health_factor and 0 < p.enhanced_health_factor < 999]
        avg_hf = round(sum(valid_hfs) / len(valid_hfs), 2) if valid_hfs else None  # None instead of 0
        
        at_risk = [p for p in positions if p.enhanced_health_factor and p.enhanced_health_factor < 1.5]
        at_risk_value = sum(p.total_collateral_usd or 0 for p in at_risk)
        
        return round_metrics({
            "total_collateral_usd": total_collateral,
            "total_debt_usd": total_debt,
            "protocol_ltv": (total_debt / total_collateral) if total_collateral > 0 else 0,
            "average_health_factor": avg_hf,
            "at_risk_value_usd": at_risk_value,
            "at_risk_percentage": ((at_risk_value / total_collateral * 100) if total_collateral > 0 else 0),
            "chains_analyzed": chain_filter or list(set(p.chain for p in positions if p.chain)),
            "lt_coverage": {
                "from_reserve_data": lt_sources['reserve'],
                "from_fallback": lt_sources['fallback'],
                "from_default": lt_sources['default'],
                "reserve_coverage_pct": (lt_sources['reserve'] / len(positions) * 100)
            },
            "timestamp": datetime.now(timezone.utc)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

# ==================== CHAIN MANAGEMENT ====================

@router_v1.get("/chains/available")
async def get_available_chains(db: Session = Depends(get_db)):
    """Get chains from both reserves and positions - FIXED to filter supported only"""
    try:
        # Get all chains from database
        reserve_chains = set(c[0] for c in db.query(Reserve.chain).distinct().all() if c[0])
        position_chains = set(c[0] for c in db.query(Position.chain).distinct().all() if c[0])
        all_chains_in_db = reserve_chains | position_chains
        
        # FIXED: Filter to only supported chains
        supported_chains_in_db = [c for c in all_chains_in_db if c in SUPPORTED_CHAINS]
        
        details = []
        for chain in sorted(supported_chains_in_db):
            reserve_count = db.query(Reserve).filter(Reserve.chain == chain).count()
            position_count = db.query(Position).filter(Position.chain == chain).count()
            reserve_latest = db.query(func.max(Reserve.query_time)).filter(Reserve.chain == chain).scalar()
            
            details.append({
                "chain": chain,
                "reserve_count": reserve_count,
                "position_count": position_count,
                "has_reserves": reserve_count > 0,
                "has_positions": position_count > 0,
                "last_reserve_update": reserve_latest if reserve_latest else "Not available"
            })
        
        return {
            "chains": sorted(supported_chains_in_db),
            "count": len(supported_chains_in_db),
            "supported_only": True,
            "details": details,
            "note": f"Showing only supported chains: {', '.join(SUPPORTED_CHAINS)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ==================== DEEP INSIGHTS ====================

@router_v1.get("/insights/protocol-health")
async def get_protocol_health_insights(
    chains: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Deep protocol health insights - FIXED to filter unsupported chains"""
    try:
        # FIXED: Default to supported chains only
        chain_filter = None
        if chains:
            chain_filter = [c.strip().lower() for c in chains.split(',')]
        else:
            chain_filter = SUPPORTED_CHAINS
        
        # Filter reserves to supported chains only
        reserve_query = db.query(Reserve).filter(
            Reserve.is_active == True,
            Reserve.chain.in_(chain_filter)
        )
        
        subq = db.query(
            Reserve.chain,
            Reserve.token_address,
            func.max(Reserve.query_time).label('max_time')
        ).filter(
            Reserve.chain.in_(chain_filter)
        ).group_by(Reserve.chain, Reserve.token_address).subquery()
        
        reserves = reserve_query.join(
            subq,
            (Reserve.chain == subq.c.chain) &
            (Reserve.token_address == subq.c.token_address) &
            (Reserve.query_time == subq.c.max_time)
        ).all()
        
        # Filter positions to supported chains
        position_query = db.query(Position).filter(
            Position.chain.in_(chain_filter)
        )
        positions = position_query.all()
        
        # Filter liquidations to supported chains
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        liq_query = db.query(LiquidationHistory).filter(
            LiquidationHistory.liquidation_date >= thirty_days_ago,
            LiquidationHistory.chain.in_(chain_filter)
        )
        liquidations = liq_query.all()
        
        # Calculate insights
        reserve_insights = {
            'total_reserves': len(reserves),
            'high_apy_reserves': len([r for r in reserves if r.borrow_apy and r.borrow_apy > 10]),
            'frozen_reserves': len([r for r in reserves if r.is_frozen]),
            'avg_supply_apy': sum(r.supply_apy or 0 for r in reserves) / len(reserves) if reserves else 0,
            'avg_borrow_apy': sum(r.borrow_apy or 0 for r in reserves) / len(reserves) if reserves else 0
        }
        
        risky_positions = [p for p in positions if p.enhanced_health_factor and p.enhanced_health_factor < 1.5]
        total_collateral = sum(p.total_collateral_usd or 0 for p in positions)
        total_debt = sum(p.total_debt_usd or 0 for p in positions)
        
        position_insights = {
            'total_positions': len(positions),
            'risky_positions': len(risky_positions),
            'total_collateral_usd': total_collateral,
            'total_debt_usd': total_debt,
            'protocol_ltv': (total_debt / total_collateral) if total_collateral > 0 else 0
        }
        
        # FIXED: Only include supported chains in breakdown
        chain_breakdown = {}
        for chain in chain_filter:
            chain_reserves = [r for r in reserves if r.chain == chain]
            chain_positions = [p for p in positions if p.chain == chain]
            chain_liqs = [l for l in liquidations if l.chain == chain]
            
            if chain_reserves or chain_positions or chain_liqs:
                chain_breakdown[chain] = {
                    'reserves': len(chain_reserves),
                    'positions': len(chain_positions),
                    'liquidations_30d': len(chain_liqs),
                    'avg_supply_apy': sum(r.supply_apy or 0 for r in chain_reserves) / len(chain_reserves) if chain_reserves else 0,
                    'total_collateral': sum(p.total_collateral_usd or 0 for p in chain_positions),
                    'risky_positions': len([p for p in chain_positions if p.enhanced_health_factor and p.enhanced_health_factor < 1.5])
                }
        
        return round_metrics({
            'timestamp': datetime.now(timezone.utc),
            'chains_analyzed': chain_filter,
            'supported_chains_only': True,
            'data_sources': {
                'reserves': len(reserves),
                'positions': len(positions),
                'liquidations_30d': len(liquidations)
            },
            'reserve_insights': reserve_insights,
            'position_insights': position_insights,
            'chain_breakdown': chain_breakdown
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")
    
def calculate_health_score(position_insights, reserve_insights, liquidation_insights) -> Dict[str, any]:
    """Calculate overall protocol health score (0-100) with safe division"""
    score = 100
    
    try:
        total_pos = position_insights.get('total_positions', 0)
        if total_pos > 0:
            critical_count = position_insights.get('critical_positions', 0)
            critical_ratio = critical_count / total_pos
            score -= critical_ratio * 40
        
        ltv = position_insights.get('protocol_ltv', 0)
        if ltv and ltv > 0.70:
            score -= min((ltv - 0.70) * 100, 30)
        
        liq_rate = liquidation_insights.get('liquidation_rate_per_day', 0)
        if liq_rate and liq_rate > 5:
            score -= min((liq_rate - 5) * 2, 20)
        
        total_reserves = reserve_insights.get('total_reserves', 0)
        if total_reserves and total_reserves > 0:
            frozen_count = reserve_insights.get('frozen_reserves', 0)
            frozen_ratio = frozen_count / total_reserves
            score -= frozen_ratio * 10
        
    except Exception as e:
        logger.warning(f"Warning in health score calculation: {e}")
    
    score = max(0, min(100, score))
    
    if score >= 80:
        status, color = 'HEALTHY', 'green'
    elif score >= 60:
        status, color = 'MODERATE', 'yellow'
    elif score >= 40:
        status, color = 'AT_RISK', 'orange'
    else:
        status, color = 'CRITICAL', 'red'
    
    return {
        'score': round(score, 2),
        'status': status,
        'color': color,
        'description': f"Protocol health is {status.lower()}"
    }

# ==================== STRESS TESTS ====================

@router_v1.post("/stress-test/custom")
async def run_custom_stress_test(
    price_drops: Dict[str, float],
    scenario_name: Optional[str] = "Custom Scenario",
    db: Session = Depends(get_db)
):
    """Run custom stress test with specific token price drops"""
    try:
        positions = db.query(Position).all()
        
        results = []
        total_at_risk = 0
        total_value_at_risk = 0
        
        for pos in positions:
            drop_pct = price_drops.get(pos.token_symbol, 0) / 100
            stressed_collateral = (pos.total_collateral_usd or 0) * (1 - drop_pct)
            
            if pos.total_debt_usd and pos.total_debt_usd > 0:
                liquidation_threshold, _ = get_liquidation_threshold(db, pos.chain, pos.token_symbol)
                stressed_hf = (stressed_collateral * liquidation_threshold) / pos.total_debt_usd
                
                if stressed_hf < 1.0:
                    total_at_risk += 1
                    total_value_at_risk += stressed_collateral
                    
                    results.append({
                        "borrower": pos.borrower_address[:10] + "..." if pos.borrower_address else "Unknown",
                        "token": pos.token_symbol,
                        "current_hf": pos.enhanced_health_factor,
                        "stressed_hf": round(stressed_hf, 3),
                        "collateral_at_risk": round(stressed_collateral, 2),
                        "price_drop": drop_pct * 100
                    })
        
        return round_metrics({
            "scenario_name": scenario_name,
            "price_drops_applied": price_drops,
            "total_positions": len(positions),
            "positions_at_risk": total_at_risk,
            "percentage_at_risk": ((total_at_risk / len(positions) * 100) if positions else 0),
            "collateral_at_risk_usd": total_value_at_risk,
            "critical_positions": sorted(results, key=lambda x: x["collateral_at_risk"], reverse=True)[:20]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom stress test failed: {str(e)}")

# ==================== BORROWER RISK SIGNALS ====================

@router_v1.get("/borrower_risk_signals")
async def get_borrower_risk_signals(
    threshold: float = 1.5,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Identify borrowers trending toward liquidation"""
    try:
        positions = db.query(Position).filter(
            Position.enhanced_health_factor < threshold,
            Position.total_debt_usd > 0
        ).order_by(Position.enhanced_health_factor).limit(limit).all()
        
        signals = []
        
        for pos in positions:
            lt, lt_source = get_liquidation_threshold(db, pos.chain, pos.token_symbol)
            current_ltv = (pos.total_debt_usd / pos.total_collateral_usd) if pos.total_collateral_usd else 0
            
            if pos.enhanced_health_factor < 1.05:
                urgency = "CRITICAL"
            elif pos.enhanced_health_factor < 1.1:
                urgency = "HIGH"
            elif pos.enhanced_health_factor < 1.3:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"
            
            signals.append({
                "borrower_address": pos.borrower_address[:10] + "..." if pos.borrower_address else "Unknown",
                "chain": pos.chain,
                "current_health_factor": round(pos.enhanced_health_factor, 3) if pos.enhanced_health_factor else 0,
                "borrower_ltv": round(current_ltv, 4),
                "liquidation_threshold": lt,
                "lt_source": lt_source,
                "collateral_usd": round(pos.total_collateral_usd or 0, 2),
                "debt_usd": round(pos.total_debt_usd or 0, 2),
                "primary_collateral": pos.token_symbol,
                "risk_category": safe_display(pos.risk_category, "string", get_risk_category(pos.enhanced_health_factor)),  # FIX 3: Safe display
                "urgency": urgency,
                "distance_to_liquidation": round((pos.enhanced_health_factor - 1.0), 3) if pos.enhanced_health_factor else 0
            })
        
        return {
            "threshold": threshold,
            "risky_borrowers_count": len(signals),
            "signals": signals,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

# ==================== RESERVE RISK METRICS ====================

@router_v1.get("/reserve_risk_metrics")
async def get_reserve_risk_metrics(
    chains: Optional[str] = None,
    min_ltv: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Evaluate which assets are most dangerous to system health - FIXED"""
    try:
        # FIXED: Default to supported chains only
        if chains:
            chain_filter = [c.strip().lower() for c in chains.split(',')]
        else:
            chain_filter = SUPPORTED_CHAINS
        
        subq = db.query(
            Reserve.chain,
            Reserve.token_address,
            func.max(Reserve.query_time).label('max_time')
        ).filter(
            Reserve.chain.in_(chain_filter)
        ).group_by(Reserve.chain, Reserve.token_address).subquery()
        
        reserve_query = db.query(Reserve).join(
            subq,
            (Reserve.chain == subq.c.chain) &
            (Reserve.token_address == subq.c.token_address) &
            (Reserve.query_time == subq.c.max_time)
        ).filter(
            Reserve.is_active == True,
            Reserve.chain.in_(chain_filter)
        )
        
        if min_ltv is not None:
            reserve_query = reserve_query.filter(Reserve.ltv >= min_ltv)
        
        reserves = reserve_query.all()
        
        # Filter positions to supported chains
        position_query = db.query(Position).filter(
            Position.chain.in_(chain_filter)
        )
        positions = position_query.all()
        
        metrics = []
        
        for reserve in reserves:
            if reserve.liquidity_rate and reserve.variable_borrow_rate:
                utilization = reserve.variable_borrow_rate / (reserve.liquidity_rate + reserve.variable_borrow_rate)
            else:
                utilization = 0
            
            asset_positions = [p for p in positions if p.token_symbol == reserve.token_symbol]
            total_exposure = sum(p.total_collateral_usd or 0 for p in asset_positions)
            
            risk_score = 0
            if reserve.is_frozen:
                risk_score += 25
            if reserve.ltv and reserve.ltv > 0.75:
                risk_score += 30
            if utilization > 0.80:
                risk_score += 25
            
            metrics.append({
                "token_symbol": reserve.token_symbol,
                "token_address": reserve.token_address,
                "chain": reserve.chain,
                "utilization_rate": round(utilization, 4),
                "ltv": reserve.ltv,
                "liquidation_threshold": reserve.liquidation_threshold,
                "total_exposure_usd": round(total_exposure, 2),
                "risk_score": risk_score,
                "risk_level": "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 30 else "LOW"
            })
        
        metrics.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return {
            "reserves_analyzed": len(metrics),
            "chains_analyzed": chain_filter,
            "supported_chains_only": True,
            "high_risk_count": len([m for m in metrics if m['risk_level'] == 'HIGH']),
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
    
# ==================== LIQUIDATION TRENDS ====================

@router_v1.get("/liquidation_trends")
async def get_liquidation_trends(
    days: int = 7,
    chains: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Liquidation patterns - FIXED to filter supported chains"""
    try:
        # FIXED: Default to supported chains
        if chains:
            chain_filter = [c.strip().lower() for c in chains.split(',')]
        else:
            chain_filter = SUPPORTED_CHAINS
        
        now = datetime.now()
        cutoff_7d = now - timedelta(days=days)
        cutoff_24h = now - timedelta(days=1)

        liq_query = db.query(LiquidationHistory).filter(
            LiquidationHistory.liquidation_date >= cutoff_7d,
            LiquidationHistory.chain.in_(chain_filter)
        )
        
        liquidations_7d = liq_query.all()
        liquidations_24h = [l for l in liquidations_7d if l.liquidation_date >= cutoff_24h]
        
        total_volume_7d = sum(l.liquidated_collateral_usd or 0 for l in liquidations_7d)
        total_volume_24h = sum(l.liquidated_collateral_usd or 0 for l in liquidations_24h)
        
        return {
            "period_days": days,
            "chains_analyzed": chain_filter,
            "supported_chains_only": True,
            "liquidations_24h": len(liquidations_24h),
            "liquidations_7d": len(liquidations_7d),
            "liquidation_volume_usd_24h": round(total_volume_24h, 2),
            "liquidation_volume_usd_7d": round(total_volume_7d, 2),
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

# ==================== CROSS-CHAIN RISK COMPARISON ====================

@router_v1.get("/crosschain_risk_comparison")
async def get_crosschain_risk_comparison(db: Session = Depends(get_db)):
    """Compare risk metrics across all Aave deployments - FIXED"""
    try:
        # FIXED: Only query supported chains
        positions = db.query(Position).filter(
            Position.chain.in_(SUPPORTED_CHAINS)
        ).all()
        
        subq = db.query(
            Reserve.chain,
            Reserve.token_address,
            func.max(Reserve.query_time).label('max_time')
        ).filter(
            Reserve.chain.in_(SUPPORTED_CHAINS)
        ).group_by(Reserve.chain, Reserve.token_address).subquery()
        
        reserves = db.query(Reserve).join(
            subq,
            (Reserve.chain == subq.c.chain) &
            (Reserve.token_address == subq.c.token_address) &
            (Reserve.query_time == subq.c.max_time)
        ).filter(
            Reserve.is_active == True,
            Reserve.chain.in_(SUPPORTED_CHAINS)
        ).all()
        
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        liquidations = db.query(LiquidationHistory).filter(
            LiquidationHistory.liquidation_date >= week_ago,
            LiquidationHistory.chain.in_(SUPPORTED_CHAINS)
        ).all()
        
        # Only iterate over supported chains
        chains = set(p.chain for p in positions if p.chain in SUPPORTED_CHAINS)
        
        comparison = []
        
        for chain in chains:
            chain_positions = [p for p in positions if p.chain == chain]
            chain_reserves = [r for r in reserves if r.chain == chain]
            chain_liqs = [l for l in liquidations if l.chain == chain]
            
            valid_hfs = [p.enhanced_health_factor for p in chain_positions 
                        if p.enhanced_health_factor and p.enhanced_health_factor < 100]
            avg_hf = sum(valid_hfs) / len(valid_hfs) if valid_hfs else 0
            
            total_collateral = sum(p.total_collateral_usd or 0 for p in chain_positions)
            total_debt = sum(p.total_debt_usd or 0 for p in chain_positions)
            debt_ratio = total_debt / total_collateral if total_collateral > 0 else 0
            
            comparison.append({
                "chain": chain,
                "average_health_factor": round(avg_hf, 3),
                "debt_collateral_ratio": round(debt_ratio, 4),
                "total_positions": len(chain_positions),
                "total_collateral_usd": round(total_collateral, 2),
                "total_debt_usd": round(total_debt, 2),
                "liquidations_7d": len(chain_liqs),
                "safety_score": round((avg_hf / debt_ratio) if debt_ratio > 0 else 100, 2)
            })
        
        comparison.sort(key=lambda x: x['safety_score'], reverse=True)
        
        return {
            "chains_analyzed": len(comparison),
            "supported_chains_only": True,
            "safest_chain": comparison[0]['chain'] if comparison else None,
            "riskiest_chain": comparison[-1]['chain'] if comparison else None,
            "comparison": comparison,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

# ==================== RISK ALERTS FEED ====================

@router_v1.get("/risk_alerts_feed")
async def get_risk_alerts_feed(
    password: str = Query(..., description="Admin password"),
    _verified: bool = Depends(verify_admin_password),
    severity: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Risk alerts feed - ⚠️ Requires admin password"""
    try:
        alerts = []
        
        critical_positions = db.query(Position).filter(
            Position.enhanced_health_factor < 1.05,
            Position.enhanced_health_factor > 0
        ).all()
        
        for pos in critical_positions:
            alerts.append({
                "alert_id": f"HF_CRITICAL_{pos.id}",
                "severity": "CRITICAL",
                "type": "HEALTH_FACTOR",
                "message": f"Position HF at {pos.enhanced_health_factor:.3f} - Liquidation imminent",
                "borrower": pos.borrower_address[:10] + "..." if pos.borrower_address else "Unknown",
                "chain": pos.chain,
                "token": pos.token_symbol,
                "health_factor": round(pos.enhanced_health_factor, 3),
                "collateral_usd": round(pos.total_collateral_usd or 0, 2),
                "action_required": "Monitor for liquidation",
                "timestamp": datetime.now(timezone.utc)
            })
        
        positions = db.query(Position).filter(
            Position.total_debt_usd > 0,
            Position.total_collateral_usd > 0
        ).all()
        
        for pos in positions:
            lt, lt_source = get_liquidation_threshold(db, pos.chain, pos.token_symbol)
            
            current_ltv = (pos.total_debt_usd / pos.total_collateral_usd) if pos.total_collateral_usd else 0
            ltv_threshold = lt * 0.9
            
            if current_ltv > ltv_threshold:
                alerts.append({
                    "alert_id": f"LTV_HIGH_{pos.id}",
                    "severity": "HIGH",
                    "type": "NEAR_MARGIN_CALL",
                    "message": f"LTV at {current_ltv:.1%}, near liquidation threshold",
                    "borrower": pos.borrower_address[:10] + "..." if pos.borrower_address else "Unknown",
                    "chain": pos.chain,
                    "token": pos.token_symbol,
                    "current_ltv": round(current_ltv, 4),
                    "liquidation_threshold": lt,
                    "lt_source": lt_source,
                    "action_required": "Add collateral or repay debt",
                    "timestamp": datetime.now(timezone.utc)
                })
        
        subq = db.query(
            Reserve.chain,
            Reserve.token_address,
            func.max(Reserve.query_time).label('max_time')
        ).group_by(Reserve.chain, Reserve.token_address).subquery()
        
        reserves = db.query(Reserve).join(
            subq,
            (Reserve.chain == subq.c.chain) &
            (Reserve.token_address == subq.c.token_address) &
            (Reserve.query_time == subq.c.max_time)
        ).filter(Reserve.is_active == True).all()
        
        for reserve in reserves:
            if reserve.liquidity_rate and reserve.variable_borrow_rate:
                utilization = reserve.variable_borrow_rate / (reserve.liquidity_rate + reserve.variable_borrow_rate)
                
                if utilization > 0.95:
                    alerts.append({
                        "alert_id": f"UTIL_HIGH_{reserve.id}",
                        "severity": "MEDIUM",
                        "type": "LIQUIDITY_SQUEEZE",
                        "message": f"{reserve.token_symbol} utilization at {utilization:.1%}",
                        "chain": reserve.chain,
                        "token": reserve.token_symbol,
                        "utilization_rate": round(utilization, 4),
                        "action_required": "Monitor liquidity availability",
                        "timestamp": datetime.now(timezone.utc)
                    })
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity.upper()]
        
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        alerts.sort(key=lambda x: severity_order.get(x['severity'], 4))
        alerts = alerts[:limit]
        
        severity_counts = {}
        for alert in alerts:
            sev = alert['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            "total_alerts": len(alerts),
            "severity_counts": severity_counts,
            "alerts": alerts,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

@router_v1.get("/protocol/risky-borrowers")
async def get_risky_borrowers_summary(
    password: str = Query(..., description="Admin password"),
    _verified: bool = Depends(verify_admin_password),
    threshold: float = 1.5,
    chains: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Risky borrowers summary - ⚠️ Requires admin password"""
    try:
        query = db.query(Position).filter(
            Position.enhanced_health_factor < threshold,
            Position.enhanced_health_factor > 0,
            Position.total_debt_usd > 0  # Must have debt
        )
        
        if chains:
            chain_list = [c.strip().lower() for c in chains.split(',')]
            query = query.filter(Position.chain.in_(chain_list))
        
        risky_positions = query.all()
        
        by_chain = {}
        for pos in risky_positions:
            chain = pos.chain
            if chain not in by_chain:
                by_chain[chain] = {
                    'count': 0,
                    'critical_count': 0,
                    'total_at_risk_usd': 0
                }
            
            by_chain[chain]['count'] += 1
            if pos.enhanced_health_factor < 1.1:
                by_chain[chain]['critical_count'] += 1
            by_chain[chain]['total_at_risk_usd'] += round(pos.total_collateral_usd or 0, 2)
        
        return {
            "threshold": threshold,
            "total_risky_borrowers": len(risky_positions),
            "breakdown_by_chain": by_chain,
            "summary_message": f"⚠️ {len(risky_positions)} addresses below {threshold} HF across {len(by_chain)} chains"
        }
        
    except Exception as e:
        logger.error(f"Risky borrowers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router_v1.get("/startup-status")
def startup_status():
    """Check API startup status - FIXED"""
    DB_AVAILABLE = os.getenv("DATABASE_URL") is not None
    
    if not DB_AVAILABLE:
        return {
            "api_loaded": True,
            "database_connected": False,
            "error": "PostgreSQL not running",
            "message": "Start PostgreSQL or set DATABASE_URL"
        }
    
    try:
        from .scheduler import unified_scheduler
        scheduler_running = unified_scheduler.scheduler.running  # FIXED
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
        "services_initialized": portfolio_service is not None
    }

   
# ==================== V2 USER MANAGEMENT ====================

@router_v2.post("/users/register")
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user for alert notifications"""
    try:
        existing = db.query(User).filter(User.username == user_data.username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        if user_data.email:
            email_exists = db.query(User).filter(User.email == user_data.email).first()
            if email_exists:
                raise HTTPException(status_code=400, detail="Email already registered")
        
        user = User(
            username=user_data.username,
            email=user_data.email,
            telegram_chat_id=user_data.telegram_chat_id,
            slack_webhook_url=user_data.slack_webhook_url,
            created_at=datetime.now(timezone.utc)
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return {
            "user_id": user.id,
            "username": user.username,
            "message": "User registered successfully",
            "next_steps": f"Create alert subscription at /api/v2/users/{user.id}/alerts/subscribe"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router_v2.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user details"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "telegram_chat_id": user.telegram_chat_id,
        "has_slack": bool(user.slack_webhook_url),
        "is_active": user.is_active,
        "created_at": user.created_at,
        "monitored_addresses_count": len(user.monitored_addresses),
        "alert_subscriptions_count": len(user.alert_subscriptions)
    }

@router_v2.post("/users/{user_id}/alerts/subscribe")
async def create_alert_subscription(
    user_id: int,
    subscription_data: AlertSubscriptionCreate,
    db: Session = Depends(get_db)
):
    """Create alert subscription for user"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if subscription_data.channel_email and not user.email:
            raise HTTPException(status_code=400, detail="Email not configured for user")
        
        if subscription_data.channel_telegram and not user.telegram_chat_id:
            raise HTTPException(status_code=400, detail="Telegram not configured for user")
        
        if subscription_data.channel_slack and not user.slack_webhook_url:
            raise HTTPException(status_code=400, detail="Slack not configured for user")
        
        subscription = AlertSubscription(
            user_id=user_id,
            channel_email=subscription_data.channel_email,
            channel_telegram=subscription_data.channel_telegram,
            channel_slack=subscription_data.channel_slack,
            health_factor_threshold=subscription_data.health_factor_threshold,
            ltv_threshold=subscription_data.ltv_threshold,
            alert_frequency_hours=subscription_data.alert_frequency_hours,
            monitored_chains=subscription_data.monitored_chains,
            minimum_risk_level=subscription_data.minimum_risk_level,
            created_at=datetime.now(timezone.utc)
        )
        
        db.add(subscription)
        db.commit()
        db.refresh(subscription)
        
        return {
            "subscription_id": subscription.id,
            "user_id": user_id,
            "channels_enabled": {
                "email": subscription.channel_email,
                "telegram": subscription.channel_telegram,
                "slack": subscription.channel_slack
            },
            "thresholds": {
                "health_factor": subscription.health_factor_threshold,
                "ltv": subscription.ltv_threshold
            },
            "message": "Alert subscription created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Subscription failed: {str(e)}")

@router_v2.post("/users/{user_id}/monitored-addresses")
async def add_monitored_address(
    user_id: int,
    address_data: MonitoredAddressCreate,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """Add wallet address to monitor"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        existing = db.query(MonitoredAddress).filter(
            MonitoredAddress.user_id == user_id,
            MonitoredAddress.wallet_address == address_data.wallet_address.lower()
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="Address already monitored")
        
        monitored = MonitoredAddress(
            user_id=user_id,
            wallet_address=address_data.wallet_address.lower(),
            label=address_data.label,
            custom_hf_threshold=address_data.custom_hf_threshold,
            notify_on_all_changes=address_data.notify_on_all_changes,
            added_at=datetime.now(timezone.utc)
        )
        
        db.add(monitored)
        db.commit()
        db.refresh(monitored)
        
        return {
            "monitored_address_id": monitored.id,
            "wallet_address": monitored.wallet_address,
            "label": monitored.label,
            "message": "Address added to monitoring"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to add address: {str(e)}")

"""
FINAL FIXES for Remaining Issues
Apply these changes to your consolidated_api.py
"""

# ============================================================
# FIX 1: Position endpoint - debt_amount showing 0 and risk_category
# Replace the positions endpoint response section
# ============================================================

@router_v1.get("/positions")
async def get_positions(
    limit: Optional[int] = 100,
    risk_category: Optional[str] = None,
    group_by_borrower: bool = False,
    db: Session = Depends(get_db)
):
    """Get all positions with optional grouping by borrower - FIXED"""
    try:
        query = db.query(Position)
        
        if risk_category:
            query = query.filter(Position.risk_category == risk_category)
        
        positions = query.limit(limit * 2).all()
        
        if group_by_borrower:
            borrower_groups = defaultdict(list)
            
            for p in positions:
                key = f"{p.borrower_address}_{p.chain}"
                borrower_groups[key].append(p)
            
            result = []
            for key, pos_list in list(borrower_groups.items())[:limit]:
                total_collateral = sum(p.total_collateral_usd or 0 for p in pos_list)
                total_debt = sum(p.total_debt_usd or 0 for p in pos_list)
                
                if total_debt > 0:
                    valid_hfs = [p.enhanced_health_factor for p in pos_list 
                                if p.enhanced_health_factor and p.enhanced_health_factor < 999]
                    min_hf = min(valid_hfs) if valid_hfs else None
                    risk_cat = next((p.risk_category for p in pos_list 
                                   if p.enhanced_health_factor == min_hf), "SAFE")
                else:
                    min_hf = None
                    risk_cat = "SAFE"
                
                result.append({
                    "borrower_address": pos_list[0].borrower_address,
                    "chain": pos_list[0].chain,
                    "position_count": len(pos_list),
                    "tokens": [p.token_symbol for p in pos_list],
                    "total_collateral_usd": round(total_collateral, 2),
                    "total_debt_usd": round(total_debt, 2),
                    "lowest_health_factor": round(min_hf, 2) if min_hf else None,
                    "risk_category": risk_cat,
                    "position_type": "collateral_only" if total_debt == 0 else "active_borrowing",
                    "last_updated": max((p.last_updated for p in pos_list if p.last_updated), default=None)
                })
            
            return result
        else:
            # FIXED: Don't show debt_amount if it's 0, and calculate risk_category
            return [
                {
                    "borrower_address": p.borrower_address,
                    "chain": p.chain,
                    "token_symbol": p.token_symbol,
                    "collateral_amount": p.collateral_amount,
                    "debt_amount": p.debt_amount if p.debt_amount and p.debt_amount > 0 else "None",  # FIXED
                    "collateral_usd": round(p.total_collateral_usd or 0, 2),
                    "debt_usd": round(p.total_debt_usd or 0, 2),
                    "enhanced_health_factor": p.enhanced_health_factor,
                    "risk_category": p.risk_category or get_risk_category(p.enhanced_health_factor),  # FIXED: Calculate if null
                    "last_updated": p.last_updated
                }
                for p in positions[:limit]
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch positions: {str(e)}")


@router_v1.get("/liquidation-history")
async def get_liquidation_history(
    limit: Optional[int] = 100,
    chain: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get liquidation history with USD values - FIXED"""
    try:
        query = db.query(LiquidationHistory)
        
        if chain:
            query = query.filter(LiquidationHistory.chain == chain)
        
        liquidations = query.order_by(
            LiquidationHistory.liquidation_date.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": liq.id,
                "liquidation_date": liq.liquidation_date,
                "chain": liq.chain,
                "collateral_symbol": liq.collateral_symbol,
                "debt_symbol": liq.debt_symbol,
                "total_collateral_seized": round(liq.total_collateral_seized or 0, 6),
                "total_debt_normalized": round(liq.total_debt_normalized or 0, 6),
                "collateral_seized_usd": round(liq.liquidated_collateral_usd or 0, 2),
                "debt_repaid_usd": "Avail Soon" if not liq.liquidated_debt_usd or liq.liquidated_debt_usd == 0 else round(liq.liquidated_debt_usd, 2)  # FIXED
            }
            for liq in liquidations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

"""
Add these endpoints to your consolidated_api.py file
Place them in the router_v1 section (around line 500-600, after existing endpoints)
"""

# ============================================================
# MISSING ENDPOINT 1: Quick Stats for Overview
# ============================================================
@router_v1.get("/data/quick-stats")
async def get_quick_stats(db: Session = Depends(get_db)):
    """
    Quick statistics for dashboard overview
    Returns: Total positions, collateral, at-risk positions, critical positions
    """
    try:
        # Total positions count
        total_positions = db.query(Position).count()
        
        # Total collateral
        total_collateral = db.query(
            func.sum(Position.total_collateral_usd)
        ).scalar() or 0
        
        # At-risk positions (HF < 1.5 and has debt)
        at_risk_positions = db.query(Position).filter(
            Position.enhanced_health_factor < 1.5,
            Position.enhanced_health_factor > 0,
            Position.total_debt_usd > 0
        ).count()
        
        # Critical positions (HF < 1.1 and has debt)
        critical_positions = db.query(Position).filter(
            Position.enhanced_health_factor < 1.1,
            Position.enhanced_health_factor > 0,
            Position.total_debt_usd > 0
        ).count()
        
        return {
            "positions": total_positions,
            "total_collateral_usd": round(total_collateral, 2),
            "at_risk_positions": at_risk_positions,
            "critical_positions": critical_positions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Quick stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MISSING ENDPOINT 2: Admin Cache Clear
# ============================================================

@router_v1.post("/admin/cache/clear")
async def admin_clear_cache(
    password: str = Query(..., description="Admin password"),
    db: Session = Depends(get_db)
):
    """
    Clear all cached data (requires admin password)
    Clears portfolio cache and any other cached data
    """
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change_me_in_production")
    
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    
    try:
        cleared_count = 0
        
        # Clear portfolio service cache if available
        if portfolio_service and hasattr(portfolio_service, 'cache'):
            cache_dir = portfolio_service.cache.cache_dir
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(cache_dir, filename))
                        cleared_count += 1
        
        return {
            "success": True,
            "message": f"Cache cleared successfully ({cleared_count} entries removed)",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MISSING ENDPOINT 3: Admin Settings View
# ============================================================
@router_v1.get("/admin/settings")
async def get_admin_settings(
    password: str = Query(..., description="Admin password")
):
    """
    Get current system settings and status
    Requires admin password for security
    """
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change_me_in_production")
    
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    
    try:
        # Get database connection status
        db_connected = False
        try:
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db_connected = True
            db.close()
        except:
            pass
        
        return {
            "cache_ttl_minutes": 15,
            "supported_chains": list(AaveRPCReserveFetcher.RPC_ENDPOINTS.keys()),
            "database_connected": db_connected,
            "services_running": {
                "portfolio_service": portfolio_service is not None,
                "alert_service": alert_service is not None
            },
            "api_version": "2.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENHANCED ENDPOINT: Positions with Token Address
# Already exists but needs token_address in response
# Update the existing /positions endpoint response to include:
# ============================================================
"""
In the existing @router_v1.get("/positions") endpoint, update the return statement to:

return [
    {
        "borrower_address": p.borrower_address,
        "chain": p.chain,
        "token_symbol": p.token_symbol,
        "token_address": p.token_address,  # ADD THIS LINE
        "collateral_amount": p.collateral_amount,
        "debt_amount": p.debt_amount if p.debt_amount and p.debt_amount > 0 else None,
        "collateral_usd": round(p.total_collateral_usd or 0, 2),
        "debt_usd": round(p.total_debt_usd or 0, 2),
        "enhanced_health_factor": p.enhanced_health_factor,
        "risk_category": p.risk_category or get_risk_category(p.enhanced_health_factor),
        "last_updated": p.last_updated
    }
    for p in positions[:limit]
]
"""


# ============================================================
# ENHANCED ENDPOINT: Reserves with Token Address
# Already exists but make sure token_address is in response
# ============================================================
"""
In the existing @router_v1.get("/reserves/rpc") endpoint, the response already includes
token_address, so no changes needed. Just verify it's there:

"reserves": [
    {
        "chain": r.chain,
        "token_symbol": r.token_symbol,
        "token_address": r.token_address,  # Should already be here
        "ltv": round(r.ltv or 0, 4),
        ...
    }
]
"""


# ============================================================
# ENHANCED ENDPOINT: Reserve Risk Metrics with Token Address
# Update existing endpoint to include token_address
# ============================================================
"""
In the existing @router_v1.get("/reserve_risk_metrics") endpoint,
update the metrics.append() to include token_address:

metrics.append({
    "token_symbol": reserve.token_symbol,
    "token_address": reserve.token_address,  # ADD THIS LINE
    "chain": reserve.chain,
    "utilization_rate": round(utilization, 4),
    ...
})
"""


# ============================================================
# OPTIONAL: Cache Statistics Endpoint
# ============================================================
@router_v1.get("/cache/stats")
async def get_cache_statistics():
    """
    Get cache statistics (public endpoint, no auth required)
    Shows cache usage and health
    """
    try:
        if not portfolio_service or not hasattr(portfolio_service, 'cache'):
            return {
                "cache_enabled": False,
                "message": "Cache not available"
            }
        
        cache_dir = portfolio_service.cache.cache_dir
        
        if not os.path.exists(cache_dir):
            return {
                "cache_enabled": True,
                "total_cached_entries": 0,
                "cache_size_mb": 0,
                "cache_directory": cache_dir
            }
        
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        
        total_size = 0
        valid_entries = 0
        expired_entries = 0
        
        for filename in cache_files:
            filepath = os.path.join(cache_dir, filename)
            total_size += os.path.getsize(filepath)
            
            try:
                with open(filepath, 'r') as f:
                    cached_data = json.load(f)
                cached_time = datetime.fromisoformat(cached_data['cached_at'])
                
                if datetime.now(timezone.utc) - cached_time > portfolio_service.cache.cache_ttl:
                    expired_entries += 1
                else:
                    valid_entries += 1
            except:
                expired_entries += 1
        
        return {
            "cache_enabled": True,
            "total_cached_entries": len(cache_files),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_directory": cache_dir,
            "cache_ttl_minutes": portfolio_service.cache.cache_ttl.total_seconds() / 60,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return {
            "cache_enabled": False,
            "error": str(e)
        }
# ============================================================
# COMPLETE UPDATED VIEW_PORTFOLIO ENDPOINT WITH ALL FIXES
# ============================================================
def validate_wallet_address(address: str) -> str:
    """Validate and normalize wallet address"""
    # Remove whitespace
    address = address.strip()
    
    # Check format
    if not address.startswith('0x'):
        raise ValueError(f"Address must start with 0x: {address}")
    
    # Check length (should be 42 chars: 0x + 40 hex)
    if len(address) != 42:
        raise ValueError(f"Invalid address length {len(address)}, should be 42: {address}")
    
    # Check if valid hex
    try:
        int(address[2:], 16)
    except ValueError:
        raise ValueError(f"Address contains invalid hex characters: {address}")
    
    # Return checksummed address
    try:
        return Web3.to_checksum_address(address)
    except Exception as e:
        raise ValueError(f"Invalid address format: {address}, error: {e}")

@router_v2.post("/portfolio/view-fast")
async def view_portfolio_fast(request: PortfolioRequest, db: Session = Depends(get_db)):
    """
    Fast portfolio view - ONLY account data, NO asset breakdown
    
    Returns in ~2-3 seconds instead of 40+ seconds
    Perfect for dashboards that only need totals and health factor
    """
    try:
        if not portfolio_service:
            raise HTTPException(status_code=503, detail="Portfolio service not initialized")
        
        # Validate address first
        try:
            validated_address = validate_wallet_address(request.wallet_address)
            logger.info(f"⚡ Fast fetch for {validated_address}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid wallet address: {str(e)}")
        
        chains = request.chains or ['ethereum']
        
        start_time = time.time()         
        
        # Get only account data (no asset breakdown)
        chain_details = {}
        total_collateral = 0.0
        total_debt = 0.0
        total_available_borrows = 0.0
        total_net_worth = 0.0
        lowest_hf = None
        active_chains = []
        
        for chain in chains:
            try:
                # ✅ FIX: Use get_working_rpc() method instead of direct RPC endpoint
                logger.info(f"🔍 Checking {chain}...")
                
                # Get working RPC endpoint
                rpc_endpoint = portfolio_service.get_working_rpc(chain)
                logger.info(f"   Using RPC: {rpc_endpoint}")
                
                # Create Web3 instance with timeout
                from web3 import Web3
                w3 = Web3(Web3.HTTPProvider(rpc_endpoint, request_kwargs={'timeout': 15}))
                
                # Verify connection with actual block number check
                if not w3.is_connected():
                    logger.warning(f"⚠️ RPC not connected for {chain}")
                    continue
                
                try:
                    block = w3.eth.block_number
                    logger.info(f"   Connected! Current block: {block}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get block number for {chain}: {e}")
                    continue
                
                # Get account data using validated address
                logger.info(f"   Fetching account data for {validated_address[:10]}...")
                account_data = portfolio_service.get_comprehensive_account_data(
                    w3, chain, validated_address
                )
                
                if not account_data:
                    logger.warning(f"⚠️ No account data returned for {chain}")
                    continue
                
                # Log the raw values
                logger.info(f"   Raw account data: collateral={account_data.get('total_collateral_usd')}, debt={account_data.get('total_debt_usd')}")
                
                has_positions = (
                    account_data.get('total_collateral_usd', 0) > 0 or
                    account_data.get('total_debt_usd', 0) > 0
                )
                
                logger.info(f"   Has positions: {has_positions}")
                
                if has_positions:
                    active_chains.append(chain)
                    total_collateral += account_data['total_collateral_usd']
                    total_debt += account_data['total_debt_usd']
                    total_available_borrows += account_data['available_borrows_usd']
                    total_net_worth += account_data['net_worth_usd']
                    
                    chain_hf = account_data['health_factor']
                    if chain_hf is not None and chain_hf != float('inf'):
                        if lowest_hf is None or chain_hf < lowest_hf:
                            lowest_hf = chain_hf
                    
                    logger.info(f"✅ {chain}: ${account_data['total_collateral_usd']:.2f} collateral, ${account_data['total_debt_usd']:.2f} debt")
                
                # Make account_data JSON serializable
                serializable_account_data = {
                    'total_collateral_usd': round(account_data.get('total_collateral_usd', 0), 2),
                    'total_debt_usd': round(account_data.get('total_debt_usd', 0), 2),
                    'available_borrows_usd': round(account_data.get('available_borrows_usd', 0), 2),
                    'current_liquidation_threshold': round(account_data.get('current_liquidation_threshold', 0), 4),
                    'ltv': round(account_data.get('ltv', 0), 4),
                    'health_factor': (
                        round(chain_hf, 3) 
                        if chain_hf is not None and chain_hf != float('inf') 
                        else None
                    ),
                    'net_worth_usd': round(account_data.get('net_worth_usd', 0), 2),
                    'risk_level': account_data.get('risk_level', 'NO_POSITIONS'),
                    'liquidation_imminent': account_data.get('liquidation_imminent', False)
                }
                
                chain_details[chain] = {
                    "has_positions": has_positions,
                    "account_data": serializable_account_data,
                    "note": "Asset breakdown not included for faster response"
                }
                
            except Exception as e:
                logger.error(f"❌ Error fetching {chain}: {e}", exc_info=True)
                # Add error to chain details for debugging
                chain_details[chain] = {
                    "has_positions": False,
                    "error": str(e),
                    "note": "Failed to fetch data"
                }
                continue
        
        # Calculate metrics
        utilization_ratio = (total_debt / total_collateral * 100) if total_collateral > 0 else 0.0
        
        total_metrics = {
            'total_collateral_usd': round(total_collateral, 2),
            'total_debt_usd': round(total_debt, 2),
            'total_available_borrows_usd': round(total_available_borrows, 2),
            'total_net_worth_usd': round(total_net_worth, 2),
            'lowest_health_factor': round(lowest_hf, 3) if lowest_hf is not None else None,
            'utilization_ratio_percent': round(utilization_ratio, 2),
        }
        
        # Risk assessment
        if lowest_hf is None or total_collateral == 0:
            risk_assessment = {
                'overall_risk_level': 'NO_POSITIONS',
                'risk_score': 0,
                'liquidation_imminent': False,
                'health_factor_status': 'NO_POSITIONS'
            }
        elif lowest_hf < 1.0:
            risk_assessment = {
                'overall_risk_level': 'CRITICAL',
                'risk_score': 100,
                'liquidation_imminent': True,
                'health_factor_status': 'CRITICAL'
            }
        elif lowest_hf < 1.5:
            risk_assessment = {
                'overall_risk_level': 'MEDIUM',
                'risk_score': 50,
                'liquidation_imminent': False,
                'health_factor_status': 'WARNING'
            }
        else:
            risk_assessment = {
                'overall_risk_level': 'LOW',
                'risk_score': 20,
                'liquidation_imminent': False,
                'health_factor_status': 'HEALTHY'
            }
        
        elapsed = time.time() - start_time
        
        response = {
            "wallet_address": validated_address.lower(),
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "fetch_time_seconds": round(elapsed, 2),
            "active_chains": active_chains,
            "total_metrics": total_metrics,
            "risk_assessment": risk_assessment,
            "chain_details": chain_details,
            "mode": "fast",
            "note": "Asset breakdown not included. Use /portfolio/view for detailed breakdown."
        }
        
        logger.info(f"⚡ Fast fetch completed in {elapsed:.2f}s - {len(active_chains)} active chains: {active_chains}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Fast portfolio error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fast fetch failed: {str(e)}")
    
# ============================================================
# FIX 3: Add comparison endpoint
# ============================================================

@router_v2.post("/portfolio/compare")
async def compare_wallets(
    wallet_addresses: List[str] = Query(..., description="List of wallet addresses to compare"),
    chain: str = Query("ethereum", description="Chain to compare on"),
    db: Session = Depends(get_db)
):
    """
    Compare multiple wallets side-by-side
    Fast comparison using account data only
    """
    try:
        if not portfolio_service:
            raise HTTPException(status_code=503, detail="Portfolio service not initialized")
        
        if len(wallet_addresses) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 wallets per comparison")
        
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(portfolio_service.rpc_endpoints.get(chain)))
        
        if not w3.is_connected():
            raise HTTPException(status_code=503, detail=f"RPC not connected for {chain}")
        
        comparisons = []
        
        for wallet in wallet_addresses:
            try:
                account_data = portfolio_service.get_comprehensive_account_data(
                    w3, chain, wallet
                )
                
                if account_data:
                    comparisons.append({
                        "wallet_address": wallet.lower(),
                        "collateral_usd": round(account_data['total_collateral_usd'], 2),
                        "debt_usd": round(account_data['total_debt_usd'], 2),
                        "health_factor": round(account_data['health_factor'], 2) if account_data['health_factor'] and account_data['health_factor'] < 999 else "∞",
                        "ltv": round(account_data['ltv'], 4),
                        "liquidation_threshold": round(account_data['current_liquidation_threshold'], 4),
                        "net_worth_usd": round(account_data['net_worth_usd'], 2),
                        "risk_level": account_data['risk_level']
                    })
                else:
                    comparisons.append({
                        "wallet_address": wallet.lower(),
                        "collateral_usd": 0,
                        "debt_usd": 0,
                        "health_factor": None,
                        "ltv": 0,
                        "liquidation_threshold": 0,
                        "net_worth_usd": 0,
                        "risk_level": "NO_POSITIONS"
                    })
                    
            except Exception as e:
                logger.error(f"Error comparing {wallet}: {e}")
                comparisons.append({
                    "wallet_address": wallet.lower(),
                    "error": str(e)
                })
        
        # Sort by net worth
        comparisons_with_values = [c for c in comparisons if 'net_worth_usd' in c]
        comparisons_with_values.sort(key=lambda x: x['net_worth_usd'], reverse=True)
        
        return {
            "chain": chain,
            "wallets_compared": len(wallet_addresses),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "comparisons": comparisons_with_values
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
        
# FIX 12: Add get_asset_details endpoint
@router_v2.post("/portfolio/view")
async def view_portfolio(request: PortfolioRequest, db: Session = Depends(get_db)):
    try:
        # Validate address first
        validated_address = validate_wallet_address(request.wallet_address)
        
        if not portfolio_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Use validated address
        portfolio = portfolio_service.get_user_portfolio(
            wallet_address=validated_address,
            chains=request.chains
        )
                        
        # CRITICAL FIX: The service returns 'chain_details' not 'portfolio_data'
        chain_details = portfolio.get('chain_details', {})
        
        # Format chain details with separated collateral and debt
        formatted_chain_details = {}
        for chain_name, chain_data in chain_details.items():
            # Get collateral and debt assets from the chain data
            collateral_assets = chain_data.get('collateral_assets', [])
            debt_assets = chain_data.get('debt_assets', [])
            
            formatted_chain_details[chain_name] = {
                "has_positions": chain_data.get('has_positions', False),
                "account_data": chain_data.get('account_data', {}),
                "collateral_assets": collateral_assets,
                "debt_assets": debt_assets,
                "summary": chain_data.get('summary', {
                    'total_collateral_assets': len(collateral_assets),
                    'total_debt_assets': len(debt_assets)
                })
            }
        
        # Get active chains
        active_chains = portfolio.get('active_chains', [])
        
        # Build response with proper structure
        response = {
            "wallet_address": portfolio.get('wallet_address', request.wallet_address),
            "fetch_timestamp": portfolio.get('fetch_timestamp', datetime.now(timezone.utc).isoformat()),
            "active_chains": active_chains,
            "total_metrics": portfolio.get('total_metrics', {
                'total_collateral_usd': 0.0,
                'total_debt_usd': 0.0,
                'total_available_borrows_usd': 0.0,
                'total_net_worth_usd': 0.0,
                'lowest_health_factor': None,
                'utilization_ratio_percent': 0.0
            }),
            "risk_assessment": portfolio.get('risk_assessment', {
                'overall_risk_level': 'NO_POSITIONS',
                'risk_score': 0,
                'liquidation_imminent': False,
                'health_factor_status': 'NO_POSITIONS'
            }),
            "chain_details": formatted_chain_details
        }
        
        # Apply rounding to all metrics
        return round_metrics(response)
        
    except Exception as e:
        logger.error(f"Portfolio endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Portfolio fetch failed: {str(e)}"
        )

@router_v2.get("/users/{user_id}/alerts/history")
async def get_alert_history(
    user_id: int,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get alert history for a user"""
    alerts = db.query(AlertHistory).filter(
        AlertHistory.user_id == user_id
    ).order_by(AlertHistory.created_at.desc()).limit(limit).all()
    
    return {
        "user_id": user_id,
        "total_alerts": len(alerts),
        "alerts": [
            {
                "alert_id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "wallet_address": alert.wallet_address,
                "chain": alert.chain,
                "delivered_via": {
                    "email": alert.sent_via_email,
                    "telegram": alert.sent_via_telegram,
                    "slack": alert.sent_via_slack
                },
                "created_at": alert.created_at
            }
            for alert in alerts
        ]
    }

@router_v2.get("/users/{user_id}/portfolio/history")
async def get_portfolio_history(
    user_id: int,
    wallet_address: str,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get historical portfolio snapshots for a monitored address"""
    try:
        monitored = db.query(MonitoredAddress).filter(
            MonitoredAddress.user_id == user_id,
            MonitoredAddress.wallet_address == wallet_address.lower()
        ).first()
        
        if not monitored:
            raise HTTPException(status_code=404, detail="Address not monitored by this user")
        
        cutoff = datetime.now() - timedelta(days=days)
        snapshots = db.query(PositionSnapshot).filter(
            PositionSnapshot.monitored_address_id == monitored.id,
            PositionSnapshot.snapshot_time >= cutoff
        ).order_by(PositionSnapshot.snapshot_time).all()
        
        if not snapshots:
            return {
                "wallet_address": wallet_address,
                "period_days": days,
                "snapshots": [],
                "message": "No historical data available"
            }
        
        history = []
        for snapshot in snapshots:
            history.append({
                "timestamp": snapshot.snapshot_time.isoformat(),
                "collateral_usd": round(snapshot.total_collateral_usd, 2),
                "debt_usd": round(snapshot.total_debt_usd, 2),
                "health_factor": round(snapshot.health_factor, 3) if snapshot.health_factor else None,
                "ltv_ratio": round(snapshot.ltv_ratio, 4) if snapshot.ltv_ratio else None,
                "risk_category": snapshot.risk_category
            })
        
        first = snapshots[0]
        last = snapshots[-1]
        
        collateral_change = last.total_collateral_usd - first.total_collateral_usd
        debt_change = last.total_debt_usd - first.total_debt_usd
        hf_change = (last.health_factor - first.health_factor) if (last.health_factor and first.health_factor) else None
        
        return {
            "wallet_address": wallet_address,
            "period_days": days,
            "snapshot_count": len(snapshots),
            "trends": {
                "collateral_change_usd": round(collateral_change, 2),
                "collateral_change_percent": round((collateral_change / first.total_collateral_usd * 100) if first.total_collateral_usd > 0 else 0, 2),
                "debt_change_usd": round(debt_change, 2),
                "debt_change_percent": round((debt_change / first.total_debt_usd * 100) if first.total_debt_usd > 0 else 0, 2),
                "health_factor_change": round(hf_change, 3) if hf_change else None
            },
            "snapshots": history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

class CacheInvalidateRequest(BaseModel):
    wallet_address: str
    chain: Optional[str] = None  # If None, invalidate all chains for wallet

@router_v2.post("/portfolio/cache/invalidate")
async def invalidate_portfolio_cache(
    request: CacheInvalidateRequest,
    password: str = Query(..., description="Admin password"),
    _verified: bool = Depends(verify_admin_password)
):
    """Invalidate cache for specific wallet - ⚠️ Requires admin password"""
    try:
        if not portfolio_service:
            raise HTTPException(status_code=503, detail="Portfolio service not initialized")
        
        wallet = request.wallet_address.lower()
        
        if request.chain:
            # Invalidate specific chain
            cache_key = portfolio_service.cache._get_cache_key(wallet, request.chain)
            cache_path = portfolio_service.cache._get_cache_path(cache_key)
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
                return {
                    "success": True,
                    "message": f"Cache invalidated for {wallet} on {request.chain}"
                }
            else:
                return {
                    "success": False,
                    "message": "No cache found for this wallet/chain"
                }
        else:
            # Invalidate all chains for this wallet
            count = 0
            for chain in portfolio_service.rpc_endpoints.keys():
                cache_key = portfolio_service.cache._get_cache_key(wallet, chain)
                cache_path = portfolio_service.cache._get_cache_path(cache_key)
                
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    count += 1
            
            return {
                "success": True,
                "message": f"Cache invalidated for {wallet} on {count} chains"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache invalidation failed: {str(e)}")


@router_v2.get("/portfolio/cache/stats")
async def get_cache_stats(
    password: str = Query(..., description="Admin password"),
    _verified: bool = Depends(verify_admin_password)
):
    """Get cache statistics - ⚠️ Requires admin password"""
    try:
        if not portfolio_service:
            raise HTTPException(status_code=503, detail="Portfolio service not initialized")
        
        cache_dir = portfolio_service.cache.cache_dir
        
        if not os.path.exists(cache_dir):
            return {
                "total_cached_entries": 0,
                "cache_size_mb": 0,
                "cache_directory": cache_dir
            }
        
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        
        total_size = 0
        valid_entries = 0
        expired_entries = 0
        
        for filename in cache_files:
            filepath = os.path.join(cache_dir, filename)
            total_size += os.path.getsize(filepath)
            
            try:
                with open(filepath, 'r') as f:
                    cached_data = json.load(f)
                cached_time = datetime.fromisoformat(cached_data['cached_at'])
                
                if datetime.now(timezone.utc) - cached_time > portfolio_service.cache.cache_ttl:
                    expired_entries += 1
                else:
                    valid_entries += 1
            except:
                expired_entries += 1
        
        return {
            "total_cached_entries": len(cache_files),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_directory": cache_dir,
            "cache_ttl_minutes": portfolio_service.cache.cache_ttl.total_seconds() / 60
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router_v2.post("/portfolio/cache/cleanup")
async def cleanup_cache(
    password: str = Query(..., description="Admin password"),
    _verified: bool = Depends(verify_admin_password)
):
    """Cleanup expired cache - ⚠️ Requires admin password"""
    try:
        if not portfolio_service:
            raise HTTPException(status_code=503, detail="Portfolio service not initialized")
        
        portfolio_service.cache.clear_expired()
        
        return {
            "success": True,
            "message": "Cache cleanup completed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache cleanup failed: {str(e)}")
    
async def get_chains_info():
    """Get information about supported chains"""
    if not portfolio_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    chains_info = {
        'ethereum': {
            'name': 'Ethereum Mainnet',
            'aave_v3': True,
            'pool_address': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
            'status': 'fully_supported'
        },
        'polygon': {
            'name': 'Polygon PoS',
            'aave_v3': True,
            'pool_address': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'status': 'fully_supported'
        },
        'avalanche': {
            'name': 'Avalanche C-Chain',
            'aave_v3': True,
            'pool_address': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'status': 'fully_supported'
        },
        'arbitrum': {
            'name': 'Arbitrum One',
            'aave_v3': True,
            'pool_address': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'status': 'fully_supported'
        },
        'optimism': {
            'name': 'Optimism',
            'aave_v3': True,
            'pool_address': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'status': 'fully_supported'
        },
        'base': {
            'name': 'Base',
            'aave_v3': True,
            'pool_address': '0xA238Dd80C259a72e81d7e4664a9801593F98d1c5',
            'status': 'fully_supported'
        },
        'bnb': {
            'name': 'BNB Smart Chain',
            'aave_v3': False,
            'alternative_protocols': ['Venus Protocol', 'Radiant Capital'],
            'status': 'not_supported',
            'reason': 'No official Aave V3 deployment'
        },
        'fantom': {
            'name': 'Fantom Opera',
            'aave_v3': False,
            'alternative_protocols': ['Geist Finance'],
            'status': 'not_supported',
            'reason': 'No official Aave V3 deployment'
        },
        'gnosis': {
            'name': 'Gnosis Chain',
            'aave_v3': False,
            'status': 'not_supported',
            'reason': 'Limited Aave V3 support'
        },
        'celo': {
            'name': 'Celo',
            'aave_v3': False,
            'status': 'not_supported',
            'reason': 'No official Aave V3 deployment'
        }
    }
    
    return {
        'total_chains': len(chains_info),
        'supported_chains': [k for k, v in chains_info.items() if v.get('aave_v3')],
        'unsupported_chains': [k for k, v in chains_info.items() if not v.get('aave_v3')],
        'chains': chains_info
    }
            
__all__ = ['router_v1', 'router_v2', 'init_services', 'SUPPORTED_CHAINS']
