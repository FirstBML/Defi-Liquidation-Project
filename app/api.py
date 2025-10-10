"""
Updated API with Reserve Endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from datetime import datetime, timezone
from sqlalchemy import func, desc
from sqlalchemy.orm import Session
from pydantic import BaseModel

import os
import pandas as pd

from .db_models import SessionLocal, Reserve, Position, LiquidationHistory, AnalysisSnapshot
from .rpc_reserve_fetcher import AaveRPCReserveFetcher
from .price_fetcher import EnhancedPriceFetcher
from .config import get_settings

import logging

# Add this at the top of api.py after imports
FALLBACK_LIQUIDATION_THRESHOLDS = {
    # Ethereum mainnet tokens
    'WETH': 0.825, 'WBTC': 0.70, 'USDC': 0.875, 'DAI': 0.77,
    'USDT': 0.80, 'LINK': 0.70, 'AAVE': 0.66, 'UNI': 0.70,
    'WMATIC': 0.70, 'stETH': 0.825, 'wstETH': 0.825,
    
    # Polygon tokens
    'MATIC': 0.70, 'MaticX': 0.70, 'WPOL': 0.70,
    
    # Arbitrum/Optimism
    'ARB': 0.70, 'OP': 0.70,
    
    # Avalanche
    'AVAX': 0.70, 'WAVAX': 0.70,
    
    # Celo-specific tokens (add more as needed)
    'CELO': 0.70, 'cUSD': 0.85, 'cEUR': 0.85, 'cREAL': 0.85,
    'USDC.e': 0.875,
    
    # Pendle tokens
    'PT-sUSDE-27NOV2025': 0.75,
    
    # Default for unknown tokens
    'DEFAULT': 0.75
}

def get_liquidation_threshold(db: Session, chain: str, token_symbol: str) -> tuple[float, str]:
    """
    Get liquidation threshold with fallback logic
    Returns: (threshold_value, source)
    source can be: 'reserve', 'fallback', 'default'
    """
    # Try to get from Reserve table (latest record)
    reserve = db.query(Reserve).filter(
        Reserve.chain == chain,
        Reserve.token_symbol == token_symbol,
        Reserve.is_active == True
    ).order_by(Reserve.query_time.desc()).first()
    
    if reserve and reserve.liquidation_threshold:
        return (reserve.liquidation_threshold, 'reserve')
    
    # Fallback to static mapping
    if token_symbol in FALLBACK_LIQUIDATION_THRESHOLDS:
        return (FALLBACK_LIQUIDATION_THRESHOLDS[token_symbol], 'fallback')
    
    # Default
    logger.warning(f"No LT found for {token_symbol} on {chain}, using default 0.75")
    return (FALLBACK_LIQUIDATION_THRESHOLDS['DEFAULT'], 'default')


logger = logging.getLogger(__name__)
router = APIRouter()

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== BASIC ENDPOINTS ====================

@router.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "DeFi Liquidation Risk API",
        "version": "2.0.0",
        "status": "operational",
        "data_source": "Blockchain RPC"
    }

@router.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy"}

# ==================== RPC RESERVE ENDPOINTS ====================

@router.get("/reserves/rpc")
async def get_rpc_reserves(
    chains: Optional[str] = None,  # Comma-separated: "ethereum,polygon"
    active_only: bool = True,
    limit: Optional[int] = 100,
    latest_only: bool = True,  # Only latest per token
    db: Session = Depends(get_db)
):
    """
    Get RPC-sourced reserve data
    Query params:
    - chains: Comma-separated list (e.g., "ethereum,polygon,arbitrum")
    - active_only: Only return active reserves (default: True)
    - limit: Max results per chain
    - latest_only: Only latest record per token (default: True)
    """
    try:
        query = db.query(Reserve)
        
        # Multi-chain filter
        if chains:
            chain_list = [c.strip().lower() for c in chains.split(',')]
            query = query.filter(Reserve.chain.in_(chain_list))
        
        if active_only:
            query = query.filter(Reserve.is_active == True)
        
        # Get latest only
        if latest_only:
            from sqlalchemy import and_
            
            # Subquery for latest timestamp per token per chain
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
            "chain_filter": chains,
            "reserves": [
                {
                    "chain": r.chain,
                    "token_symbol": r.token_symbol,
                    "token_address": r.token_address,
                    "supply_apy": round(r.supply_apy or 0, 4),
                    "borrow_apy": round(r.borrow_apy or 0, 4),
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


# ==================== UNIFIED DATA REFRESH ENDPOINT ====================


class RefreshRequest(BaseModel):
    """Request model for data refresh"""
    chains: Optional[List[str]] = None
    refresh_reserves: bool = True
    refresh_positions: bool = True
    refresh_liquidations: bool = True
    prices_only: bool = False  # If True, only update prices, not full RPC fetch

@router.post("/data/refresh")
async def unified_data_refresh(
    body: Optional[RefreshRequest] = None,
    db: Session = Depends(get_db)
):
    """
    🔄 UNIFIED DATA REFRESH ENDPOINT
    
    Refreshes all data sources:
    - RPC Reserve data (from blockchain)
    - Position data (from Dune)
    - Liquidation history (from Dune)
    - Token prices (from CoinGecko)
    
    Usage:
    - Manual: POST /api/data/refresh
    - Full refresh: {"refresh_reserves": true, "refresh_positions": true}
    - Prices only: {"prices_only": true}
    - Specific chains: {"chains": ["ethereum", "polygon"]}
    """
    try:
        from .price_fetcher import EnhancedPriceFetcher
        from .rpc_reserve_fetcher import AaveRPCReserveFetcher
        from dune_client.client import DuneClient
        
        settings = get_settings()
        request = body or RefreshRequest()
        
        results = {
            "timestamp": datetime.now(timezone.utc),
            "chains": request.chains or "all",
            "reserves": None,
            "positions": None,
            "liquidations": None,
            "prices": None
        }
        
        # Initialize price fetcher (needed for everything)
        logger.info("Initializing price fetcher...")
        price_fetcher = EnhancedPriceFetcher(
            api_key=getattr(settings, 'COINGECKO_API_KEY', None)
        )
        
        # ========== 1. REFRESH RESERVES ==========
        if request.refresh_reserves and not request.prices_only:
            logger.info("🔄 Refreshing reserve data from RPC...")
            try:
                rpc_fetcher = AaveRPCReserveFetcher(price_fetcher=price_fetcher)
                all_data = rpc_fetcher.fetch_all_chains(chains=request.chains)
                
                stored_count = 0
                for chain, df in all_data.items():
                    for _, row in df.iterrows():
                        reserve = Reserve(
                            chain=row['chain'],
                            token_address=row['token_address'],
                            token_symbol=row['token_symbol'],
                            token_name=row['token_name'],
                            decimals=row['decimals'],
                            liquidity_rate=row['liquidity_rate'],
                            variable_borrow_rate=row['variable_borrow_rate'],
                            stable_borrow_rate=row['stable_borrow_rate'],
                            supply_apy=row['supply_apy'],
                            borrow_apy=row['borrow_apy'],
                            ltv=row['ltv'],
                            liquidation_threshold=row['liquidation_threshold'],
                            liquidation_bonus=row['liquidation_bonus'],
                            is_active=row['is_active'],
                            is_frozen=row['is_frozen'],
                            borrowing_enabled=row['borrowing_enabled'],
                            stable_borrowing_enabled=row['stable_borrowing_enabled'],
                            liquidity_index=row['liquidity_index'],
                            variable_borrow_index=row['variable_borrow_index'],
                            atoken_address=row['atoken_address'],
                            variable_debt_token_address=row['variable_debt_token_address'],
                            price_usd=row.get('price_usd', 0.0),
                            price_available=row.get('price_available', False),
                            last_update_timestamp=row['last_update_timestamp'],
                            query_time=row['query_time']
                        )
                        db.add(reserve)
                        stored_count += 1
                
                db.commit()
                results["reserves"] = {
                    "status": "success",
                    "count": stored_count,
                    "chains_processed": list(all_data.keys())
                }
                logger.info(f"✅ Stored {stored_count} reserves")
                
            except Exception as e:
                logger.error(f"Reserve refresh failed: {e}")
                results["reserves"] = {"status": "error", "error": str(e)}
        
       # ========== 2. REFRESH POSITIONS (FROM DUNE) - FIXED ==========
        """
        Fixed API with Dune Client Compatibility
        Key fixes:
        1. Removed query_id parameter (not supported in dune-client 1.9.1)
        2. Using get_latest_result_dataframe() instead
        3. Better error handling for Dune API calls
        """

        # In your api.py, replace the positions refresh section with this:

        # ========== 2. REFRESH POSITIONS (FROM DUNE) - FIXED ==========
        if request.refresh_positions and not request.prices_only:
            logger.info("🔄 Refreshing position data from Dune...")
            try:
                dune_key = os.getenv("DUNE_API_KEY_CURRENT_POSITION")
                if dune_key:
                    from dune_client.client import DuneClient
                    
                    dune = DuneClient(api_key=dune_key)
                    
                    # ✅ CORRECT METHOD for dune-client 1.9.1
                    logger.info("Calling Dune API endpoint...")
                    
                    # Use the correct method for your dune-client version
                    try:
                        # Method 1: Using query ID directly (most compatible)
                        query_id = 4559935  # Your current-position query ID
                        result = dune.get_latest_result(query_id)
                        
                        # Extract rows based on result type
                        if hasattr(result, 'get_rows'):
                            rows = result.get_rows()
                        elif hasattr(result, 'result') and hasattr(result.result, 'rows'):
                            rows = result.result.rows
                        else:
                            # Fallback: try to access as dict
                            rows = result.get('result', {}).get('rows', [])
                        
                        logger.info(f"Got {len(rows) if rows else 0} rows from Dune")
                        
                    except Exception as e:
                        logger.error(f"Dune API error: {e}")
                        # Try alternative method
                        try:
                            result_df = dune.get_latest_result_dataframe(query_id)
                            rows = result_df.to_dict('records') if result_df is not None else []
                        except Exception as e2:
                            logger.error(f"Dataframe method also failed: {e2}")
                            rows = []
                    
                    if rows:
                        # Clear old positions
                        db.query(Position).delete()
                        
                        position_count = 0
                        for row in rows:
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
                                risk_category=row.get('risk_category'),
                                last_updated=datetime.now(timezone.utc)
                            )
                            db.add(position)
                            position_count += 1
                        
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
                else:
                    results["positions"] = {
                        "status": "skipped",
                        "reason": "DUNE_API_KEY_CURRENT_POSITION not set"
                    }
                    
            except Exception as e:
                import traceback
                logger.error(f"Position refresh failed: {e}")
                logger.error(traceback.format_exc())
                results["positions"] = {"status": "error", "error": str(e)}

        # ========== 3. REFRESH LIQUIDATIONS (FROM DUNE) - FIXED ==========
        if request.refresh_liquidations and not request.prices_only:
            logger.info("🔄 Refreshing liquidation data from Dune...")
            try:
                dune_key = os.getenv("DUNE_API_KEY_LIQUIDATION_HISTORY")
                if dune_key:
                    from dune_client.client import DuneClient
                    
                    dune = DuneClient(api_key=dune_key)
                    
                    logger.info("Calling Dune API endpoint...")
                    
                    try:
                        # Use query ID directly
                        query_id = 4559847  # Your liquidation-history query ID
                        result = dune.get_latest_result(query_id)
                        
                        # Extract rows
                        if hasattr(result, 'get_rows'):
                            rows = result.get_rows()
                        elif hasattr(result, 'result') and hasattr(result.result, 'rows'):
                            rows = result.result.rows
                        else:
                            rows = result.get('result', {}).get('rows', [])
                        
                        logger.info(f"Got {len(rows) if rows else 0} liquidation rows from Dune")
                        
                    except Exception as e:
                        logger.error(f"Dune API error: {e}")
                        try:
                            result_df = dune.get_latest_result_dataframe(query_id)
                            rows = result_df.to_dict('records') if result_df is not None else []
                        except Exception as e2:
                            logger.error(f"Dataframe method also failed: {e2}")
                            rows = []
                    
                    if rows:
                        # Clear old liquidations
                        db.query(LiquidationHistory).delete()
                        
                        liq_count = 0
                        for row in rows:
                            # Calculate USD value
                            collateral_seized = row.get('total_collateral_seized', 0) or 0
                            liquidated_usd = 0.0
                            
                            if collateral_seized > 0:
                                symbol = row.get('collateral_symbol')
                                if symbol:
                                    try:
                                        price_data = price_fetcher.get_batch_prices([{
                                            'symbol': symbol,
                                            'address': row.get('collateral_token'),
                                            'chain': row.get('chain')
                                        }])
                                        if symbol in price_data:
                                            price = price_data[symbol].get('price', 0)
                                            liquidated_usd = collateral_seized * price
                                    except Exception as price_err:
                                        logger.warning(f"Failed to get price for {symbol}: {price_err}")
                            
                            liquidation = LiquidationHistory(
                                liquidation_date=pd.to_datetime(row.get('liquidation_date')) if row.get('liquidation_date') else datetime.now(timezone.utc),
                                chain=row.get('chain'),
                                borrower=row.get('borrower'),
                                collateral_symbol=row.get('collateral_symbol'),
                                debt_symbol=row.get('debt_symbol'),
                                collateral_asset=row.get('collateral_token'),
                                debt_asset=row.get('debt_token'),
                                total_collateral_seized=collateral_seized,
                                total_debt_normalized=row.get('total_debt_normalized'),
                                liquidated_collateral_usd=liquidated_usd,
                                liquidation_count=row.get('liquidation_count'),
                                query_time=datetime.now(timezone.utc)
                            )
                            db.add(liquidation)
                            liq_count += 1
                        
                        db.commit()
                        results["liquidations"] = {
                            "status": "success",
                            "count": liq_count
                        }
                        logger.info(f"✅ Stored {liq_count} liquidations")
                    else:
                        results["liquidations"] = {
                            "status": "success",
                            "count": 0,
                            "note": "No rows returned from Dune"
                        }
                else:
                    results["liquidations"] = {
                        "status": "skipped",
                        "reason": "DUNE_API_KEY_LIQUIDATION_HISTORY not set"
                    }
                    
            except Exception as e:
                import traceback
                logger.error(f"Liquidation refresh failed: {e}")
                logger.error(traceback.format_exc())
                results["liquidations"] = {"status": "error", "error": str(e)}
                
                
        # ========== 4. REFRESH PRICES ONLY (FAST) ==========
        if request.prices_only:
            logger.info("🔄 Refreshing prices only...")
            try:
                reserves = db.query(Reserve).filter(Reserve.is_active == True).all()
                
                from collections import defaultdict
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
        
        return {
            "status": "completed",
            "results": results,
            "summary": {
                "reserves_updated": results["reserves"]["count"] if results["reserves"] and results["reserves"].get("status") == "success" else 0,
                "positions_updated": results["positions"]["count"] if results["positions"] and results["positions"].get("status") == "success" else 0,
                "liquidations_updated": results["liquidations"]["count"] if results["liquidations"] and results["liquidations"].get("status") == "success" else 0,
                "prices_updated": results["prices"]["updated"] if results["prices"] and results["prices"].get("status") == "success" else 0
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Unified refresh failed: {e}")
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")

@router.get("/reserves/rpc/summary")
async def get_rpc_reserve_summary(db: Session = Depends(get_db)):
    """Get summary statistics of RPC reserve data"""
    try:
        # Get latest reserves per chain
        from sqlalchemy import case
        
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

# ==================== POSITION ENDPOINTS ====================

# Fix 2: Positions endpoint - Group by borrower and add USD values
@router.get("/positions")
async def get_positions(
    limit: Optional[int] = 100,
    risk_category: Optional[str] = None,
    group_by_borrower: bool = False,
    db: Session = Depends(get_db)
):
    """Get all positions with optional grouping by borrower"""
    try:
        query = db.query(Position)
        
        if risk_category:
            query = query.filter(Position.risk_category == risk_category)
        
        positions = query.limit(limit * 2).all()  # Get more to account for grouping
        
        # In api.py, update the positions endpoint grouping section:

        if group_by_borrower:
            from collections import defaultdict
            borrower_groups = defaultdict(list)
            
            for p in positions:
                key = f"{p.borrower_address}_{p.chain}"
                borrower_groups[key].append(p)
            
            result = []
            for key, pos_list in list(borrower_groups.items())[:limit]:
                total_collateral = sum(p.total_collateral_usd or 0 for p in pos_list)
                total_debt = sum(p.total_debt_usd or 0 for p in pos_list)
                
                # Calculate health factor
                if total_debt > 0:
                    # Has debt - use actual HF
                    valid_hfs = [p.enhanced_health_factor for p in pos_list if p.enhanced_health_factor]
                    min_hf = min(valid_hfs) if valid_hfs else None
                    risk_cat = next((p.risk_category for p in pos_list if p.enhanced_health_factor == min_hf), None)
                else:
                    # No debt - collateral only position
                    min_hf = float('inf')  # Infinite HF (no liquidation risk)
                    risk_cat = "SAFE"
                
                result.append({
                    "borrower_address": pos_list[0].borrower_address,
                    "chain": pos_list[0].chain,
                    "position_count": len(pos_list),
                    "tokens": [p.token_symbol for p in pos_list],
                    "total_collateral_usd": round(total_collateral, 2),
                    "total_debt_usd": round(total_debt, 2),
                    "lowest_health_factor": round(min_hf, 4) if min_hf != float('inf') else "∞",
                    "risk_category": risk_cat,
                    "position_type": "collateral_only" if total_debt == 0 else "active_borrowing",
                    "last_updated": max(p.last_updated for p in pos_list if p.last_updated)
                })
            
            return result
        else:
            # Return individual positions with USD values
            return [
                {
                    "borrower_address": p.borrower_address,
                    "chain": p.chain,
                    "token_symbol": p.token_symbol,
                    "collateral_amount": p.collateral_amount,
                    "debt_amount": p.debt_amount,
                    "collateral_usd": round(p.total_collateral_usd or 0, 2),
                    "debt_usd": round(p.total_debt_usd or 0, 2),
                    "enhanced_health_factor": p.enhanced_health_factor,
                    "risk_category": p.risk_category,
                    "last_updated": p.last_updated
                }
                for p in positions[:limit]
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch positions: {str(e)}")


@router.get("/positions/risky")
async def get_risky_positions(
    threshold_hf: Optional[float] = 1.5,
    db: Session = Depends(get_db)
):
    """Get positions with low health factors"""
    try:
        positions = db.query(Position).filter(
            Position.enhanced_health_factor < threshold_hf
        ).order_by(Position.enhanced_health_factor).all()
        
        return {
            "threshold": threshold_hf,
            "count": len(positions),
            "positions": [
                {
                    "borrower_address": p.borrower_address,
                    "token_symbol": p.token_symbol,
                    "health_factor": p.enhanced_health_factor,
                    "collateral_usd": p.total_collateral_usd,
                    "debt_usd": p.total_debt_usd,
                    "risk_category": p.risk_category
                }
                for p in positions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch risky positions: {str(e)}")

@router.get("/positions_summary")
async def get_positions_summary(db: Session = Depends(get_db)):
    """Get summarized view of borrower positions"""
    try:
        positions = db.query(Position).all()
        if not positions:
            raise HTTPException(status_code=404, detail="No positions found")

        summary = [
            {
                "borrower": p.borrower_address,
                "chain": p.chain,
                "token_symbol": p.token_symbol,
                "total_collateral_usd": round(p.total_collateral_usd or 0.0, 2),
                "total_debt_usd": round(p.total_debt_usd or 0.0, 2),
                "health_factor": round(p.health_factor or 0.0, 4),
            }
            for p in positions
        ]

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch positions: {str(e)}")

# ==================== LIQUIDATION HISTORY ====================

# Fix Liquidation history - Add USD values to response

@router.get("/liquidation-history")
async def get_liquidation_history(
    limit: Optional[int] = 100,
    chain: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get liquidation history with USD values"""
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
                "debt_repaid_usd": round(liq.liquidated_debt_usd or 0, 2)  # Changed from total_debt_usd
            }
            for liq in liquidations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
            
# ==================== DASHBOARD ====================

@router.get("/protocol_risk_summary")
async def get_protocol_risk_summary(
    chains: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Comprehensive protocol-level risk metrics with LT source tracking
    """
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
                "average_health_factor": 0,
                "at_risk_value_usd": 0,
                "chains_analyzed": chain_filter or [],
                "lt_coverage": {"reserve": 0, "fallback": 0, "default": 0}
            }
        
        # Track LT sources
        lt_sources = {"reserve": 0, "fallback": 0, "default": 0}
        
        for pos in positions:
            _, lt_source = get_liquidation_threshold(db, pos.chain, pos.token_symbol)
            lt_sources[lt_source] += 1
        
        total_collateral = sum(p.total_collateral_usd or 0 for p in positions)
        total_debt = sum(p.total_debt_usd or 0 for p in positions)
        
        valid_hfs = [p.enhanced_health_factor for p in positions 
                     if p.enhanced_health_factor and p.enhanced_health_factor < 100]
        avg_hf = sum(valid_hfs) / len(valid_hfs) if valid_hfs else 0
        
        at_risk = [p for p in positions if p.enhanced_health_factor and p.enhanced_health_factor < 1.5]
        at_risk_value = sum(p.total_collateral_usd or 0 for p in at_risk)
        
        return {
            "total_collateral_usd": round(total_collateral, 2),
            "total_debt_usd": round(total_debt, 2),
            "protocol_ltv": round(total_debt / total_collateral, 4) if total_collateral > 0 else 0,
            "average_health_factor": round(avg_hf, 3),
            "at_risk_value_usd": round(at_risk_value, 2),
            "at_risk_percentage": round((at_risk_value / total_collateral * 100) if total_collateral > 0 else 0, 2),
            "chains_analyzed": chain_filter or list(set(p.chain for p in positions if p.chain)),
            "lt_coverage": {
                "from_reserve_data": lt_sources['reserve'],
                "from_fallback": lt_sources['fallback'],
                "from_default": lt_sources['default'],
                "reserve_coverage_pct": round((lt_sources['reserve'] / len(positions) * 100), 2)
            },
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
    
# ==================== CHAIN MANAGEMENT ====================    

@router.get("/chains/available")
async def get_available_chains(db: Session = Depends(get_db)):
    """Get chains from both reserves and positions"""
    try:
        # Get chains from reserves
        reserve_chains = set(c[0] for c in db.query(Reserve.chain).distinct().all() if c[0])
        
        # Get chains from positions  
        position_chains = set(c[0] for c in db.query(Position.chain).distinct().all() if c[0])
        
        # Combined unique chains
        all_chains = sorted(reserve_chains | position_chains)
        
        details = []
        for chain in all_chains:
            reserve_count = db.query(Reserve).filter(Reserve.chain == chain).count()
            position_count = db.query(Position).filter(Position.chain == chain).count()
            
            reserve_latest = db.query(func.max(Reserve.query_time)).filter(Reserve.chain == chain).scalar()
            
            # FIX: Display "NA on reserve" instead of null
            last_reserve_display = reserve_latest if reserve_latest else "NA on reserve"
            
            details.append({
                "chain": chain,
                "reserve_count": reserve_count,
                "position_count": position_count,
                "has_reserves": reserve_count > 0,
                "has_positions": position_count > 0,
                "last_reserve_update": last_reserve_display  # Changed from reserve_latest
            })
        
        return {
            "chains": all_chains,
            "count": len(all_chains),
            "reserve_chains": sorted(reserve_chains),
            "position_chains": sorted(position_chains),
            "details": details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# ==================== DEEP INSIGHTS ENDPOINT ====================

@router.get("/insights/protocol-health")
async def get_protocol_health_insights(
    chains: Optional[str] = None,  # "ethereum,polygon"
    db: Session = Depends(get_db)
):
    """
    Deep protocol health insights combining Reserves, Positions, and Liquidations
    Returns comprehensive risk analysis for frontend dashboards
    """
    try:
        # Parse chains
        chain_filter = None
        if chains:
            chain_filter = [c.strip().lower() for c in chains.split(',')]
        
        # 1. GET LATEST RESERVES DATA
        reserve_query = db.query(Reserve).filter(Reserve.is_active == True)
        if chain_filter:
            reserve_query = reserve_query.filter(Reserve.chain.in_(chain_filter))
        
        # Get only latest per token
        subq = db.query(
            Reserve.chain,
            Reserve.token_address,
            func.max(Reserve.query_time).label('max_time')
        ).group_by(Reserve.chain, Reserve.token_address).subquery()
        
        reserves = reserve_query.join(
            subq,
            (Reserve.chain == subq.c.chain) &
            (Reserve.token_address == subq.c.token_address) &
            (Reserve.query_time == subq.c.max_time)
        ).all()
        
        # 2. GET POSITIONS DATA
        position_query = db.query(Position)
        if chain_filter:
            position_query = position_query.filter(Position.chain.in_(chain_filter))
        positions = position_query.all()
        
        # 3. GET LIQUIDATION HISTORY (last 30 days)
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        
        liq_query = db.query(LiquidationHistory).filter(
            LiquidationHistory.liquidation_date >= thirty_days_ago
        )
        if chain_filter:
            liq_query = liq_query.filter(LiquidationHistory.chain.in_(chain_filter))
        liquidations = liq_query.all()
        
        # 4. COMPUTE CROSS-DATA INSIGHTS
        
        # Reserve insights
        reserve_insights = {
            'total_reserves': len(reserves),
            'high_apy_reserves': len([r for r in reserves if r.borrow_apy > 10]),
            'frozen_reserves': len([r for r in reserves if r.is_frozen]),
            'avg_supply_apy': sum(r.supply_apy or 0 for r in reserves) / len(reserves) if reserves else 0,
            'avg_borrow_apy': sum(r.borrow_apy or 0 for r in reserves) / len(reserves) if reserves else 0,
            'high_ltv_reserves': len([r for r in reserves if (r.ltv or 0) > 0.75]),
            'tokens_with_prices': sum(1 for r in reserves if (r.price_usd or 0) > 0),
            'price_coverage': (sum(1 for r in reserves if (r.price_usd or 0) > 0) / len(reserves) * 100) if reserves else 0
        }
        
        # Position insights
        risky_positions = [p for p in positions if p.enhanced_health_factor and p.enhanced_health_factor < 1.5]
        critical_positions = [p for p in positions if p.enhanced_health_factor and p.enhanced_health_factor < 1.1]
        
        total_collateral = sum(p.total_collateral_usd or 0 for p in positions)
        total_debt = sum(p.total_debt_usd or 0 for p in positions)
        
        position_insights = {
            'total_positions': len(positions),
            'risky_positions': len(risky_positions),
            'critical_positions': len(critical_positions),
            'total_collateral_usd': total_collateral,
            'total_debt_usd': total_debt,
            'protocol_ltv': (total_debt / total_collateral) if total_collateral > 0 else 0,
            'avg_health_factor': sum(p.enhanced_health_factor or 0 for p in positions if p.enhanced_health_factor and p.enhanced_health_factor < 100) / len([p for p in positions if p.enhanced_health_factor and p.enhanced_health_factor < 100]) if positions else 0,
            'risk_distribution': {
                'SAFE': len([p for p in positions if p.risk_category == 'SAFE']),
                'LOW_RISK': len([p for p in positions if p.risk_category == 'LOW_RISK']),
                'MEDIUM_RISK': len([p for p in positions if p.risk_category == 'MEDIUM_RISK']),
                'HIGH_RISK': len([p for p in positions if p.risk_category == 'HIGH_RISK']),
                'CRITICAL': len([p for p in positions if p.risk_category == 'CRITICAL']),
                'LIQUIDATION_IMMINENT': len([p for p in positions if p.risk_category == 'LIQUIDATION_IMMINENT'])
            }
        }
        
        # Liquidation insights
        total_liquidated_value = sum(l.liquidated_collateral_usd or 0 for l in liquidations)
        
        liquidation_insights = {
            'liquidations_30d': len(liquidations),
            'total_liquidated_usd': total_liquidated_value,
            'avg_liquidation_size': total_liquidated_value / len(liquidations) if liquidations else 0,
            'unique_liquidators': len(set(l.liquidator for l in liquidations if l.liquidator)),
            'liquidation_rate_per_day': len(liquidations) / 30 if liquidations else 0,
            'top_liquidated_assets': {}
        }
        
        # Top liquidated assets
        from collections import Counter
        asset_counter = Counter(l.collateral_symbol for l in liquidations if l.collateral_symbol)
        liquidation_insights['top_liquidated_assets'] = dict(asset_counter.most_common(5))
        
        # 5. CROSS-DATA CORRELATIONS
        
        # Tokens at risk (appear in both risky positions and reserves)
        risky_token_symbols = set(p.token_symbol for p in risky_positions if p.token_symbol)
        reserve_token_symbols = set(r.token_symbol for r in reserves)
        tokens_at_risk = list(risky_token_symbols & reserve_token_symbols)
        
        # Find reserve data for risky tokens
        risky_token_details = []
        for token in tokens_at_risk[:10]:  # Top 10
            reserve = next((r for r in reserves if r.token_symbol == token), None)
            position_count = len([p for p in risky_positions if p.token_symbol == token])
            
            if reserve:
                risky_token_details.append({
                    'token_symbol': token,
                    'risky_position_count': position_count,
                    'supply_apy': round(reserve.supply_apy or 0, 2),
                    'borrow_apy': round(reserve.borrow_apy or 0, 2),
                    'liquidation_threshold': reserve.liquidation_threshold,
                    'is_frozen': reserve.is_frozen,
                    'price_usd': reserve.price_usd
                })
        
        # 6. CHAIN BREAKDOWN
        chain_breakdown = {}
        for chain in (chain_filter or set(r.chain for r in reserves)):
            chain_reserves = [r for r in reserves if r.chain == chain]
            chain_positions = [p for p in positions if p.chain == chain]
            chain_liqs = [l for l in liquidations if l.chain == chain]
            
            chain_breakdown[chain] = {
                'reserves': len(chain_reserves),
                'positions': len(chain_positions),
                'liquidations_30d': len(chain_liqs),
                'avg_supply_apy': sum(r.supply_apy or 0 for r in chain_reserves) / len(chain_reserves) if chain_reserves else 0,
                'total_collateral': sum(p.total_collateral_usd or 0 for p in chain_positions),
                'risky_positions': len([p for p in chain_positions if p.enhanced_health_factor and p.enhanced_health_factor < 1.5])
            }
        
        # 7. ALERTS & WARNINGS
        alerts = []
        
        if position_insights['critical_positions'] > 10:
            alerts.append({
                'severity': 'HIGH',
                'type': 'CRITICAL_POSITIONS',
                'message': f"{position_insights['critical_positions']} positions with HF < 1.1",
                'action': 'Monitor closely, prepare liquidation infrastructure'
            })
        
        if position_insights['protocol_ltv'] > 0.70:
            alerts.append({
                'severity': 'HIGH',
                'type': 'HIGH_PROTOCOL_LTV',
                'message': f"Protocol LTV at {position_insights['protocol_ltv']:.1%}",
                'action': 'Review risk parameters'
            })
        
        if reserve_insights['frozen_reserves'] > 5:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'FROZEN_RESERVES',
                'message': f"{reserve_insights['frozen_reserves']} reserves frozen",
                'action': 'Review frozen assets'
            })
        
        if liquidation_insights['liquidation_rate_per_day'] > 10:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'HIGH_LIQUIDATION_RATE',
                'message': f"{liquidation_insights['liquidation_rate_per_day']:.1f} liquidations/day",
                'action': 'Investigate market conditions'
            })
        
        # 8. RETURN COMPREHENSIVE INSIGHTS
        return {
            'timestamp': datetime.now(timezone.utc),
            'chains_analyzed': list(chain_filter) if chain_filter else 'all',
            'data_sources': {
                'reserves': len(reserves),
                'positions': len(positions),
                'liquidations_30d': len(liquidations)
            },
            'reserve_insights': reserve_insights,
            'position_insights': position_insights,
            'liquidation_insights': liquidation_insights,
            'tokens_at_risk': risky_token_details,
            'chain_breakdown': chain_breakdown,
            'alerts': alerts,
            'health_score': calculate_health_score(
                position_insights,
                reserve_insights,
                liquidation_insights
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

"""
Replace the calculate_health_score function in api.py with this version
Fixes division by zero errors
"""

def calculate_health_score(position_insights, reserve_insights, liquidation_insights) -> Dict[str, any]:
    """Calculate overall protocol health score (0-100) with safe division"""
    score = 100
    
    try:
        # Safe division for critical positions
        total_pos = position_insights.get('total_positions', 0)
        if total_pos > 0:
            critical_count = position_insights.get('critical_positions', 0)
            critical_ratio = critical_count / total_pos
            score -= critical_ratio * 40  # Max 40 point deduction
        
        # Safe LTV check
        ltv = position_insights.get('protocol_ltv', 0)
        if ltv and ltv > 0.70:
            score -= min((ltv - 0.70) * 100, 30)  # Max 30 point deduction
        
        # Safe liquidation rate
        liq_rate = liquidation_insights.get('liquidation_rate_per_day', 0)
        if liq_rate and liq_rate > 5:
            score -= min((liq_rate - 5) * 2, 20)  # Max 20 point deduction
        
        # Safe frozen reserves ratio
        total_reserves = reserve_insights.get('total_reserves', 0)
        if total_reserves and total_reserves > 0:
            frozen_count = reserve_insights.get('frozen_reserves', 0)
            frozen_ratio = frozen_count / total_reserves
            score -= frozen_ratio * 10  # Max 10 point deduction
        
    except Exception as e:
        # If anything fails, just return default score
        print(f"Warning in health score calculation: {e}")
    
    # Clamp score between 0 and 100
    score = max(0, min(100, score))
    
    # Determine status
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

@router.post("/stress-test/custom")
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
            stressed_collateral = pos.total_collateral_usd * (1 - drop_pct)
            
            if pos.total_debt_usd and pos.total_debt_usd > 0:
                liquidation_threshold = pos.liquidation_threshold or 0.85
                stressed_hf = (stressed_collateral * liquidation_threshold) / pos.total_debt_usd
                
                if stressed_hf < 1.0:
                    total_at_risk += 1
                    total_value_at_risk += stressed_collateral
                    
                    results.append({
                        "borrower": pos.borrower_address[:10] + "...",
                        "token": pos.token_symbol,
                        "current_hf": pos.enhanced_health_factor,
                        "stressed_hf": round(stressed_hf, 3),
                        "collateral_at_risk": round(stressed_collateral, 2),
                        "price_drop": drop_pct * 100
                    })
        
        return {
            "scenario_name": scenario_name,
            "price_drops_applied": price_drops,
            "total_positions": len(positions),
            "positions_at_risk": total_at_risk,
            "percentage_at_risk": round((total_at_risk / len(positions) * 100) if positions else 0, 2),
            "collateral_at_risk_usd": round(total_value_at_risk, 2),
            "critical_positions": sorted(results, key=lambda x: x["collateral_at_risk"], reverse=True)[:20]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom stress test failed: {str(e)}")
    
"""
Advanced Risk Analysis Endpoints
Add these to your api.py file
"""

# ==================== PROTOCOL RISK SUMMARY ====================

@router.get("/protocol_risk_summary")
async def get_protocol_risk_summary(
    chains: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Comprehensive protocol-level risk metrics with LT source tracking
    """
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
                "average_health_factor": 0,
                "at_risk_value_usd": 0,
                "chains_analyzed": chain_filter or [],
                "lt_coverage": {"reserve": 0, "fallback": 0, "default": 0}
            }
        
        # Track LT sources
        lt_sources = {"reserve": 0, "fallback": 0, "default": 0}
        
        for pos in positions:
            _, lt_source = get_liquidation_threshold(db, pos.chain, pos.token_symbol)
            lt_sources[lt_source] += 1
        
        total_collateral = sum(p.total_collateral_usd or 0 for p in positions)
        total_debt = sum(p.total_debt_usd or 0 for p in positions)
        
        valid_hfs = [p.enhanced_health_factor for p in positions 
                     if p.enhanced_health_factor and p.enhanced_health_factor < 100]
        avg_hf = sum(valid_hfs) / len(valid_hfs) if valid_hfs else 0
        
        at_risk = [p for p in positions if p.enhanced_health_factor and p.enhanced_health_factor < 1.5]
        at_risk_value = sum(p.total_collateral_usd or 0 for p in at_risk)
        
        return {
            "total_collateral_usd": round(total_collateral, 2),
            "total_debt_usd": round(total_debt, 2),
            "protocol_ltv": round(total_debt / total_collateral, 4) if total_collateral > 0 else 0,
            "average_health_factor": round(avg_hf, 3),
            "at_risk_value_usd": round(at_risk_value, 2),
            "at_risk_percentage": round((at_risk_value / total_collateral * 100) if total_collateral > 0 else 0, 2),
            "chains_analyzed": chain_filter or list(set(p.chain for p in positions if p.chain)),
            "lt_coverage": {
                "from_reserve_data": lt_sources['reserve'],
                "from_fallback": lt_sources['fallback'],
                "from_default": lt_sources['default'],
                "reserve_coverage_pct": round((lt_sources['reserve'] / len(positions) * 100), 2)
            },
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
    

# ==================== BORROWER RISK SIGNALS ====================

@router.get("/borrower_risk_signals")
async def get_borrower_risk_signals(
    threshold: float = 1.5,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Identify borrowers trending toward liquidation
    Now with proper liquidation_threshold from Reserve JOIN + fallback
    """
    try:
        # Get risky positions
        positions = db.query(Position).filter(
            Position.enhanced_health_factor < threshold,
            Position.total_debt_usd > 0
        ).order_by(Position.enhanced_health_factor).limit(limit).all()
        
        signals = []
        
        for pos in positions:
            # Get liquidation threshold with fallback
            lt, lt_source = get_liquidation_threshold(db, pos.chain, pos.token_symbol)
            
            # Calculate current LTV
            current_ltv = (pos.total_debt_usd / pos.total_collateral_usd) if pos.total_collateral_usd else 0
            
            # Determine urgency
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
                "lt_source": lt_source,  # Shows if from reserve/fallback/default
                "collateral_usd": round(pos.total_collateral_usd or 0, 2),
                "debt_usd": round(pos.total_debt_usd or 0, 2),
                "primary_collateral": pos.token_symbol,
                "risk_category": pos.risk_category,
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

@router.get("/reserve_risk_metrics")
async def get_reserve_risk_metrics(
    chains: Optional[str] = None,
    min_ltv: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """
    Evaluate which assets are most dangerous to system health
    Returns: Utilization, LTV, LT, liquidation bonus, top borrower exposure
    """
    try:
        chain_filter = [c.strip().lower() for c in chains.split(',')] if chains else None
        
        # Get latest reserves only
        subq = db.query(
            Reserve.chain,
            Reserve.token_address,
            func.max(Reserve.query_time).label('max_time')
        ).group_by(Reserve.chain, Reserve.token_address).subquery()
        
        reserve_query = db.query(Reserve).join(
            subq,
            (Reserve.chain == subq.c.chain) &
            (Reserve.token_address == subq.c.token_address) &
            (Reserve.query_time == subq.c.max_time)
        ).filter(Reserve.is_active == True)
        
        if chain_filter:
            reserve_query = reserve_query.filter(Reserve.chain.in_(chain_filter))
        
        if min_ltv is not None:
            reserve_query = reserve_query.filter(Reserve.ltv >= min_ltv)
        
        reserves = reserve_query.all()
        
        # Get positions for exposure calculation
        position_query = db.query(Position)
        if chain_filter:
            position_query = position_query.filter(Position.chain.in_(chain_filter))
        positions = position_query.all()
        
        metrics = []
        
        for reserve in reserves:
            # Calculate utilization from rates
            if reserve.liquidity_rate and reserve.variable_borrow_rate:
                utilization = reserve.variable_borrow_rate / (reserve.liquidity_rate + reserve.variable_borrow_rate)
            else:
                utilization = 0
            
            # Find positions using this asset
            asset_positions = [p for p in positions if p.token_symbol == reserve.token_symbol]
            total_exposure = sum(p.total_collateral_usd or 0 for p in asset_positions)
            
            # Top borrowers for this asset
            top_borrowers = sorted(asset_positions, key=lambda x: x.total_collateral_usd or 0, reverse=True)[:3]
            
            # Risk score (higher = more risky)

            # In reserve_risk_metrics endpoint, replace risk score calculation:

            risk_score = 0

            # Frozen reserves (but check if there's actual activity)
            if reserve.is_frozen:
                if total_exposure > 0 or len(asset_positions) > 0:
                    risk_score += 25  # Only add if frozen WITH exposure
                else:
                    risk_score += 10  # Lower score if frozen but no exposure

            # High LTV
            if reserve.ltv and reserve.ltv > 0.75:
                risk_score += 30

            # High utilization
            if utilization > 0.80:
                risk_score += 25

            # Low liquidation threshold
            if reserve.liquidation_threshold and reserve.liquidation_threshold < 0.75:
                risk_score += 20

            # Large single borrower concentration (>50% of total exposure)
            if total_exposure > 0 and top_borrowers:
                concentration = (top_borrowers[0].total_collateral_usd or 0) / total_exposure
                if concentration > 0.5:
                    risk_score += 15
            
            metrics.append({
                "token_symbol": reserve.token_symbol,
                "chain": reserve.chain,
                "utilization_rate": round(utilization, 4),
                "ltv": reserve.ltv,
                "liquidation_threshold": reserve.liquidation_threshold,
                "liquidation_bonus": reserve.liquidation_bonus,
                "supply_apy": round(reserve.supply_apy or 0, 2),
                "borrow_apy": round(reserve.borrow_apy or 0, 2),
                "is_frozen": reserve.is_frozen,
                "borrowing_enabled": reserve.borrowing_enabled,
                "price_usd": reserve.price_usd,
                "total_exposure_usd": round(total_exposure, 2),
                "position_count": len(asset_positions),
                "top_borrower_exposure": round(top_borrowers[0].total_collateral_usd or 0, 2) if top_borrowers else 0,
                "risk_score": risk_score,
                "risk_level": "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 30 else "LOW"
            })
        
        # Sort by risk score
        metrics.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return {
            "reserves_analyzed": len(metrics),
            "high_risk_count": len([m for m in metrics if m['risk_level'] == 'HIGH']),
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


# ==================== LIQUIDATION TRENDS ====================

@router.get("/liquidation_trends")
async def get_liquidation_trends(
    days: int = 7,
    chains: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Identify liquidation patterns and predict new liquidation zones
    Returns: 24h/7d liquidations, volume, top assets, avg HF before liquidation
    """
    try:
        from datetime import timedelta
        from collections import Counter
        
        chain_filter = [c.strip().lower() for c in chains.split(',')] if chains else None
        
        now = datetime.now()
        cutoff_7d = now - timedelta(days=days)
        cutoff_24h = now - timedelta(days=1)

        # Get liquidations
        liq_query = db.query(LiquidationHistory).filter(
            LiquidationHistory.liquidation_date >= cutoff_7d
        )
        if chain_filter:
            liq_query = liq_query.filter(LiquidationHistory.chain.in_(chain_filter))
        
        liquidations_7d = liq_query.all()
        liquidations_24h = [l for l in liquidations_7d if l.liquidation_date >= cutoff_24h]
        
        # Calculate metrics with safe division
        total_volume_7d = sum(l.liquidated_collateral_usd or 0 for l in liquidations_7d)
        total_volume_24h = sum(l.liquidated_collateral_usd or 0 for l in liquidations_24h)
        
        # Top liquidated assets
        asset_counter = Counter(l.collateral_symbol for l in liquidations_7d if l.collateral_symbol)
        top_assets = [
            {"asset": asset, "count": count}
            for asset, count in asset_counter.most_common(10)
        ]
        
        # Average HF before liquidation - FIXED: Safe None handling
        hfs_before = [l.health_factor_before for l in liquidations_7d 
                     if l.health_factor_before and l.health_factor_before < 1000]  # Reasonable upper bound
        avg_hf_before = sum(hfs_before) / len(hfs_before) if hfs_before else None
        
        # Chain distribution with safe division
        chain_counter = Counter(l.chain for l in liquidations_7d if l.chain)
        chain_distribution = [
            {
                "chain": chain, 
                "count": count, 
                "percentage": round(count / len(liquidations_7d) * 100, 2) if liquidations_7d else 0
            }
            for chain, count in chain_counter.most_common()
        ]
        
        # Trend analysis (compare 24h vs 7d average)
        daily_avg_7d = len(liquidations_7d) / days if liquidations_7d else 0
        trend = "STABLE"
        if liquidations_24h:
            if len(liquidations_24h) > daily_avg_7d * 1.2:
                trend = "INCREASING"
            elif len(liquidations_24h) < daily_avg_7d * 0.8:
                trend = "DECREASING"
        
        result = {
            "period_days": days,
            "liquidations_24h": len(liquidations_24h),
            "liquidations_7d": len(liquidations_7d),
            "liquidation_volume_usd_24h": round(total_volume_24h, 2),
            "liquidation_volume_usd_7d": round(total_volume_7d, 2),
            "top_liquidated_assets": top_assets,
            "chain_distribution": chain_distribution,
            "trend": trend,
            "daily_average_7d": round(daily_avg_7d, 2),
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Only add average HF if we have valid data
        if avg_hf_before is not None:
            result["average_health_factor_before"] = round(avg_hf_before, 3)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
    
# ==================== CROSS-CHAIN RISK COMPARISON ====================

@router.get("/crosschain_risk_comparison")
async def get_crosschain_risk_comparison(db: Session = Depends(get_db)):
    """
    Compare risk metrics across all Aave deployments
    Returns: Avg HF per chain, debt/collateral ratio, liquidations, top tokens
    """
    try:
        # Get all positions
        positions = db.query(Position).all()
        
        # Get all reserves
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
        
        # Get liquidations (last 7 days)
        from datetime import timedelta
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        liquidations = db.query(LiquidationHistory).filter(
            LiquidationHistory.liquidation_date >= week_ago
        ).all()
        
        # Group by chain
        chains = set(p.chain for p in positions if p.chain)
        
        comparison = []
        
        for chain in chains:
            chain_positions = [p for p in positions if p.chain == chain]
            chain_reserves = [r for r in reserves if r.chain == chain]
            chain_liqs = [l for l in liquidations if l.chain == chain]
            
            # Calculate metrics
            valid_hfs = [p.enhanced_health_factor for p in chain_positions 
                        if p.enhanced_health_factor and p.enhanced_health_factor < 100]
            avg_hf = sum(valid_hfs) / len(valid_hfs) if valid_hfs else 0
            
            total_collateral = sum(p.total_collateral_usd or 0 for p in chain_positions)
            total_debt = sum(p.total_debt_usd or 0 for p in chain_positions)
            debt_ratio = total_debt / total_collateral if total_collateral > 0 else 0
            
            # Top collateral tokens
            from collections import Counter
            token_counter = Counter(p.token_symbol for p in chain_positions if p.token_symbol)
            top_tokens = [token for token, _ in token_counter.most_common(5)]
            
            # Reserves with LT < Avg(HF)
            risky_reserves = [r for r in chain_reserves if r.liquidation_threshold and r.liquidation_threshold < avg_hf]
            
            comparison.append({
                "chain": chain,
                "average_health_factor": round(avg_hf, 3),
                "debt_collateral_ratio": round(debt_ratio, 4),
                "total_positions": len(chain_positions),
                "total_collateral_usd": round(total_collateral, 2),
                "total_debt_usd": round(total_debt, 2),
                "liquidations_7d": len(chain_liqs),
                "top_collateral_tokens": top_tokens,
                "reserves_count": len(chain_reserves),
                "risky_reserves_count": len(risky_reserves),
                "safety_score": round((avg_hf / debt_ratio) if debt_ratio > 0 else 100, 2)
            })
        
        # Sort by safety score (descending)
        comparison.sort(key=lambda x: x['safety_score'], reverse=True)
        
        return {
            "chains_analyzed": len(comparison),
            "safest_chain": comparison[0]['chain'] if comparison else None,
            "riskiest_chain": comparison[-1]['chain'] if comparison else None,
            "comparison": comparison,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


# ==================== RISK ALERTS FEED ====================

@router.get("/risk_alerts_feed")
async def get_risk_alerts_feed(
    severity: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Real-time risk alert feed with proper LT handling
    """
    try:
        alerts = []
        
        # 1. Critical Health Factor Alerts (unchanged)
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
        
        # 2. Near Margin Call Alerts (UPDATED with LT lookup)
        positions = db.query(Position).filter(
            Position.total_debt_usd > 0,
            Position.total_collateral_usd > 0
        ).all()
        
        for pos in positions:
            # Get liquidation threshold with fallback
            lt, lt_source = get_liquidation_threshold(db, pos.chain, pos.token_symbol)
            
            current_ltv = (pos.total_debt_usd / pos.total_collateral_usd) if pos.total_collateral_usd else 0
            ltv_threshold = lt * 0.9  # 90% of liquidation threshold
            
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
        
        # 3. Liquidity Squeeze Alerts (unchanged)
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
        
        # Filter by severity if specified
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity.upper()]
        
        # Sort by severity and limit
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        alerts.sort(key=lambda x: severity_order.get(x['severity'], 4))
        alerts = alerts[:limit]
        
        # Count by severity
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
    
# ==================== AVAILABLE CHAINS ENDPOINT ====================

@router.get("/chains/available")
async def get_available_chains(db: Session = Depends(get_db)):
    """Get list of chains with data for dropdown menus"""
    try:
        chains = db.query(Reserve.chain).distinct().all()
        chain_list = [c[0] for c in chains if c[0]]
        
        chain_details = []
        for chain in chain_list:
            count = db.query(Reserve).filter(Reserve.chain == chain).count()
            latest = db.query(func.max(Reserve.query_time)).filter(Reserve.chain == chain).scalar()
            
            chain_details.append({
                "chain": chain,
                "reserve_count": count,
                "last_updated": latest 
            })
        
        return {
            "chains": sorted(chain_list),
            "count": len(chain_list),
            "details": chain_details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/rpc-test")
async def test_rpc_connection():
    """Test if RPC endpoints are accessible"""
    from web3 import Web3
    
    rpcs = {
        "ethereum": os.getenv("ETHEREUM_RPC_URL", "https://eth.llamarpc.com"),
        "polygon": os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
        "arbitrum": os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"),
    }
    
    results = {}
    for chain, url in rpcs.items():
        try:
            w3 = Web3(Web3.HTTPProvider(url))
            connected = w3.is_connected()
            block = w3.eth.block_number if connected else 0
            results[chain] = {
                "url": url,
                "connected": connected,
                "latest_block": block
            }
        except Exception as e:
            results[chain] = {
                "url": url,
                "connected": False,
                "error": str(e)
            }
    
    return results

    """
Add this to your api.py to show data freshness and cache status
Insert after the /startup-status equivalent endpoint
"""

@router.get("/data/status")
async def get_data_status(db: Session = Depends(get_db)):
    """
    Check data availability and freshness
    Shows if database has data and how old it is
    """
    try:
        from datetime import timedelta
        
        now = datetime.now(timezone.utc)
        status = {
            "timestamp": now.isoformat(),
            "database_connected": True,
            "data_available": False,
            "data_age_status": "unknown",
            "sources": {}
        }
        
        # Check reserves
        reserve_count = db.query(Reserve).count()
        latest_reserve = db.query(Reserve).order_by(Reserve.query_time.desc()).first()
        
        if latest_reserve:
            age = now - latest_reserve.query_time.replace(tzinfo=timezone.utc)
            age_hours = age.total_seconds() / 3600
            
            if age_hours < 6:
                reserve_status = "fresh"
            elif age_hours < 24:
                reserve_status = "recent"
            elif age_hours < 168:  # 1 week
                reserve_status = "stale"
            else:
                reserve_status = "very_old"
            
            status["sources"]["reserves"] = {
                "count": reserve_count,
                "latest_update": latest_reserve.query_time.isoformat(),
                "age_hours": round(age_hours, 2),
                "status": reserve_status,
                "chains": db.query(Reserve.chain).distinct().count()
            }
        else:
            status["sources"]["reserves"] = {
                "count": 0,
                "status": "no_data",
                "message": "⚠️ No reserve data - run /api/data/refresh"
            }
        
        # Check positions
        position_count = db.query(Position).count()
        latest_position = db.query(Position).order_by(Position.last_updated.desc()).first()
        
        if latest_position:
            age = now - latest_position.last_updated.replace(tzinfo=timezone.utc)
            age_hours = age.total_seconds() / 3600
            
            if age_hours < 6:
                position_status = "fresh"
            elif age_hours < 24:
                position_status = "recent"
            elif age_hours < 168:
                position_status = "stale"
            else:
                position_status = "very_old"
            
            status["sources"]["positions"] = {
                "count": position_count,
                "latest_update": latest_position.last_updated.isoformat(),
                "age_hours": round(age_hours, 2),
                "status": position_status,
                "at_risk": db.query(Position).filter(
                    Position.enhanced_health_factor < 1.5
                ).count()
            }
        else:
            status["sources"]["positions"] = {
                "count": 0,
                "status": "no_data",
                "message": "⚠️ No position data - run /api/data/refresh"
            }
        
        # Check liquidations
        liq_count = db.query(LiquidationHistory).count()
        latest_liq = db.query(LiquidationHistory).order_by(
            LiquidationHistory.liquidation_date.desc()
        ).first()
        
        if latest_liq:
            # For liquidations, check the actual event date, not query time
            age = now - latest_liq.liquidation_date.replace(tzinfo=timezone.utc)
            age_days = age.total_seconds() / 86400
            
            status["sources"]["liquidations"] = {
                "count": liq_count,
                "latest_event": latest_liq.liquidation_date.isoformat(),
                "age_days": round(age_days, 2),
                "status": "recent" if age_days < 7 else "older",
                "last_30_days": db.query(LiquidationHistory).filter(
                    LiquidationHistory.liquidation_date >= now - timedelta(days=30)
                ).count()
            }
        else:
            status["sources"]["liquidations"] = {
                "count": 0,
                "status": "no_data",
                "message": "⚠️ No liquidation data"
            }
        
        # Overall status
        has_reserves = reserve_count > 0
        has_positions = position_count > 0
        
        if has_reserves and has_positions:
            status["data_available"] = True
            
            # Determine overall age status
            reserve_age = status["sources"]["reserves"].get("age_hours", 999)
            position_age = status["sources"]["positions"].get("age_hours", 999)
            oldest_age = max(reserve_age, position_age)
            
            if oldest_age < 6:
                status["data_age_status"] = "fresh"
                status["recommendation"] = "✅ Data is up to date"
            elif oldest_age < 24:
                status["data_age_status"] = "recent"
                status["recommendation"] = "Data is reasonably fresh"
            else:
                status["data_age_status"] = "stale"
                status["recommendation"] = "⚠️ Consider refreshing data at /api/data/refresh"
        else:
            status["data_available"] = False
            status["data_age_status"] = "no_data"
            status["recommendation"] = "❌ No data found - POST to /api/data/refresh to populate"
        
        return status
        
    except Exception as e:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database_connected": False,
            "error": str(e),
            "recommendation": "Check database connection"
        }


@router.get("/data/quick-stats")
async def get_quick_stats(db: Session = Depends(get_db)):
    """
    Quick dashboard stats - minimal query overhead
    """
    try:
        # Single queries for counts
        reserve_count = db.query(func.count(Reserve.id)).scalar()
        position_count = db.query(func.count(Position.id)).scalar()
        liq_count = db.query(func.count(LiquidationHistory.id)).scalar()
        
        # At-risk positions count
        at_risk = db.query(func.count(Position.id)).filter(
            Position.enhanced_health_factor < 1.5,
            Position.enhanced_health_factor > 0
        ).scalar()
        
        # Critical positions
        critical = db.query(func.count(Position.id)).filter(
            Position.enhanced_health_factor < 1.1,
            Position.enhanced_health_factor > 0
        ).scalar()
        
        # Total value locked (approximate)
        total_collateral = db.query(
            func.sum(Position.total_collateral_usd)
        ).scalar() or 0
        
        total_debt = db.query(
            func.sum(Position.total_debt_usd)
        ).scalar() or 0
        
        return {
            "reserves": reserve_count,
            "positions": position_count,
            "liquidations_history": liq_count,
            "at_risk_positions": at_risk,
            "critical_positions": critical,
            "total_collateral_usd": round(total_collateral, 2),
            "total_debt_usd": round(total_debt, 2),
            "protocol_ltv": round(total_debt / total_collateral, 4) if total_collateral > 0 else 0,
            "has_data": reserve_count > 0 or position_count > 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))