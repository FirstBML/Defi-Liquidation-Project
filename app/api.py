# api.py
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import pandas as pd
import os
import sys
from dune_client.client import DuneClient

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from datetime import datetime, timezone
from sqlalchemy import func

from .alerts import notify_critical_alerts
from .advanced_features import router as advanced_router
from .db_models import SessionLocal, Position, AnalysisSnapshot
from .analyzer import analyze_positions
from .scheduler import job_run
from .config import get_settings, ADMIN_API_KEY
from .AAveRiskAnalyze import EnhancedPriceFetcher  # Import your price fetcher
from pydantic import BaseModel

"""
Complete API Router for DeFi Liquidation Risk System
Replace your current app/api.py with this file
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from datetime import datetime, timezone
from sqlalchemy import func

# Create the main router
router = APIRouter()

# Import database models
from .db_models import SessionLocal, Reserve, Position, LiquidationHistory, AnalysisSnapshot

# Dependency for database session
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
        "version": "1.0.0",
        "status": "operational"
    }

@router.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy"}

# ==================== POSITION ENDPOINTS ====================

@router.get("/positions")
async def get_positions(
    limit: Optional[int] = 100,
    risk_category: Optional[str] = None,
    db = Depends(get_db)
):
    """Get all positions with optional filtering"""
    try:
        query = db.query(Position)
        
        if risk_category:
            query = query.filter(Position.risk_category == risk_category)
        
        positions = query.limit(limit).all()
        
        return [
            {
                "borrower_address": p.borrower_address,
                "chain": p.chain,
                "token_symbol": p.token_symbol,
                "collateral_amount": p.collateral_amount,
                "debt_amount": p.debt_amount,
                "enhanced_health_factor": p.enhanced_health_factor,
                "risk_category": p.risk_category,
                "last_updated": p.last_updated
            }
            for p in positions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch positions: {str(e)}")

@router.get("/positions/risky")
async def get_risky_positions(
    threshold_hf: Optional[float] = 1.5,
    db = Depends(get_db)
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

# ==================== RESERVE ENDPOINTS ====================

@router.get("/reserves")
async def get_reserves(db = Depends(get_db)):
    """Get all reserve data"""
    try:
        reserves = db.query(Reserve).all()
        
        return [
            {
                "id": r.id,
                "token_symbol": r.token_symbol,
                "chain": r.chain,
                "total_liquidity": r.total_liquidity,
                "utilization_rate": r.utilization_rate,
                "liquidation_threshold": r.liquidation_threshold,
                "price_usd": r.price_usd,
                "is_active": r.is_active
            }
            for r in reserves
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reserves: {str(e)}")

# ==================== LIQUIDATION HISTORY ====================

@router.get("/liquidation-history")
async def get_liquidation_history(
    limit: Optional[int] = 100,
    chain: Optional[str] = None,
    db = Depends(get_db)
):
    """Get liquidation history"""
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
                "borrower": liq.borrower,
                "collateral_symbol": liq.collateral_symbol,
                "debt_symbol": liq.debt_symbol,
                "total_collateral_seized": liq.total_collateral_seized,
                "total_debt_normalized": liq.total_debt_normalized
            }
            for liq in liquidations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch liquidation history: {str(e)}")

# ==================== DASHBOARD ====================

@router.get("/dashboard/summary")
async def get_dashboard_summary(db = Depends(get_db)):
    """Get comprehensive dashboard summary"""
    try:
        # Position statistics
        total_positions = db.query(Position).count()
        
        # Risk distribution
        risk_stats = db.query(
            Position.risk_category,
            func.count(Position.id).label('count'),
            func.sum(Position.total_collateral_usd).label('total_collateral'),
            func.sum(Position.total_debt_usd).label('total_debt')
        ).group_by(Position.risk_category).all()
        
        risk_distribution = {}
        total_collateral = 0
        total_debt = 0
        
        for stat in risk_stats:
            risk_distribution[stat.risk_category or 'UNKNOWN'] = {
                'count': stat.count,
                'total_collateral': float(stat.total_collateral or 0),
                'total_debt': float(stat.total_debt or 0)
            }
            total_collateral += float(stat.total_collateral or 0)
            total_debt += float(stat.total_debt or 0)
        
        # Protocol health
        protocol_ltv = total_debt / total_collateral if total_collateral > 0 else 0
        
        # Recent liquidations
        recent_liquidations = db.query(LiquidationHistory).order_by(
            LiquidationHistory.liquidation_date.desc()
        ).limit(5).all()
        
        return {
            "protocol_overview": {
                "total_positions": total_positions,
                "total_collateral_usd": total_collateral,
                "total_debt_usd": total_debt,
                "protocol_ltv": protocol_ltv
            },
            "risk_distribution": risk_distribution,
            "recent_liquidations": [
                {
                    "date": liq.liquidation_date,
                    "collateral_seized": liq.total_collateral_seized,
                    "collateral_symbol": liq.collateral_symbol
                }
                for liq in recent_liquidations
            ],
            "timestamp": datetime.now(timezone.utc)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")

# ==================== ADVANCED STRESS TESTS ====================

@router.post("/stress-test/custom")
async def run_custom_stress_test(
    price_drops: Dict[str, float],
    scenario_name: Optional[str] = "Custom Scenario",
    db = Depends(get_db)
):
    """
    Run custom stress test with specific token price drops
    Body: {"WETH": 30, "WBTC": 25, "USDC": 0}
    """
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

@router.get("/stress-test/scenarios")
async def run_multiple_scenarios(db = Depends(get_db)):
    """Run standard stress scenarios: 10%, 20%, 30%, 50% drops"""
    try:
        scenarios = {"10%": 10, "20%": 20, "30%": 30, "50% Black Swan": 50}
        positions = db.query(Position).all()
        
        results = {}
        for name, drop in scenarios.items():
            at_risk = 0
            value_at_risk = 0
            
            for pos in positions:
                stressed_coll = pos.total_collateral_usd * (1 - drop / 100)
                if pos.total_debt_usd > 0:
                    lt = pos.liquidation_threshold or 0.85
                    stressed_hf = (stressed_coll * lt) / pos.total_debt_usd
                    if stressed_hf < 1.0:
                        at_risk += 1
                        value_at_risk += stressed_coll
            
            results[name] = {
                "positions_at_risk": at_risk,
                "percentage": round((at_risk / len(positions) * 100) if positions else 0, 2),
                "value_at_risk": round(value_at_risk, 2)
            }
        
        return {"scenarios": results, "total_positions": len(positions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenarios failed: {str(e)}")

# ==================== HEALTH FACTOR TRACKING ====================

@router.get("/health-factor/declining")
async def get_declining_positions(
    threshold: Optional[float] = 1.5,
    limit: Optional[int] = 50,
    db = Depends(get_db)
):
    """Get positions with declining health factors"""
    try:
        positions = db.query(Position).filter(
            Position.enhanced_health_factor < threshold,
            Position.enhanced_health_factor.isnot(None)
        ).order_by(Position.enhanced_health_factor).limit(limit).all()
        
        return {
            "threshold": threshold,
            "count": len(positions),
            "positions": [
                {
                    "borrower": p.borrower_address,
                    "token": p.token_symbol,
                    "chain": p.chain,
                    "health_factor": p.enhanced_health_factor,
                    "collateral_usd": p.total_collateral_usd,
                    "debt_usd": p.total_debt_usd,
                    "risk": p.risk_category,
                    "urgency": "CRITICAL" if p.enhanced_health_factor < 1.1 else "HIGH" if p.enhanced_health_factor < 1.3 else "MEDIUM"
                }
                for p in positions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

@router.get("/alerts/trigger")
async def trigger_alerts(db = Depends(get_db)):
    """Manually trigger alert check for critical positions"""
    try:
        critical = db.query(Position).filter(
            Position.risk_category.in_(['CRITICAL', 'LIQUIDATION_IMMINENT'])
        ).count()
        
        if critical > 0:
            # Import and run alert function
            import subprocess
            subprocess.Popen([sys.executable, "automated_alerts.py"])
            
            return {
                "status": "alerts_triggered",
                "critical_positions": critical,
                "message": "Alert check initiated"
            }
        else:
            return {
                "status": "no_alerts_needed",
                "message": "No critical positions detected"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")