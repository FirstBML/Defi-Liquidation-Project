"""
Advanced features for DeFi Risk System - Integrated with existing alerts and config
"""
from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Dict, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Import from your existing modules
from .dependencies import get_db, verify_admin_key
from .config import settings
from .alerts import send_slack_alert, send_email_alert, notify_critical_alerts

router = APIRouter()

# Pydantic models
class StressScenario(BaseModel):
    """Stress test scenario configuration"""
    scenario_name: str
    price_drops: Dict[str, float]  # token_symbol: drop_percentage
    description: Optional[str] = None

class AlertConfig(BaseModel):
    """Alert configuration"""
    channel: str  # "slack", "email", "both"
    threshold_hf: float = 1.2
    min_position_size: float = 100000.0
    enabled: bool = True

# ==================== STRESS SCENARIOS ====================

@router.post("/stress-test/custom")
async def run_custom_stress_test(
    scenario: StressScenario, 
    db = Depends(get_db),
    _admin = Depends(verify_admin_key)
):
    """
    Run custom stress test with configurable price drops per token
    
    Example:
    {
        "scenario_name": "Market Crash",
        "price_drops": {"WETH": 30, "WBTC": 25},
        "description": "30% ETH drop, 25% BTC drop"
    }
    """
    try:
        from .db_models import Position
        
        positions = db.query(Position).all()
        
        results = []
        total_at_risk = 0
        total_value_at_risk = 0
        
        for pos in positions:
            price_drop = scenario.price_drops.get(pos.token_symbol, 0) / 100
            stressed_collateral = pos.total_collateral_usd * (1 - price_drop)
            
            if pos.total_debt_usd and pos.total_debt_usd > 0:
                liquidation_threshold = pos.liquidation_threshold or 0.85
                stressed_hf = (stressed_collateral * liquidation_threshold) / pos.total_debt_usd
            else:
                stressed_hf = float('inf')
            
            if stressed_hf < 1.0:
                total_at_risk += 1
                total_value_at_risk += stressed_collateral
                
                results.append({
                    "borrower_address": pos.borrower_address,
                    "token_symbol": pos.token_symbol,
                    "current_hf": pos.enhanced_health_factor or pos.health_factor,
                    "stressed_hf": round(stressed_hf, 3),
                    "collateral_at_risk": round(stressed_collateral, 2),
                    "debt_usd": pos.total_debt_usd,
                    "price_drop_applied": price_drop * 100
                })
        
        return {
            "scenario_name": scenario.scenario_name,
            "description": scenario.description,
            "price_drops_applied": scenario.price_drops,
            "summary": {
                "total_positions_analyzed": len(positions),
                "positions_at_risk": total_at_risk,
                "percentage_at_risk": round((total_at_risk / len(positions) * 100) if positions else 0, 2),
                "total_collateral_at_risk_usd": round(total_value_at_risk, 2)
            },
            "at_risk_positions": sorted(results, key=lambda x: x["collateral_at_risk"], reverse=True)[:20],
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")

@router.get("/stress-test/standard")
async def run_standard_stress_tests(db = Depends(get_db)):
    """
    Run standard stress test scenarios from config
    Default: 10%, 20%, 30% drops plus black swan
    """
    try:
        from .db_models import Position
        
        # Use scenarios from config or defaults
        scenarios = getattr(settings, 'SCENARIOS', {
            "10%_drop": 10,
            "20%_drop": 20,
            "30%_drop": 30,
            "50%_black_swan": 50
        })
        
        positions = db.query(Position).all()
        results = {}
        
        for scenario_name, drop_pct in scenarios.items():
            # Convert if stored as decimal (0.1) to percentage (10)
            if drop_pct < 1:
                drop_pct = drop_pct * 100
                
            at_risk_count = 0
            at_risk_value = 0
            critical_positions = []
            
            for pos in positions:
                stressed_collateral = pos.total_collateral_usd * (1 - drop_pct / 100)
                
                if pos.total_debt_usd and pos.total_debt_usd > 0:
                    liquidation_threshold = pos.liquidation_threshold or 0.85
                    stressed_hf = (stressed_collateral * liquidation_threshold) / pos.total_debt_usd
                    
                    if stressed_hf < 1.0:
                        at_risk_count += 1
                        at_risk_value += stressed_collateral
                        
                        # Track whales
                        whale_threshold = getattr(settings, 'ALERT_WHALE_MIN_USD', 100000)
                        if pos.total_collateral_usd > whale_threshold:
                            critical_positions.append({
                                "borrower": pos.borrower_address[:10] + "...",
                                "token": pos.token_symbol,
                                "collateral": round(stressed_collateral, 2),
                                "stressed_hf": round(stressed_hf, 3)
                            })
            
            results[scenario_name] = {
                "positions_liquidated": at_risk_count,
                "percentage_liquidated": round((at_risk_count / len(positions) * 100) if positions else 0, 2),
                "collateral_at_risk_usd": round(at_risk_value, 2),
                "critical_whales": sorted(critical_positions, key=lambda x: x["collateral"], reverse=True)[:5]
            }
        
        return {
            "scenarios": results,
            "total_positions_analyzed": len(positions),
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Standard stress tests failed: {str(e)}")

# ==================== HEALTH FACTOR HISTORY ====================

@router.get("/health-factor/history/{borrower_address}")
async def get_health_factor_history(borrower_address: str, db = Depends(get_db)):
    """Get health factor trend for a specific borrower"""
    try:
        from .db_models import Position
        
        position = db.query(Position).filter(
            Position.borrower_address == borrower_address
        ).first()
        
        if not position:
            raise HTTPException(status_code=404, detail="Borrower not found")
        
        current_hf = position.enhanced_health_factor or position.health_factor
        
        # Determine trend
        if current_hf < 1.0:
            trend = "critical"
        elif current_hf < 1.2:
            trend = "declining"
        elif current_hf < 1.5:
            trend = "stable"
        else:
            trend = "improving"
        
        return {
            "borrower_address": borrower_address,
            "token_symbol": position.token_symbol,
            "current_health_factor": current_hf,
            "collateral_usd": position.total_collateral_usd,
            "debt_usd": position.total_debt_usd,
            "trend": trend,
            "risk_category": position.risk_category,
            "alert_status": "URGENT" if current_hf < 1.1 else "MONITOR" if current_hf < 1.5 else "OK",
            "timestamp": datetime.now(timezone.utc)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get HF history: {str(e)}")

@router.get("/health-factor/declining")
async def get_declining_health_factors(
    limit: Optional[int] = 50,
    db = Depends(get_db)
):
    """Get positions with declining health factors"""
    try:
        from .db_models import Position
        
        threshold = getattr(settings, 'ALERT_HEALTH_FACTOR_THRESHOLD', 1.5)
        
        positions = db.query(Position).filter(
            Position.enhanced_health_factor < threshold
        ).order_by(Position.enhanced_health_factor).limit(limit).all()
        
        results = []
        for pos in positions:
            hf = pos.enhanced_health_factor or pos.health_factor or 0
            
            results.append({
                "borrower_address": pos.borrower_address,
                "token_symbol": pos.token_symbol,
                "health_factor": hf,
                "collateral_usd": pos.total_collateral_usd,
                "debt_usd": pos.total_debt_usd,
                "risk_category": pos.risk_category,
                "urgency": "CRITICAL" if hf < 1.1 else "HIGH" if hf < 1.3 else "MEDIUM"
            })
        
        return {
            "declining_positions": results,
            "total_count": len(results),
            "critical_count": len([r for r in results if r["health_factor"] < 1.1]),
            "threshold_used": threshold,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get declining HFs: {str(e)}")

# ==================== ALERTS & NOTIFICATIONS ====================

@router.post("/alerts/send")
async def send_alert(
    config: AlertConfig, 
    db = Depends(get_db),
    _admin = Depends(verify_admin_key)
):
    """
    Send alerts for critical positions using existing alert system
    Channels: slack, email, both
    """
    try:
        from .db_models import Position
        
        # Get critical positions
        critical_positions = db.query(Position).filter(
            Position.enhanced_health_factor < config.threshold_hf,
            Position.total_collateral_usd > config.min_position_size
        ).all()
        
        if not critical_positions:
            return {
                "status": "no_alerts_needed",
                "message": "No positions meet alert criteria",
                "timestamp": datetime.now(timezone.utc)
            }
        
        # Prepare alert data
        alert_count = len(critical_positions)
        total_at_risk = sum(p.total_collateral_usd for p in critical_positions)
        
        alert_data = {
            "alert_type": "critical_liquidation_risk",
            "alert_count": alert_count,
            "total_at_risk_usd": total_at_risk,
            "threshold_hf": config.threshold_hf,
            "min_position_size": config.min_position_size,
            "top_positions": [
                {
                    "borrower": p.borrower_address[:10] + "...",
                    "token": p.token_symbol,
                    "hf": p.enhanced_health_factor or p.health_factor,
                    "collateral": p.total_collateral_usd
                }
                for p in critical_positions[:5]
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Send to configured channel(s) using existing alert system
        sent_to = []
        
        if config.channel in ["slack", "both"] and config.enabled:
            message = f"CRITICAL LIQUIDATION ALERT\n\n{alert_count} positions at risk!\nTotal: ${total_at_risk:,.0f}\n\nTop 5 positions:\n"
            for i, pos in enumerate(alert_data["top_positions"], 1):
                message += f"{i}. {pos['borrower']} ({pos['token']}) HF: {pos['hf']:.3f}\n"
            
            if send_slack_alert(message):
                sent_to.append("Slack")
        
        if config.channel in ["email", "both"] and config.enabled:
            subject = f"CRITICAL: {alert_count} positions at liquidation risk"
            body = f"""Critical Liquidation Alert
            
Total positions at risk: {alert_count}
Total collateral at risk: ${total_at_risk:,.0f}
Threshold HF: {config.threshold_hf}

Top 5 Critical Positions:
"""
            for pos in alert_data["top_positions"]:
                body += f"\n- {pos['borrower']} ({pos['token']}): HF {pos['hf']:.3f}, ${pos['collateral']:,.0f}"
            
            if send_email_alert(subject, body):
                sent_to.append("Email")
        
        return {
            "status": "alerts_sent",
            "alert_data": alert_data,
            "sent_to": sent_to,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send alerts: {str(e)}")

@router.post("/alerts/critical")
async def send_critical_alerts(
    db = Depends(get_db),
    _admin = Depends(verify_admin_key)
):
    """
    Automatically detect and send alerts for all critical positions
    Uses thresholds from config
    """
    try:
        from .db_models import Position
        
        threshold_hf = getattr(settings, 'ALERT_HEALTH_FACTOR_THRESHOLD', 1.1)
        whale_min = getattr(settings, 'ALERT_WHALE_MIN_USD', 1000000)
        
        # Get all critical positions
        critical = db.query(Position).filter(
            Position.enhanced_health_factor < threshold_hf
        ).all()
        
        # Get whale positions  
        whales = db.query(Position).filter(
            Position.enhanced_health_factor < 1.5,
            Position.total_collateral_usd > whale_min
        ).all()
        
        if not (critical or whales):
            return {
                "status": "no_critical_positions",
                "message": "System is healthy",
                "timestamp": datetime.now(timezone.utc)
            }
        
        # Use existing notify_critical_alerts function
        alert_payload = {
            "critical_positions": len(critical),
            "whale_positions_at_risk": len(whales),
            "total_at_risk": sum(p.total_collateral_usd for p in critical),
            "threshold_hf": threshold_hf,
            "whale_threshold_usd": whale_min,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        notify_critical_alerts(alert_payload)
        
        return {
            "status": "alerts_sent",
            "summary": alert_payload,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send critical alerts: {str(e)}")