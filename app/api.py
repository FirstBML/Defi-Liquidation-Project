# api.py
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import pandas as pd
import os
from dune_client.client import DuneClient

from .db_models import SessionLocal, Position, AnalysisSnapshot
from .analyzer import analyze_positions
from .scheduler import job_run
from .config import get_settings, ADMIN_API_KEY
from .AAveRiskAnalyze import EnhancedPriceFetcher  # Import your price fetcher
from pydantic import BaseModel

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
# Authentication helper
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if ADMIN_API_KEY and x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# ---- Data fetching functions ----
def get_positions_data() -> pd.DataFrame:
    """
    Fetch positions data from Dune Analytics (same as AAveRiskAnalyze.py)
    """
    try:
        DUNE_API_KEY = os.getenv("DUNE_API_KEY_CURRENT_POSITION")
        if not DUNE_API_KEY:
            raise ValueError("DUNE_API_KEY_CURRENT_POSITION environment variable not set")
        
        dune = DuneClient(api_key=DUNE_API_KEY)
        response = dune.get_custom_endpoint_result("firstbml", "current-position", limit=5000)
        
        if hasattr(response, "result") and hasattr(response.result, "rows"):
            df_positions = pd.DataFrame(response.result.rows)
            print(f"✅ Successfully fetched {len(df_positions)} positions from Dune Analytics")
            return df_positions
        else:
            print("❌ No position data found in Dune response")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error fetching positions data: {str(e)}")
        return pd.DataFrame()

def get_liquidations_data() -> pd.DataFrame:
    """
    Fetch liquidation history from Dune Analytics
    """
    try:
        DUNE_API_KEY = os.getenv("DUNE_API_KEY_LIQUIDATION_HISTORY")
        if not DUNE_API_KEY:
            return pd.DataFrame()
        
        dune = DuneClient(api_key=DUNE_API_KEY)
        response = dune.get_custom_endpoint_result("firstbml", "liquidation-history", limit=5000)
        
        if hasattr(response, "result") and hasattr(response.result, "rows"):
            return pd.DataFrame(response.result.rows)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️ Error fetching liquidations data: {str(e)}")
        return pd.DataFrame()

def get_reserves_data() -> pd.DataFrame:
    """
    Fetch reserve data from Dune Analytics
    """
    try:
        DUNE_API_KEY = os.getenv("DUNE_API_KEY_RESERVE")
        if not DUNE_API_KEY:
            return pd.DataFrame()
        
        dune = DuneClient(api_key=DUNE_API_KEY)
        response = dune.get_custom_endpoint_result("firstbml", "current-reserve", limit=5000)
        
        if hasattr(response, "result") and hasattr(response.result, "rows"):
            return pd.DataFrame(response.result.rows)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️ Error fetching reserves data: {str(e)}")
        return pd.DataFrame()

# ---- Pydantic models for clean output ----
class PositionOut(BaseModel):
    borrower_address: str
    chain: str
    token_symbol: str
    token_address: str
    collateral_amount: float
    debt_amount: float
    enhanced_health_factor: Optional[float]
    risk_category: Optional[str]
    last_updated: datetime

class SnapshotOut(BaseModel):
    timestamp: datetime
    summary: dict

# ---- Endpoints ----

@router.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@router.get("/positions", response_model=List[PositionOut])
def get_positions(db: Session = Depends(get_db), limit: int = 200):
    return db.query(Position).limit(limit).all()

@router.get("/snapshots", response_model=List[SnapshotOut])
def get_snapshots(db: Session = Depends(get_db), limit: int = 20):
    return (
        db.query(AnalysisSnapshot)
        .order_by(AnalysisSnapshot.timestamp.desc())
        .limit(limit)
        .all()
    )

@router.get("/snapshots/latest", response_model=SnapshotOut)
def latest_snapshot(db: Session = Depends(get_db)):
    snap = db.query(AnalysisSnapshot).order_by(AnalysisSnapshot.timestamp.desc()).first()
    if not snap:
        raise HTTPException(status_code=404, detail="No snapshot found")
    return snap

@router.post("/analyze", dependencies=[Depends(verify_api_key)])
def run_analysis():
    """Run analysis with current market data from Dune Analytics"""
    try:
        # Initialize dependencies
        price_fetcher = EnhancedPriceFetcher()
        settings = get_settings()
        positions_data = get_positions_data()  # Get positions from Dune Analytics
        
        if positions_data.empty:
            return {"error": "No positions data available from Dune Analytics"}
        
        # Run analysis
        result = analyze_positions(price_fetcher, settings, positions_data)
        return {"status": "analysis complete", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        result = job_run()  # This calls the comprehensive job_run function
        return {"status": "ok", "message": "analysis run triggered", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dune-status")
def check_dune_status():
    """Check Dune Analytics connectivity and data availability"""
    try:
        positions_data = get_positions_data()
        liquidations_data = get_liquidations_data()
        reserves_data = get_reserves_data()
        
        return {
            "dune_analytics_status": "connected",
            "positions_count": len(positions_data),
            "liquidations_count": len(liquidations_data),
            "reserves_count": len(reserves_data),
            "environment_variables": {
                "DUNE_API_KEY_cURRENT_POSITION": "✅ Set" if os.getenv("DUNE_API_KEY_CURRENT_POSITION") else "❌ Missing",
                "DUNE_API_KEY_LIQUIDATION_HISTORY": "✅ Set" if os.getenv("DUNE_API_KEY_LIQUIDATION_HISTORY") else "❌ Missing",
                "DUNE_API_KEY_RESERVE": "✅ Set" if os.getenv("DUNE_API_KEY_RESERVE") else "❌ Missing",
                "COINGECKO_API_KEY": "✅ Set" if os.getenv("COINGECKO_API_KEY") else "❌ Missing"
            }
        }
    except Exception as e:
        return {"dune_analytics_status": "error", "error": str(e)}
    
