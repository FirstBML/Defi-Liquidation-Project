# scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session
import os
import json
from datetime import datetime
import pandas as pd
import traceback

# Relative imports for your modules
from .db_models import SessionLocal
from .analyzer import analyze_positions, AAVERiskAnalyzer
from .price_fetcher import EnhancedPriceFetcher
from .alerts import notify_critical_alerts

# Try to import config, but provide defaults if not available
try:
    from .config import get_settings
except ImportError:
    # Fallback if config is not properly set up
    def get_settings():
        class SimpleSettings:
            COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
            SCHEDULE_INTERVAL_MINUTES = 15
            SCENARIOS = {"10%_drop": 0.10, "20%_drop": 0.20, "30%_drop": 0.30}
        return SimpleSettings()

scheduler = BackgroundScheduler()

def job_run():
    """
    Single synchronous run of the analysis pipeline.
    """
    try:
        settings = get_settings()

        # 1. load data from Dune or fallback CSVs
        df_reserve = _load_dune_table("firstbml", "current-reserve", "DUNE_API_KEY_RESERVE")
        df_positions = _load_dune_table("firstbml", "current-position", "DUNE_API_KEY_CURRENT_POSITION")
        df_liqs = _load_dune_table("firstbml", "liquidation-history", "DUNE_API_KEY_LIQUIDATION_HISTORY")

        # fallback: if positions empty, try local CSV in working folder
        if df_positions.empty:
            try:
                df_positions = pd.read_csv("aave_positions_detailed_latest.csv")
            except Exception:
                pass

        # 2. price fetcher
        fetcher = EnhancedPriceFetcher(api_key=getattr(settings, 'COINGECKO_API_KEY', None), 
                                     cache_ttl=getattr(settings, 'PRICE_CACHE_TTL', 300))
        analyzer = AAVERiskAnalyzer(fetcher, settings=settings)

        # prepare token dicts for batch pricing
        tokens = []
        if not df_positions.empty:
            for _, r in df_positions.iterrows():
                tokens.append({"symbol": r.get("token_symbol"), "address": r.get("token_address"), "chain": r.get("chain")})
        
        # get batch prices
        prices = fetcher.get_batch_prices(tokens, progress=None)

        # 3. clean and compute enhanced metrics
        df_clean = analyzer.clean_positions(df_positions)
        df_enh = analyzer.compute_enhanced_metrics(df_clean, prices)

        # 4. run scenarios & generate alerts
        scenario_results = analyzer.run_stress_scenarios(df_enh, getattr(settings, 'SCENARIOS', {
            "10%_drop": 0.10,
            "20%_drop": 0.20, 
            "30%_drop": 0.30
        }))
        alerts = analyzer.generate_alerts(df_enh, scenario_results)

        # 5. persist results (positions + snapshot)
        analyzer.persist_results(df_enh, prices, {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": len(df_enh),
            "scenarios": scenario_results,
            "alerts": alerts
        })
        return {"success": True, "positions": len(df_enh)}
    
    except Exception as e:
        print("Job failed:", e)
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def _load_dune_table(namespace: str, endpoint: str, env_var_key: str):
    """Helper function to load data from Dune Analytics"""
    try:
        from dune_client.client import DuneClient
        settings = get_settings()
        key = getattr(settings, env_var_key, os.getenv(env_var_key))
        if not key:
            return pd.DataFrame()
        
        dune = DuneClient(api_key=key)
        resp = dune.get_custom_endpoint_result(namespace, endpoint, limit=5000)
        if hasattr(resp, "result") and hasattr(resp.result, "rows"):
            return pd.DataFrame(resp.result.rows)
    except Exception:
        traceback.print_exc()
    return pd.DataFrame()

def start_scheduler():
    """Start the background scheduler"""
    settings = get_settings()
    interval_minutes = getattr(settings, 'SCHEDULE_INTERVAL_MINUTES', 1440)
    
    scheduler.add_job(job_run, "interval", minutes=interval_minutes, id="aave_risk_job")
    scheduler.start()
    print(f"✅ Scheduler started with {interval_minutes} minute interval")

def stop_scheduler():
    """Stop the background scheduler"""
    scheduler.shutdown()
    print("✅ Scheduler stopped")