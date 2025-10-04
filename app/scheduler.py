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

"""
Updated scheduler with RPC reserve fetching integrated
"""
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session
import os
import json
from datetime import datetime, timezone
import pandas as pd
import traceback
import logging

from .db_models import SessionLocal, Reserve
from .analyzer import AAVERiskAnalyzer
from .price_fetcher import EnhancedPriceFetcher
from .alerts import notify_critical_alerts
from .rpc_reserve_fetcher import AaveRPCReserveFetcher

try:
    from .config import get_settings
except ImportError:
    def get_settings():
        class SimpleSettings:
            COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
            SCHEDULE_INTERVAL_MINUTES = 15
            SCENARIOS = {"10%_drop": 0.10, "20%_drop": 0.20, "30%_drop": 0.30}
        return SimpleSettings()

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

def fetch_and_store_rpc_reserves(db: Session, price_fetcher: EnhancedPriceFetcher) -> int:
    """
    Fetch reserve data from RPC and store in database
    Returns: Number of reserves stored
    """
    try:
        logger.info("Starting RPC reserve fetch...")
        
        # Initialize RPC fetcher with price fetcher
        rpc_fetcher = AaveRPCReserveFetcher(price_fetcher=price_fetcher)
        
        # Fetch all chains
        all_chain_data = rpc_fetcher.fetch_all_chains()
        
        if not all_chain_data:
            logger.warning("No RPC data fetched from any chain")
            return 0
        
        total_stored = 0
        
        for chain, df in all_chain_data.items():
            if df.empty:
                continue
            
            logger.info(f"Storing {len(df)} reserves for {chain}")
            
            for _, row in df.iterrows():
                try:
                    # Create new reserve record with timestamp
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
                        query_time=row['query_time'],
                        created_at=datetime.now(timezone.utc)
                    )
                    
                    db.add(reserve)
                    total_stored += 1
                    
                except Exception as e:
                    logger.error(f"Failed to store reserve {row.get('token_symbol', 'UNKNOWN')}: {e}")
                    continue
            
            db.commit()
            logger.info(f"{chain}: Committed {len(df)} reserves to database")
        
        logger.info(f"Total reserves stored: {total_stored}")
        return total_stored
        
    except Exception as e:
        logger.error(f"RPC reserve fetch failed: {e}")
        traceback.print_exc()
        db.rollback()
        return 0

def job_run():
    """
    Complete synchronous run of the analysis pipeline
    Now includes RPC reserve fetching
    """
    try:
        settings = get_settings()
        db = SessionLocal()
        
        try:
            logger.info("="*60)
            logger.info("STARTING SCHEDULED JOB RUN")
            logger.info("="*60)
            
            # 1. Initialize price fetcher
            logger.info("Step 1: Initializing price fetcher...")
            fetcher = EnhancedPriceFetcher(
                api_key=getattr(settings, 'COINGECKO_API_KEY', None),
                cache_ttl=getattr(settings, 'PRICE_CACHE_TTL', 300)
            )
            
            # 2. Fetch and store RPC reserves (REPLACES Dune reserve data)
            logger.info("Step 2: Fetching reserves from RPC...")
            reserves_stored = fetch_and_store_rpc_reserves(db, fetcher)
            logger.info(f"✅ Stored {reserves_stored} reserves from blockchain")
            
            # 3. Load position data from Dune
            logger.info("Step 3: Loading position data...")
            df_positions = _load_dune_table(
                "firstbml", 
                "current-position", 
                "DUNE_API_KEY_CURRENT_POSITION"
            )
            
            # Fallback to CSV if Dune fails
            if df_positions.empty:
                try:
                    df_positions = pd.read_csv("aave_positions_detailed_latest.csv")
                    logger.info("Loaded positions from fallback CSV")
                except Exception:
                    logger.warning("No position data available")
                    return {"success": False, "error": "No position data"}
            
            # 4. Prepare tokens for batch pricing
            logger.info("Step 4: Fetching token prices...")
            tokens = []
            if not df_positions.empty:
                for _, r in df_positions.iterrows():
                    tokens.append({
                        "symbol": r.get("token_symbol"),
                        "address": r.get("token_address"),
                        "chain": r.get("chain")
                    })
            
            prices = fetcher.get_batch_prices(tokens, progress=None)
            logger.info(f"✅ Fetched prices for {len(prices)} tokens")
            
            # 5. Initialize analyzer and process positions
            logger.info("Step 5: Analyzing positions...")
            analyzer = AAVERiskAnalyzer(fetcher, settings=settings)
            df_clean = analyzer.clean_positions(df_positions)
            df_enhanced = analyzer.compute_enhanced_metrics(df_clean, prices)
            
            # 6. Run stress scenarios
            logger.info("Step 6: Running stress scenarios...")
            scenarios = getattr(settings, 'SCENARIOS', {
                "10%_drop": 0.10,
                "20%_drop": 0.20,
                "30%_drop": 0.30
            })
            scenario_results = analyzer.run_stress_scenarios(df_enhanced, scenarios)
            
            # 7. Generate alerts
            logger.info("Step 7: Generating alerts...")
            alerts = analyzer.generate_alerts(df_enhanced, scenario_results)
            
            # 8. Persist results
            logger.info("Step 8: Persisting results...")
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reserves_fetched": reserves_stored,
                "positions_analyzed": len(df_enhanced),
                "scenarios": scenario_results,
                "alerts": alerts
            }
            analyzer.persist_results(df_enhanced, prices, summary)
            
            # 9. Send critical alerts if needed
            if alerts.get('critical_count', 0) > 0:
                logger.warning(f"⚠️ {alerts['critical_count']} critical alerts detected")
                notify_critical_alerts(alerts)
            
            logger.info("="*60)
            logger.info("JOB COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            return {
                "success": True,
                "reserves_stored": reserves_stored,
                "positions_analyzed": len(df_enhanced),
                "critical_alerts": alerts.get('critical_count', 0)
            }
            
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"Job failed: {e}")
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
    except Exception as e:
        logger.error(f"Dune fetch failed: {e}")
    return pd.DataFrame()
# In scheduler.py, add price-only refresh function:

def refresh_prices_only(db: Session, price_fetcher: EnhancedPriceFetcher) -> int:
    """
    Refresh only prices for existing reserves (faster than full RPC fetch)
    Run this more frequently than full RPC refresh
    """
    try:
        logger.info("Starting price-only refresh...")
        
        # Get all active reserves
        reserves = db.query(Reserve).filter(Reserve.is_active == True).all()
        
        if not reserves:
            logger.warning("No reserves to update prices for")
            return 0
        
        # Group by chain for batch fetching
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
            
            logger.info(f"Fetching prices for {len(tokens)} tokens on {chain}...")
            prices = price_fetcher.get_batch_prices(tokens, progress=None)
            
            for reserve in chain_reserves:
                price_data = prices.get(reserve.token_symbol, {})
                if isinstance(price_data, dict):
                    price = price_data.get('price', 0.0)
                else:
                    price = float(price_data) if price_data else 0.0
                
                if price > 0:
                    reserve.price_usd = price
                    reserve.price_available = True
                    updated_count += 1
            
            db.commit()
            logger.info(f"{chain}: Updated {updated_count} prices")
        
        logger.info(f"Price refresh complete: {updated_count}/{len(reserves)} updated")
        return updated_count
        
    except Exception as e:
        logger.error(f"Price refresh failed: {e}")
        db.rollback()
        return 0
    
def start_scheduler():
    """Start both RPC and price refresh jobs"""
    settings = get_settings()
    
    # Full RPC refresh every 1440 minutes (24 hours)
    scheduler.add_job(job_run, "interval", minutes=1440, id="aave_risk_job")
    
    # Price-only refresh every 720 minutes (12 hours)
    scheduler.add_job(
        lambda: refresh_prices_only(SessionLocal(), EnhancedPriceFetcher(api_key=settings.COINGECKO_API_KEY)),
        "interval",
        minutes=720,
        id="price_refresh_job"
    )
    
    scheduler.start()
    logger.info("Scheduler started: RPC (24h), Prices (6h)")