"""
Simplified scheduler using the unified refresh endpoint
"""
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timezone
import logging
import requests
import os

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

# API base URL (adjust for your deployment)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def scheduled_full_refresh():
    """Run full data refresh via API endpoint"""
    try:
        logger.info("="*60)
        logger.info(f"SCHEDULED FULL REFRESH: {datetime.now(timezone.utc)}")
        logger.info("="*60)
        
        response = requests.post(
            f"{API_BASE_URL}/api/data/refresh",
            json={
                "refresh_reserves": True,
                "refresh_positions": True,
                "refresh_liquidations": True,
                "prices_only": False
            },
            timeout=300  # 5 min timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Refresh completed: {result['summary']}")
        else:
            logger.error(f"❌ Refresh failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Scheduled refresh failed: {e}")

def scheduled_price_refresh():
    """Run quick price-only refresh"""
    try:
        logger.info(f"🔄 Price refresh: {datetime.now(timezone.utc)}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/data/refresh",
            json={"prices_only": True},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Prices updated: {result['summary']['prices_updated']}")
        else:
            logger.error(f"❌ Price refresh failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Price refresh failed: {e}")

def start_scheduler():
    """Start the scheduler with two jobs"""
    
    # Full refresh every 24 hours
    scheduler.add_job(
        scheduled_full_refresh,
        "interval",
        hours=24,
        id="full_refresh",
        name="Full Data Refresh (Daily)"
    )
    
    # Price refresh every 12 hours
    scheduler.add_job(
        scheduled_price_refresh,
        "interval",
        hours=12,
        id="price_refresh",
        name="Price Update (12h)"
    )
    
    scheduler.start()
    logger.info("✅ Scheduler started:")
    logger.info("   - Full refresh: Every 24 hours")
    logger.info("   - Price refresh: Every 12 hours")
    
    # Run initial refresh immediately
    logger.info("Running initial refresh...")
    scheduled_full_refresh()