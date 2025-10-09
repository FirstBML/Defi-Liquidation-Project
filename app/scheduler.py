"""
Simplified scheduler using the unified refresh endpoint
Fixed for Railway deployment
"""
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timezone
import logging
import requests
import os

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

# CRITICAL: Use Railway's internal service URL or localhost in production
# Railway services can talk to each other via internal URLs
def get_api_url():
    """Get the correct API URL for internal calls"""
    # In Railway, services should call themselves via localhost
    # External URL is only for outside access
    port = os.getenv("PORT", "8000")
    return f"http://localhost:{port}"

API_BASE_URL = get_api_url()

def scheduled_full_refresh():
    """Run full data refresh via API endpoint"""
    try:
        logger.info("="*60)
        logger.info(f"SCHEDULED FULL REFRESH: {datetime.now(timezone.utc)}")
        logger.info("="*60)
        
        url = f"{API_BASE_URL}/api/data/refresh"
        logger.info(f"Calling: {url}")
        
        response = requests.post(
            url,
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
            logger.info(f"✅ Refresh completed: {result.get('summary', {})}")
        else:
            logger.error(f"❌ Refresh failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"Scheduled refresh failed: {e}", exc_info=True)

def scheduled_price_refresh():
    """Run quick price-only refresh"""
    try:
        logger.info(f"🔄 Price refresh: {datetime.now(timezone.utc)}")
        
        url = f"{API_BASE_URL}/api/data/refresh"
        response = requests.post(
            url,
            json={"prices_only": True},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Prices updated: {result.get('summary', {}).get('prices_updated', 0)}")
        else:
            logger.error(f"❌ Price refresh failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Price refresh failed: {e}", exc_info=True)

def start_scheduler():
    """Start the scheduler with two jobs"""
    try:
        # Check if scheduler is already running
        if scheduler.running:
            logger.warning("Scheduler already running, skipping start")
            return
        
        # Full refresh every 24 hours
        scheduler.add_job(
            scheduled_full_refresh,
            "interval",
            hours=24,
            id="full_refresh",
            name="Full Data Refresh (Daily)",
            replace_existing=True
        )
        
        # Price refresh every 12 hours
        scheduler.add_job(
            scheduled_price_refresh,
            "interval",
            hours=1,
            id="price_refresh",
            name="Price Update (12h)",
            replace_existing=True
        )
        
        scheduler.start()
        logger.info("✅ Scheduler started:")
        logger.info("   - Full refresh: Every 24 hours")
        logger.info("   - Price refresh: Every 12 hours")
        
        # DON'T run initial refresh on startup - Railway times out
        # Let user trigger manually via /api/data/refresh
        logger.info("⏭️  Skipping initial refresh (trigger manually at /docs)")
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}", exc_info=True)
        # Don't raise - allow app to start even if scheduler fails