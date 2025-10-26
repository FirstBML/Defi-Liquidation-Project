"""
Unified Scheduler - FIXED VERSION
Location: app/scheduler.py
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timezone
import logging
import os
from typing import Optional
import requests
from sqlalchemy import func  # FIXED: Import func from sqlalchemy

logger = logging.getLogger(__name__)

def cleanup_portfolio_cache():
    """Scheduled task to clean up expired portfolio cache"""
    try:
        from app.portfolio_tracker_service import portfolio_service
        if portfolio_service and hasattr(portfolio_service, 'cache'):
            portfolio_service.cache.clear_expired()
            logger.info("✅ Portfolio cache cleanup completed")
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")


class UnifiedScheduler:
    """Centralized scheduler for all background tasks"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.running = False
        
        # Get the correct port from environment
        self.api_port = int(os.getenv('PORT', '8080'))
        self.api_base_url = f"http://localhost:{self.api_port}"
        
        logger.info(f"🔧 Scheduler configured for API at {self.api_base_url}")
    
    def start_scheduler(self):
        """Start the unified scheduler with all jobs"""
        try:
            # Job 1: Full data refresh (daily at 2 AM UTC)
            self.scheduler.add_job(
                func=self._full_data_refresh,
                trigger=CronTrigger(hour=2, minute=0, timezone='UTC'),
                id='full_data_refresh',
                name='Full Data Refresh (Daily 2:00 AM UTC)',
                replace_existing=True
            )
            
            # Job 2: Price updates only (every 6 hours)
            self.scheduler.add_job(
                func=self._price_update,
                trigger=IntervalTrigger(hours=6),
                id='price_update',
                name='Price Update (6h)',
                replace_existing=True
            )
            
            # Job 3: User address monitoring (daily at 8 AM UTC)
            self.scheduler.add_job(
                func=self._monitor_user_addresses,
                trigger=CronTrigger(hour=8, minute=0, timezone='UTC'),
                id='user_address_monitoring',
                name='User Address Monitoring (Daily 8:00 AM UTC)',
                replace_existing=True
            )
            
            # Job 4: Protocol risk analysis (daily at 10 AM UTC)
            self.scheduler.add_job(
                func=self._protocol_risk_analysis,
                trigger=CronTrigger(hour=10, minute=0, timezone='UTC'),
                id='protocol_risk_analysis',
                name='Protocol Risk Analysis (Daily 10:00 AM UTC)',
                replace_existing=True
            )
            
            # Job 5: Quick health check (every 12 hours)
            self.scheduler.add_job(
                func=self._quick_health_check,
                trigger=IntervalTrigger(hours=12),
                id='quick_health_check',
                name='Quick Health Check (12h)',
                replace_existing=True
            )
            
            # Start the scheduler
            self.scheduler.start()
            self.running = True
            
            # Log startup
            logger.info("=" * 70)
            logger.info("🚀 UNIFIED SCHEDULER STARTED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info("📅 SCHEDULED JOBS:")
            logger.info("   2:00 AM UTC - Full Data Refresh (24h)")
            logger.info("   Every 6h    - Price Updates")
            logger.info("   8:00 AM UTC - User Address Monitoring (24h)")
            logger.info("   10:00 AM UTC - Protocol Risk Analysis (24h)")
            logger.info("   Every 12h   - Quick Health Check")
            logger.info("=" * 70)
            
            # Run initial health check after a short delay
            import time
            time.sleep(3)  # Wait for API to be fully ready
            self._quick_health_check()
            
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {e}")
            raise
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        if self.running:
            self.scheduler.shutdown()
            self.running = False
            logger.info("⏹️ Unified Scheduler stopped")
    
    # ==================== JOB IMPLEMENTATIONS ====================
    
    def _full_data_refresh(self):
        """Full data refresh: reserves, positions, liquidations"""
        logger.info("🔄 Starting full data refresh...")
        
        try:
            # Get admin password from environment
            admin_password = os.getenv("ADMIN_PASSWORD", "change_me_in_production")
            
            response = requests.post(
                f"{self.api_base_url}/api/data/refresh",
                params={"password": admin_password},  # FIXED: Add password param
                json={
                    "refresh_reserves": True,
                    "refresh_positions": True,
                    "refresh_liquidations": True,
                    "prices_only": False
                },
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Full data refresh completed successfully")
                logger.info(f"   Summary: {result.get('summary', {})}")
            else:
                logger.error(f"❌ Full refresh failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            logger.error("❌ Full refresh timed out (>5 minutes)")
        except Exception as e:
            logger.error(f"❌ Full refresh error: {e}")
    
    def _price_update(self):
        """Quick price-only update"""
        logger.info("💰 Starting price update...")
        
        try:
            # Get admin password from environment
            admin_password = os.getenv("ADMIN_PASSWORD", "change_me_in_production")
            
            response = requests.post(
                f"{self.api_base_url}/api/data/refresh",
                params={"password": admin_password},  # FIXED: Add password param
                json={"prices_only": True},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                updated = result.get('results', {}).get('prices', {}).get('updated', 0)
                logger.info(f"✅ Price update completed: {updated} prices updated")
            else:
                logger.error(f"❌ Price update failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Price update error: {e}")
    
    def _monitor_user_addresses(self):
        """Monitor all user-subscribed addresses for risk changes"""
        logger.info("👥 Starting user address monitoring...")
        
        try:
            from .db_models import SessionLocal, User, MonitoredAddress
            from .portfolio_tracker_service import PortfolioTrackerService
            from .alert_service import AlertService
            
            db = SessionLocal()
            portfolio_service = PortfolioTrackerService()
            alert_service = AlertService()
            
            # Get all active monitored addresses
            monitored = db.query(MonitoredAddress).filter(
                MonitoredAddress.is_active == True
            ).all()
            
            logger.info(f"   Checking {len(monitored)} monitored addresses...")
            
            alerts_sent = 0
            for addr in monitored:
                try:
                    # Check for risk changes
                    risk_changes = portfolio_service.check_address_risk_changes(
                        db=db,
                        monitored_address_id=addr.id,
                        wallet_address=addr.wallet_address
                    )
                    
                    # Send alerts if needed
                    if risk_changes.get('should_alert'):
                        alert_service.send_alert(
                            db=db,
                            user_id=addr.user_id,
                            alert_type='RISK_CHANGE',
                            severity=risk_changes.get('severity', 'MEDIUM'),
                            message=risk_changes.get('message'),
                            details=risk_changes
                        )
                        alerts_sent += 1
                        
                except Exception as addr_error:
                    logger.error(f"   Error checking {addr.wallet_address}: {addr_error}")
                    continue
            
            db.close()
            logger.info(f"✅ User monitoring completed: {alerts_sent} alerts sent")
            
        except Exception as e:
            logger.error(f"❌ User monitoring error: {e}")
    
    def _protocol_risk_analysis(self):
        """Generate and store protocol-wide risk snapshots"""
        logger.info("📊 Starting protocol risk analysis...")
        
        try:
            from .db_models import SessionLocal, AnalysisSnapshot, Position, Reserve
            
            db = SessionLocal()
            
            # Get current metrics
            total_positions = db.query(Position).count()
            risky_positions = db.query(Position).filter(
                Position.enhanced_health_factor < 1.5,
                Position.enhanced_health_factor > 0
            ).count()
            
            # FIXED: Use func from sqlalchemy, not db.func
            total_collateral = db.query(
                func.sum(Position.total_collateral_usd)
            ).scalar() or 0
            
            total_debt = db.query(
                func.sum(Position.total_debt_usd)
            ).scalar() or 0
            
            # Calculate health score
            health_score = self._calculate_health_score(
                total_positions, risky_positions, total_collateral, total_debt
            )
            
            # Store snapshot
            snapshot = AnalysisSnapshot(
                snapshot_time=datetime.now(timezone.utc),
                total_positions=total_positions,
                risky_positions=risky_positions,
                total_collateral_usd=float(total_collateral),
                total_debt_usd=float(total_debt),
                protocol_health_score=float(health_score)
            )
            
            db.add(snapshot)
            db.commit()
            db.close()
            
            logger.info(f"✅ Protocol analysis completed")
            logger.info(f"   Positions: {total_positions} (risky: {risky_positions})")
            logger.info(f"   TVL: ${total_collateral:,.0f}, Debt: ${total_debt:,.0f}")
            logger.info(f"   Health Score: {health_score:.1f}/100")
            
        except Exception as e:
            logger.error(f"❌ Protocol analysis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _quick_health_check(self):
        """Quick system health check"""
        logger.info("❤️  Running quick health check...")
        
        try:
            from .db_models import SessionLocal
            from sqlalchemy import text
            
            # Check database
            db_healthy = False
            try:
                db = SessionLocal()
                db.execute(text("SELECT 1"))
                db.close()
                db_healthy = True
            except Exception as db_error:
                logger.error(f"   Database: ❌ {db_error}")
            
            if db_healthy:
                logger.info("   Database: ✅ Healthy")
            
            # Check API
            api_healthy = False
            try:
                response = requests.get(
                    f"{self.api_base_url}/api/health",
                    timeout=10
                )
                api_healthy = response.status_code == 200
            except Exception as api_error:
                logger.info(f"   API: ❌ Error: {api_error}")
            
            if api_healthy:
                logger.info("   API: ✅ Healthy")
            
            logger.info("✅ Health check completed")
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
    
    def _calculate_health_score(self, total_pos: int, risky_pos: int, 
                                collateral: float, debt: float) -> float:
        """Calculate simple protocol health score (0-100)"""
        if total_pos == 0:
            return 100.0
        
        score = 100.0
        
        # Deduct for risky positions
        risky_ratio = risky_pos / total_pos
        score -= risky_ratio * 40
        
        # Deduct for high LTV
        if collateral > 0:
            ltv = debt / collateral
            if ltv > 0.7:
                score -= (ltv - 0.7) * 100
        
        return max(0.0, min(100.0, score))

# Global scheduler instance
unified_scheduler = UnifiedScheduler()

# Export for use in main.py
__all__ = ['unified_scheduler']