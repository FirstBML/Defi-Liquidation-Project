# config.py (simple version without pydantic)
import os
from typing import Optional

class Settings:
    """Simple settings class without pydantic dependency"""
    
    def __init__(self):
        # Database
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./risk_analysis.db")
        
        # API Keys
        self.ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
        self.COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
        self.DUNE_API_KEY_CURRENT_POSITION = os.getenv("DUNE_API_KEY_CURRENT_POSITION")
        self.DUNE_API_KEY_LIQUIDATION_HISTORY = os.getenv("DUNE_API_KEY_LIQUIDATION_HISTORY")
        self.DUNE_API_KEY_RESERVE = os.getenv("DUNE_API_KEY_RESERVE")
        
        # Alerts
        self.SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
        
        # Email alert settings
        self.SMTP_HOST = os.getenv("SMTP_HOST")
        self.SMTP_PORT = int(os.getenv("SMTP_PORT", "0")) if os.getenv("SMTP_PORT") else None
        self.SMTP_USER = os.getenv("SMTP_USER")
        self.SMTP_PASS = os.getenv("SMTP_PASS")
        self.ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
        self.ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO")
        
        # Telegram alert settings
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
 
        # Risk thresholds
        self.ALERT_HEALTH_FACTOR_THRESHOLD = float(os.getenv("ALERT_HEALTH_FACTOR_THRESHOLD", "1.1"))
        self.ALERT_WHALE_MIN_USD = float(os.getenv("ALERT_WHALE_MIN_USD", "1000000"))
        
        # Scheduler
        self.SCHEDULE_INTERVAL_MINUTES = int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "1440"))
        
        # Price caching
        self.PRICE_CACHE_TTL = int(os.getenv("PRICE_CACHE_TTL", "300"))
    
        # Stress test scenarios
        self.SCENARIOS = {
            "10%_drop": 0.10,
            "20%_drop": 0.20, 
            "30%_drop": 0.30
        }

# Singleton instance
_settings = None

def get_settings() -> Settings:
    """Get settings singleton instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# For backward compatibility
settings = get_settings()
ADMIN_API_KEY = settings.ADMIN_API_KEY