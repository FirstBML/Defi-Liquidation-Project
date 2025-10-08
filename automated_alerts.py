#!/usr/bin/env python3
"""
Automated alert system for critical positions
Checks database and sends alerts via Slack/Email/Telegram
"""
import sys
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into os.environ

import os
print("SLACK_WEBHOOK_URL =", os.getenv("SLACK_WEBHOOK_URL"))

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_and_alert_critical_positions():
    """Check for critical positions and send alerts"""
    try:
        from app.db_models import SessionLocal, Position
        from app.alerts import send_slack_alert, send_email_alert, send_telegram_alert
        from app.config import settings
        
        session = SessionLocal()
        
        # Get critical positions
        threshold = getattr(settings, 'ALERT_HEALTH_FACTOR_THRESHOLD', 1.1)
        whale_min = getattr(settings, 'ALERT_WHALE_MIN_USD', 1000000)
        
        critical_positions = session.query(Position).filter(
            Position.enhanced_health_factor < threshold,
            Position.enhanced_health_factor.isnot(None)
        ).order_by(Position.enhanced_health_factor).all()
        
        whale_positions = session.query(Position).filter(
            Position.enhanced_health_factor < 1.5,
            Position.total_collateral_usd > whale_min,
            Position.enhanced_health_factor.isnot(None)
        ).all()
        
        session.close()
        
        if not critical_positions:
            print("No critical positions detected")
            return True
        
        # Build alert message
        alert_message = f"""🚨 CRITICAL LIQUIDATION ALERT 🚨

Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

CRITICAL POSITIONS: {len(critical_positions)}
Threshold Health Factor: {threshold}

TOP 10 MOST CRITICAL:
"""
        
        for i, pos in enumerate(critical_positions[:10], 1):
            hf = pos.enhanced_health_factor or 0
            alert_message += f"\n{i}. Borrower: {pos.borrower_address[:10]}..."
            alert_message += f"\n   Token: {pos.token_symbol} ({pos.chain})"
            alert_message += f"\n   Health Factor: {hf:.4f}"
            alert_message += f"\n   Collateral: ${pos.total_collateral_usd:,.0f}"
            alert_message += f"\n   Debt: ${pos.total_debt_usd:,.0f}"
            alert_message += f"\n   Risk: {pos.risk_category}\n"
        
        if whale_positions:
            total_whale_collateral = sum(p.total_collateral_usd for p in whale_positions)
            alert_message += f"\n\nWHALE ALERT (>${whale_min:,.0f}):"
            alert_message += f"\n{len(whale_positions)} whale positions at risk"
            alert_message += f"\nTotal whale collateral at risk: ${total_whale_collateral:,.0f}\n"
        
        alert_message += f"\n\nACTION REQUIRED: Monitor these positions closely"
        alert_message += f"\nDashboard: http://127.0.0.1:8000/api/dashboard/summary"
        
        # Send alerts
        alerts_sent = []
        
        # Slack
        if send_slack_alert(alert_message):
            alerts_sent.append("Slack")
            print("Alert sent to Slack")
        
        # Email
        subject = f"CRITICAL: {len(critical_positions)} positions at liquidation risk"
        if send_email_alert(subject, alert_message):
            alerts_sent.append("Email")
            print("Alert sent to Email")
        
        # Telegram
        if send_telegram_alert(alert_message):
            alerts_sent.append("Telegram")
            print("Alert sent to Telegram")
        
        if not alerts_sent:
            print("WARNING: No alerts were delivered. Check your config.")
            # Print to console as fallback
            print("\n" + "="*60)
            print(alert_message)
            print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Alert check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running automated alert check...")
    check_and_alert_critical_positions()