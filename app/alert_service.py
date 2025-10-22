"""
Enhanced Alert Service  
Multi-channel notifications
Location: app/alert_service.py
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

# Import from existing alerts.py
from .alerts import send_email_alert, send_telegram_alert, send_slack_alert

logger = logging.getLogger(__name__)

class AlertService:
    """Enhanced alert service with multi-channel support"""
    
    def __init__(self):
        self.email_configured = bool(os.getenv("SMTP_HOST"))
        self.telegram_configured = bool(os.getenv("TELEGRAM_BOT_TOKEN"))
        self.slack_configured = bool(os.getenv("SLACK_WEBHOOK_URL"))
    
    async def send_alert(
        self,
        db: Session,
        user_id: int,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict] = None,
        wallet_address: Optional[str] = None,
        chain: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Send alert through all configured channels
        
        Returns:
            {'email': bool, 'telegram': bool, 'slack': bool}
        """
        from .db_models import User, AlertSubscription, AlertHistory
        
        # Get user and subscription
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            return {'email': False, 'telegram': False, 'slack': False}
        
        subscription = db.query(AlertSubscription).filter(
            AlertSubscription.user_id == user_id,
            AlertSubscription.is_active == True
        ).first()
        
        if not subscription:
            return {'email': False, 'telegram': False, 'slack': False}
        
        results = {
            'email': False,
            'telegram': False,
            'slack': False
        }
        
        # Format message
        formatted_message = self._format_message(severity, alert_type, message, details)
        
        # Send via enabled channels
        if subscription.channel_email and user.email and self.email_configured:
            results['email'] = send_email_alert(
                subject=f"[{severity}] {alert_type}",
                body=formatted_message
            )
        
        if subscription.channel_telegram and user.telegram_chat_id and self.telegram_configured:
            results['telegram'] = send_telegram_alert(formatted_message)
        
        if subscription.channel_slack and user.slack_webhook_url and self.slack_configured:
            results['slack'] = send_slack_alert(formatted_message)
        
        # Log alert history
        alert_record = AlertHistory(
            user_id=user_id,
            alert_type=alert_type,
            severity=severity,
            wallet_address=wallet_address,
            chain=chain,
            message=message,
            details=details,
            sent_via_email=results['email'],
            sent_via_telegram=results['telegram'],
            sent_via_slack=results['slack'],
            delivery_status='sent' if any(results.values()) else 'failed',
            created_at=datetime.now(timezone.utc),
            delivered_at=datetime.now(timezone.utc) if any(results.values()) else None
        )
        db.add(alert_record)
        db.commit()
        
        return results
    
    def _format_message(
        self,
        severity: str,
        alert_type: str,
        message: str,
        details: Optional[Dict] = None
    ) -> str:
        """Format alert message"""
        emoji = {
            'CRITICAL': '🚨',
            'HIGH': '⚠️',
            'MEDIUM': '🔶',
            'LOW': 'ℹ️'
        }.get(severity, '📢')
        
        formatted = f"{emoji} *{severity}* - {alert_type}\n\n{message}\n"
        
        if details:
            formatted += "\n*Details:*\n"
            for key, value in details.items():
                formatted += f"• {key}: {value}\n"
        
        return formatted