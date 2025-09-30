# alerts.py
import smtplib
import json
import logging
from email.message import EmailMessage
from typing import Dict, Any
import requests

from .config import settings  # works since config.py is in root

logger = logging.getLogger("aave_alerts")
logger.setLevel(logging.INFO)


def send_slack_alert(message: str) -> bool:
    """Send an alert to Slack using a webhook URL."""
    if not settings.SLACK_WEBHOOK_URL:
        logger.info("Slack webhook not configured")
        return False

    payload = {"text": message}
    try:
        r = requests.post(settings.SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if r.status_code in (200, 201):
            logger.info("Slack alert sent successfully")
            return True
        logger.error(f"Slack failed: {r.status_code} {r.text[:200]}")
    except Exception:
        logger.exception("Slack error")

    return False


def send_email_alert(subject: str, body: str) -> bool:
    """Send an alert email using SMTP."""
    if not (settings.SMTP_HOST and settings.ALERT_EMAIL_TO and settings.ALERT_EMAIL_FROM):
        logger.info("Email not configured")
        return False

    msg = EmailMessage()
    msg["From"] = settings.ALERT_EMAIL_FROM
    msg["To"] = settings.ALERT_EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            if settings.SMTP_USER and settings.SMTP_PASS:
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASS)
            server.send_message(msg)
        logger.info("Email alert sent successfully")
        return True
    except Exception:
        logger.exception("Email send failed")
        return False


def send_telegram_alert(message: str) -> bool:
    """Send an alert message via Telegram bot."""
    if not (settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID):
        logger.info("Telegram bot token or chat ID not configured")
        return False

    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            logger.info("Telegram alert sent successfully")
            return True
        logger.error(f"Telegram failed: {r.status_code} {r.text[:200]}")
    except Exception:
        logger.exception("Telegram send failed")

    return False


def notify_critical_alerts(alerts: Dict[str, Any]) -> None:
    """Send critical alerts via Slack, Email, and Telegram."""
    text = json.dumps(alerts, indent=2, default=str)
    logger.warning("CRITICAL ALERTS: %s", text)

    slack_ok = send_slack_alert(f"🚨 CRITICAL AAVE ALERT 🚨\n{text}")
    email_ok = send_email_alert("🚨 AAVE Risk Alerts", text)
    telegram_ok = send_telegram_alert(f"🚨 CRITICAL AAVE ALERT 🚨\n{text}")

    if not (slack_ok or email_ok or telegram_ok):
        logger.error("No alert was successfully delivered")