"""
Multi-channel notification dispatchers for SocialPulse AI.
Includes Console, SMTP Email, and Webhook (Slack / Telegram / Discord) dispatchers
with automatic mock fallbacks when API credentials are absent.
"""

import abc
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, Optional
import httpx

from src.utils.logger import logger


class BaseNotifier(abc.ABC):
    """Abstract base class for all notification channels."""

    @abc.abstractmethod
    def send(self, alert_payload: Dict[str, Any]) -> bool:
        """Dispatch alert payload; returns True on success."""
        pass


class ConsoleNotifier(BaseNotifier):
    """Outputs alert message to system log and terminal."""

    def send(self, alert_payload: Dict[str, Any]) -> bool:
        severity = alert_payload.get("severity", "INFO").upper()
        brand = alert_payload.get("brand", "General")
        score = alert_payload.get("crisis_score", 0.0)
        message = alert_payload.get("message", "")

        banner = (
            f"\n🚨 [CRISIS ALERT: {severity}] | Brand: {brand} | Risk Score: {score}/100\n"
            f"   Message: {message}\n"
            f"   Dispatched at: {alert_payload.get('timestamp')}\n"
        )
        if severity == "CRITICAL":
            logger.critical(banner)
        elif severity == "WARNING":
            logger.warning(banner)
        else:
            logger.info(banner)
        return True


class EmailNotifier(BaseNotifier):
    """Dispatches alerts via SMTP or logs mock transmission if unconfigured."""

    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_pass = os.getenv("SMTP_PASSWORD", "")
        self.recipient = os.getenv("ALERT_EMAIL_RECIPIENT", "")

    def send(self, alert_payload: Dict[str, Any]) -> bool:
        if not (self.smtp_user and self.smtp_pass and self.recipient):
            logger.info(
                f"[MOCK EMAIL NOTIFIER] To: {self.recipient or 'ops@example.com'} | "
                f"Subject: SocialPulse Alert: {alert_payload.get('brand')} [{alert_payload.get('severity')}] | "
                f"Content: {alert_payload.get('message')}"
            )
            return True

        try:
            msg = MIMEMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = self.recipient
            msg["Subject"] = f"SocialPulse Alert: {alert_payload.get('brand')} [{alert_payload.get('severity')}]"

            body = (
                f"Crisis Alert Triggered\n"
                f"Brand: {alert_payload.get('brand')}\n"
                f"Severity: {alert_payload.get('severity')}\n"
                f"Risk Score: {alert_payload.get('crisis_score')}\n"
                f"Details: {alert_payload.get('message')}\n"
            )
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=5) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            logger.info(f"Email alert successfully sent to {self.recipient}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookNotifier(BaseNotifier):
    """Dispatches alerts to Slack / Discord / Telegram webhooks."""

    def __init__(self):
        self.slack_url = os.getenv("SLACK_WEBHOOK_URL", "")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    def send(self, alert_payload: Dict[str, Any]) -> bool:
        if not (self.slack_url or (self.telegram_token and self.telegram_chat_id)):
            logger.info(
                f"[MOCK WEBHOOK NOTIFIER] Webhook dispatch: {alert_payload.get('brand')} - "
                f"Level: {alert_payload.get('severity')} - Score: {alert_payload.get('crisis_score')}"
            )
            return True

        success = True
        # Send Slack Webhook if available
        if self.slack_url:
            try:
                slack_body = {
                    "text": (
                        f":rotating_light: *SocialPulse Crisis Alert [{alert_payload.get('severity')}]*\n"
                        f"*Brand:* {alert_payload.get('brand')}\n"
                        f"*Risk Score:* {alert_payload.get('crisis_score')}/100\n"
                        f"*Summary:* {alert_payload.get('message')}"
                    )
                }
                with httpx.Client(timeout=4.0) as client:
                    resp = client.post(self.slack_url, json=slack_body)
                    if resp.status_code >= 400:
                        success = False
            except Exception as e:
                logger.error(f"Slack webhook error: {e}")
                success = False

        # Send Telegram if available
        if self.telegram_token and self.telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                text = (
                    f"🚨 *SocialPulse Alert [{alert_payload.get('severity')}]*\n"
                    f"Brand: {alert_payload.get('brand')}\n"
                    f"Risk Score: {alert_payload.get('crisis_score')}\n"
                    f"Message: {alert_payload.get('message')}"
                )
                with httpx.Client(timeout=4.0) as client:
                    client.post(url, json={"chat_id": self.telegram_chat_id, "text": text, "parse_mode": "Markdown"})
            except Exception as e:
                logger.error(f"Telegram webhook error: {e}")
                success = False

        return success

