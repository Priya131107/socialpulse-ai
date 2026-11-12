"""
Alert manager with anti-spam cooldown throttling and SQLite audit trail.
Manages alert triggers, severity escalation, and multi-channel dispatching.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from src.alerts.notifiers import BaseNotifier, ConsoleNotifier, EmailNotifier, WebhookNotifier
from src.utils.config import settings
from src.utils.logger import logger


class AlertRecord:
    """Represents a single dispatched alert event."""

    def __init__(
        self,
        alert_id: Optional[int],
        brand: str,
        severity: str,
        crisis_score: float,
        message: str,
        keywords: str,
        timestamp: str,
    ):
        self.alert_id = alert_id
        self.brand = brand
        self.severity = severity.upper()
        self.crisis_score = float(crisis_score)
        self.message = message
        self.keywords = keywords
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "brand": self.brand,
            "severity": self.severity,
            "crisis_score": self.crisis_score,
            "message": self.message,
            "keywords": self.keywords,
            "timestamp": self.timestamp,
        }


class AlertManager:
    """Manages alert rules, cooldown throttling, and multi-channel dispatch."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        cooldown_minutes: int = 30,
        notifiers: Optional[List[BaseNotifier]] = None,
    ):
        self.db_path = db_path or settings.ALERTS_DB_PATH
        self.cooldown_minutes = cooldown_minutes
        self.notifiers = notifiers or [
            ConsoleNotifier(),
            EmailNotifier(),
            WebhookNotifier(),
        ]
        self._init_db()

    def _init_db(self):
        """Initialize SQLite alerts audit table."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    brand TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    crisis_score REAL NOT NULL,
                    message TEXT NOT NULL,
                    keywords TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def is_cooling_down(self, brand: str, severity: str) -> bool:
        """
        Check if an active cooldown applies to this brand and severity.
        If severity escalated to CRITICAL from WARNING, cooldown is bypassed.
        """
        cooldown_threshold = (
            datetime.now() - timedelta(minutes=self.cooldown_minutes)
        ).strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT severity, timestamp FROM alerts
                WHERE brand = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT 1
            """, (brand, cooldown_threshold))
            row = cursor.fetchone()

            if not row:
                return False

            last_severity, _ = row
            # If last was WARNING and now is CRITICAL, allow escalation bypass
            if last_severity == "WARNING" and severity.upper() == "CRITICAL":
                return False

            return True

    def trigger_alert(
        self,
        brand: str,
        severity: str,
        crisis_score: float,
        message: str,
        keywords: Optional[List[str]] = None,
        force: bool = False,
    ) -> Optional[AlertRecord]:
        """
        Dispatch an alert if not throttled by cooldown.
        Returns the created AlertRecord, or None if suppressed.
        """
        severity = severity.upper()
        if not force and self.is_cooling_down(brand, severity):
            logger.info(f"Alert suppressed for {brand} [{severity}] due to {self.cooldown_minutes}m cooldown.")
            return None

        kw_str = ", ".join(keywords) if keywords else ""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Persist to SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (brand, severity, crisis_score, message, keywords, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (brand, severity, crisis_score, message, kw_str, now_str))
            conn.commit()
            record_id = cursor.lastrowid

        record = AlertRecord(
            alert_id=record_id,
            brand=brand,
            severity=severity,
            crisis_score=crisis_score,
            message=message,
            keywords=kw_str,
            timestamp=now_str,
        )

        # Dispatch to all configured notifiers
        payload = record.to_dict()
        for notifier in self.notifiers:
            try:
                notifier.send(payload)
            except Exception as e:
                logger.error(f"Notifier {notifier.__class__.__name__} failed: {e}")

        return record

    def get_recent_alerts(
        self,
        limit: int = 50,
        brand: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent dispatched alerts from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if brand:
                cursor.execute("""
                    SELECT id, brand, severity, crisis_score, message, keywords, timestamp
                    FROM alerts WHERE brand = ?
                    ORDER BY id DESC LIMIT ?
                """, (brand, limit))
            else:
                cursor.execute("""
                    SELECT id, brand, severity, crisis_score, message, keywords, timestamp
                    FROM alerts
                    ORDER BY id DESC LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()

        return [
            {
                "alert_id": r[0],
                "brand": r[1],
                "severity": r[2],
                "crisis_score": r[3],
                "message": r[4],
                "keywords": r[5],
                "timestamp": r[6],
            }
            for r in rows
        ]


# Global alert manager instance
alert_manager = AlertManager()

