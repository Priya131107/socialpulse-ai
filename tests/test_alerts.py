"""
Unit tests for alert manager, cooldown throttling, escalation, and notifiers.
"""

import tempfile
from pathlib import Path
import pytest

from src.alerts.alert_manager import AlertManager
from src.alerts.notifiers import ConsoleNotifier, EmailNotifier, WebhookNotifier


@pytest.fixture
def temp_alert_manager():
    """Fixture providing an isolated SQLite database for alert testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    mgr = AlertManager(db_path=temp_db, cooldown_minutes=15)
    yield mgr

    # Cleanup
    p = Path(temp_db)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


def test_console_notifier():
    notifier = ConsoleNotifier()
    success = notifier.send({
        "brand": "TechCorp",
        "severity": "CRITICAL",
        "crisis_score": 85.0,
        "message": "Major server outage detected",
        "timestamp": "2026-10-20 14:00:00",
    })
    assert success is True


def test_mock_email_and_webhook_notifiers():
    email_notifier = EmailNotifier()
    webhook_notifier = WebhookNotifier()

    payload = {
        "brand": "SwiftAir",
        "severity": "WARNING",
        "crisis_score": 55.0,
        "message": "Spike in luggage complaint threads",
        "timestamp": "2026-10-20 15:00:00",
    }
    assert email_notifier.send(payload) is True
    assert webhook_notifier.send(payload) is True


def test_alert_manager_trigger_and_audit(temp_alert_manager):
    record = temp_alert_manager.trigger_alert(
        brand="TechCorp",
        severity="WARNING",
        crisis_score=52.5,
        message="Negative buzz building up",
        keywords=["latency", "outage"],
    )

    assert record is not None
    assert record.brand == "TechCorp"
    assert record.severity == "WARNING"
    assert record.crisis_score == 52.5

    alerts = temp_alert_manager.get_recent_alerts(limit=10)
    assert len(alerts) == 1
    assert alerts[0]["brand"] == "TechCorp"


def test_alert_cooldown_throttling(temp_alert_manager):
    # First alert should trigger
    r1 = temp_alert_manager.trigger_alert(
        brand="NovaPay",
        severity="WARNING",
        crisis_score=48.0,
        message="Payment checkout delay",
    )
    assert r1 is not None

    # Immediate second alert for same brand and severity should be throttled
    r2 = temp_alert_manager.trigger_alert(
        brand="NovaPay",
        severity="WARNING",
        crisis_score=50.0,
        message="Payment checkout delay continuation",
    )
    assert r2 is None

    # But escalation to CRITICAL bypasses cooldown
    r3 = temp_alert_manager.trigger_alert(
        brand="NovaPay",
        severity="CRITICAL",
        crisis_score=88.0,
        message="Emergency: Payment gateways completely unresponsive!",
    )
    assert r3 is not None
    assert r3.severity == "CRITICAL"

