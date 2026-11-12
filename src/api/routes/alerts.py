"""
Alert dispatch and testing API routes.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.alerts.alert_manager import alert_manager
from src.api.schemas import AlertTestRequest, AlertTestResponse
from src.utils.logger import logger

router = APIRouter(prefix="/alerts", tags=["Alerts & Notifications"])


@router.post(
    "/test",
    response_model=AlertTestResponse,
    summary="Trigger Test Alert Dispatch",
)
def test_alert(payload: AlertTestRequest):
    """
    Test alert dispatch across configured notifiers (Console, Email, Webhook).
    """
    try:
        record = alert_manager.trigger_alert(
            brand=payload.brand,
            severity=payload.severity,
            crisis_score=payload.crisis_score,
            message=payload.message,
            keywords=payload.keywords,
            force=True,  # Force bypass cooldown during test
        )
        return AlertTestResponse(
            success=True,
            status="dispatched",
            alert_record=record.to_dict() if record else None,
        )
    except Exception as e:
        logger.error(f"Failed to dispatch test alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/recent",
    summary="Get Recent Dispatched Alerts Audit Trail",
)
def get_recent_alerts(
    brand: Optional[str] = Query(None, description="Optional brand filter"),
    limit: int = Query(25, ge=1, le=100),
):
    try:
        return alert_manager.get_recent_alerts(limit=limit, brand=brand)
    except Exception as e:
        logger.error(f"Error fetching recent alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
