"""
Brand Crisis Status & Longitudinal History API routes.
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    CrisisDataPoint,
    CrisisHistoryResponse,
    CrisisStatusResponse,
)
from src.data.loader import DataLoader
from src.models.crisis_detector import load_crisis_detector
from src.utils.logger import logger

router = APIRouter(prefix="/crisis", tags=["Brand Crisis Monitoring"])

_cached_raw_df = None
_cached_detector = None


def get_detector_and_data():
    global _cached_raw_df, _cached_detector
    if _cached_raw_df is None:
        loader = DataLoader()
        _cached_raw_df = loader.load_raw()
        _cached_raw_df["timestamp"] = pd.to_datetime(_cached_raw_df["timestamp"])
    if _cached_detector is None:
        _cached_detector = load_crisis_detector()
    return _cached_detector, _cached_raw_df


@router.get(
    "/status",
    response_model=CrisisStatusResponse,
    summary="Get Real-Time Brand Crisis Risk Score",
)
def get_crisis_status(brand: str = Query(..., description="Target brand name, e.g. TechCorp, SwiftAir")):
    detector, df = get_detector_and_data()
    brand_df = df[df["brand"].str.lower() == brand.lower()]

    if brand_df.empty:
        raise HTTPException(status_code=404, detail=f"No posts found for brand '{brand}'.")

    try:
        latest_time = brand_df["timestamp"].max()
        analysis = detector.analyze_brand(brand, brand_df, latest_time)
        return CrisisStatusResponse(
            brand=analysis["brand"],
            timestamp=analysis["timestamp"],
            crisis_risk_score=analysis["crisis_risk_score"],
            risk_level=analysis["risk_level"],
            signals=analysis["signals"],
            metrics=analysis["metrics"],
            diagnostic_keywords=analysis["diagnostic_keywords"],
        )
    except Exception as e:
        logger.error(f"Error evaluating crisis status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history",
    response_model=CrisisHistoryResponse,
    summary="Get Historical Crisis Risk Trajectory",
)
def get_crisis_history(
    brand: str = Query(..., description="Target brand name"),
    window_hours: int = Query(24, ge=6, le=720, description="History timeline window in hours"),
):
    detector, df = get_detector_and_data()
    brand_df = df[df["brand"].str.lower() == brand.lower()].sort_values("timestamp")

    if brand_df.empty:
        raise HTTPException(status_code=404, detail=f"No posts found for brand '{brand}'.")

    try:
        max_time = brand_df["timestamp"].max()
        start_time = max_time - timedelta(hours=window_hours)
        timeline_df = brand_df[brand_df["timestamp"] >= start_time]

        step = max(1, window_hours // 24)
        sample_times = []
        curr = start_time
        while curr <= max_time:
            sample_times.append(curr)
            curr += timedelta(hours=step)

        history_points = []
        for t in sample_times:
            sub = brand_df[brand_df["timestamp"] <= t]
            if not sub.empty:
                res = detector.analyze_brand(brand, sub, t)
                history_points.append(
                    CrisisDataPoint(
                        timestamp=res["timestamp"],
                        risk_score=res["crisis_risk_score"],
                        risk_level=res["risk_level"],
                        volume_1h=res["metrics"]["volume_1h"],
                        negative_ratio_1h=res["signals"]["negative_ratio_1h"],
                    )
                )

        return CrisisHistoryResponse(
            brand=brand,
            total_data_points=len(history_points),
            history=history_points,
        )
    except Exception as e:
        logger.error(f"Error evaluating crisis history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
