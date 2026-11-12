"""
End-to-End System Integration Test for SocialPulse AI.
Verifies the complete flow: Data Ingestion -> Preprocessing -> Feature Extraction
-> Virality Prediction -> Sentiment Analysis -> Crisis Detection -> Alert Dispatch.
"""

from datetime import datetime
import pandas as pd
import pytest

from src.alerts.alert_manager import alert_manager
from src.data.loader import DataLoader
from src.data.preprocess import TextPreprocessor
from src.models.crisis_detector import load_crisis_detector
from src.models.sentiment_model import predict_sentiment
from src.models.viral_model import ViralityPredictor


def test_full_socialpulse_end_to_end_flow():
    # 1. Ingestion & Preprocessing
    preprocessor = TextPreprocessor()
    raw_post = {
        "post_id": "TEST_E2E_001",
        "text": "Terrible service from @SwiftAir! Flight delayed, rude staff, lost my bag! Disgusting experience! 😡",
        "platform": "Twitter",
        "timestamp": "2026-10-25 15:30:00",
        "hashtags_count": 1,
        "mentions_count": 1,
        "has_media": 1,
        "follower_count": 28000,
        "likes": 320,
        "shares": 180,
        "comments": 95,
        "brand": "SwiftAir",
        "sentiment_label": "Negative",
    }
    df_raw = pd.DataFrame([raw_post])
    df_proc, _ = preprocessor.preprocess_dataset(df_raw, cutoffs=(50.0, 150.0))

    assert "cleaned_text" in df_proc.columns
    assert "engagement_score" in df_proc.columns
    assert "engagement_label" in df_proc.columns
    assert df_proc["engagement_label"].iloc[0] in ["Low", "Moderate", "Viral"]

    # 2. Virality Prediction Engine
    predictor = ViralityPredictor()
    v_res = predictor.predict(raw_post)
    assert v_res["prediction"] in ["Low", "Moderate", "Viral"]
    assert 0.0 <= v_res["confidence"] <= 1.0
    assert len(v_res["probabilities"]) == 3
    assert len(v_res["suggestions"]) > 0

    # 3. Sentiment Analysis
    s_res = predict_sentiment(raw_post["text"])
    assert s_res["label"] == "Negative"
    assert s_res["confidence"] > 0.60

    # 4. Crisis Detection
    detector = load_crisis_detector()
    c_res = detector.analyze_brand("SwiftAir", df_raw)
    assert 0.0 <= c_res["crisis_risk_score"] <= 100.0
    assert c_res["risk_level"] in ["Normal", "Warning", "Critical"]

    # 5. Alert Triggering & Audit Trail
    alert_record = alert_manager.trigger_alert(
        brand="SwiftAir",
        severity="CRITICAL",
        crisis_score=85.0,
        message="End-to-end integration test alert",
        keywords=["flight", "cancel"],
        force=True,
    )
    assert alert_record is not None
    assert alert_record.brand == "SwiftAir"

    recent = alert_manager.get_recent_alerts(limit=5, brand="SwiftAir")
    assert len(recent) > 0
    assert recent[0]["brand"] == "SwiftAir"
