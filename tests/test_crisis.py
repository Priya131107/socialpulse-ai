"""
Unit tests for crisis detection engine, multi-window aggregations, and anomaly tripwires.
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest

from src.models.crisis_detector import CrisisDetector


@pytest.fixture
def mock_brand_posts() -> pd.DataFrame:
    """Fixture with normal baseline followed by a sharp crisis spike."""
    base_time = datetime(2026, 10, 20, 12, 0, 0)
    records = []

    # 100 baseline posts spread over 24 hours (mostly neutral/positive)
    for i in range(100):
        t = base_time - timedelta(hours=24) + timedelta(minutes=i * 14)
        records.append({
            "post_id": f"P_BASE_{i}",
            "text": "Great software update from @TechCorp!",
            "platform": "Twitter",
            "timestamp": t,
            "has_media": 0,
            "follower_count": 5000,
            "likes": 20,
            "shares": 5,
            "comments": 2,
            "brand": "TechCorp",
            "sentiment_label": "Positive" if i % 3 != 0 else "Negative",
        })

    # Crisis flood: 35 angry posts in the last 30 minutes
    for i in range(35):
        t = base_time - timedelta(minutes=i)
        records.append({
            "post_id": f"P_CRISIS_{i}",
            "text": "Server outage on @TechCorp! Critical bug and database leak! #BoycottTechCorp",
            "platform": "Twitter",
            "timestamp": t,
            "has_media": 1,
            "follower_count": 12000,
            "likes": 150,
            "shares": 90,
            "comments": 45,
            "brand": "TechCorp",
            "sentiment_label": "Negative",
        })

    return pd.DataFrame(records)


def test_window_feature_extraction(mock_brand_posts):
    detector = CrisisDetector()
    target_time = datetime(2026, 10, 20, 12, 0, 0)
    feats = detector.extract_window_features(mock_brand_posts, target_time)

    assert "volume_1h" in feats
    assert "volume_24h" in feats
    assert "negative_ratio_1h" in feats
    assert feats["volume_1h"] >= 35
    assert feats["negative_ratio_1h"] > 0.70


def test_crisis_detection_spike_escalation(mock_brand_posts):
    detector = CrisisDetector()
    target_time = datetime(2026, 10, 20, 12, 0, 0)
    result = detector.analyze_brand("TechCorp", mock_brand_posts, target_time)

    assert result["brand"] == "TechCorp"
    assert "crisis_risk_score" in result
    assert 0.0 <= result["crisis_risk_score"] <= 100.0

    # Under this acute crisis, score should be >= 70 and level should be Critical
    assert result["crisis_risk_score"] >= 70.0
    assert result["risk_level"] == "Critical"
    assert result["signals"]["rule_tripwire_triggered"] is True

    # Diagnostic keywords extracted
    assert isinstance(result["diagnostic_keywords"], list)
    assert len(result["diagnostic_keywords"]) > 0


def test_normal_baseline_state():
    """Verify healthy brands are categorized as Normal with low risk scores."""
    detector = CrisisDetector()
    base_time = datetime(2026, 10, 20, 12, 0, 0)

    # 40 positive/neutral posts over 24h
    records = [
        {
            "post_id": f"NORM_{i}",
            "text": "Enjoying the smooth service from @TechCorp!",
            "platform": "Twitter",
            "timestamp": base_time - timedelta(hours=i % 24),
            "has_media": 0,
            "follower_count": 2000,
            "likes": 10,
            "shares": 2,
            "comments": 1,
            "brand": "TechCorp",
            "sentiment_label": "Positive",
        }
        for i in range(40)
    ]
    df = pd.DataFrame(records)
    res = detector.analyze_brand("TechCorp", df, base_time)

    assert res["risk_level"] == "Normal"
    assert res["crisis_risk_score"] < 40.0
