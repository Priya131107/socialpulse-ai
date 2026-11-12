"""
Integration tests for FastAPI REST API endpoints using TestClient.
"""

from fastapi.testclient import TestClient
import pytest

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "SocialPulse AI"
    assert "/docs" in data["docs_url"]


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "models_loaded" in data
    assert "database_connected" in data


def test_predict_engagement_endpoint():
    payload = {
        "text": "Excited to launch our new open source AI developer kit! Check it out: https://pulse.io #AI #Dev 🚀",
        "platform": "Twitter",
        "follower_count": 15000,
        "has_media": 1,
        "timestamp": "2026-10-15 14:00:00",
    }
    response = client.post("/predict/engagement", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in ["Low", "Moderate", "Viral"]
    assert "probabilities" in data
    assert "confidence" in data
    assert "suggestions" in data
    assert len(data["suggestions"]) > 0


def test_analyze_sentiment_single_endpoint():
    payload = {
        "text": "The new update resolved all our issues. Truly outstanding performance! 🚀",
        "engine": "vader",
    }
    response = client.post("/analyze/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["label"] == "Positive"


def test_analyze_sentiment_batch_endpoint():
    payload = {
        "texts": [
            "Phenomenal product, love it!",
            "Scheduled system maintenance tonight.",
            "Worst customer service ever encountered.",
        ],
        "engine": "vader",
    }
    response = client.post("/analyze/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert len(data["results"]) == 3
    assert data["results"][0]["label"] == "Positive"
    assert data["results"][2]["label"] == "Negative"


def test_crisis_status_endpoint():
    response = client.get("/crisis/status?brand=TechCorp")
    assert response.status_code == 200
    data = response.json()

    assert data["brand"] == "TechCorp"
    assert "crisis_risk_score" in data
    assert 0.0 <= data["crisis_risk_score"] <= 100.0
    assert data["risk_level"] in ["Normal", "Warning", "Critical"]
    assert "diagnostic_keywords" in data


def test_crisis_history_endpoint():
    response = client.get("/crisis/history?brand=TechCorp&window_hours=24")
    assert response.status_code == 200
    data = response.json()

    assert data["brand"] == "TechCorp"
    assert "history" in data
    assert isinstance(data["history"], list)


def test_alert_test_endpoint():
    payload = {
        "brand": "TechCorp",
        "severity": "WARNING",
        "crisis_score": 58.0,
        "message": "Automated API alert integration test",
        "keywords": ["test", "api"],
    }
    response = client.post("/alerts/test", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["status"] == "dispatched"
    assert data["alert_record"]["brand"] == "TechCorp"
