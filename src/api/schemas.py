"""
Pydantic v2 schemas and validation models for SocialPulse AI API contracts.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Health Schemas ---
class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str = "SocialPulse AI"
    version: str = "1.0.0"
    models_loaded: Dict[str, bool]
    database_connected: bool
    timestamp: str


# --- Engagement Prediction Schemas ---
class PostEngagementRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Raw social post text content")
    platform: str = Field("Twitter", description="Platform: Twitter, Instagram, LinkedIn, Reddit")
    follower_count: int = Field(5000, ge=0, description="Account follower reach")
    has_media: int = Field(0, ge=0, le=1, description="1 if image/video attached, else 0")
    timestamp: Optional[str] = Field(None, description="Scheduled post timestamp (YYYY-MM-DD HH:MM:SS)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Excited to launch our new open-source agent framework! Check it out: https://pulse.io #AI #OpenSource 🚀",
                "platform": "Twitter",
                "follower_count": 25000,
                "has_media": 1,
                "timestamp": "2026-10-15 14:00:00",
            }
        }


class FeatureImpact(BaseModel):
    feature: str
    impact: float
    direction: str


class PostEngagementResponse(BaseModel):
    prediction: str = Field(..., description="Predicted tier: Low, Moderate, Viral")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution")
    confidence: float = Field(..., description="Confidence score for top class")
    top_contributing_features: List[FeatureImpact] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    model_used: str


# --- Sentiment Schemas ---
class SentimentRequest(BaseModel):
    text: Optional[str] = Field(None, description="Single text post to analyze")
    texts: Optional[List[str]] = Field(None, description="Batch list of texts to analyze")
    engine: Optional[str] = Field("vader", description="Sentiment engine: 'vader' or 'transformer'")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The new developer platform update is remarkably fast and intuitive! Loving it! 🚀",
                "engine": "vader",
            }
        }


class SentimentScores(BaseModel):
    pos: float
    neu: float
    neg: float
    compound: Optional[float] = None


class SentimentItemResponse(BaseModel):
    label: str
    confidence: float
    scores: SentimentScores
    engine: str


class SentimentResponse(BaseModel):
    results: List[SentimentItemResponse]


# --- Crisis Schemas ---
class CrisisStatusResponse(BaseModel):
    brand: str
    timestamp: str
    crisis_risk_score: float
    risk_level: str
    signals: Dict[str, Any]
    metrics: Dict[str, Any]
    diagnostic_keywords: List[str]


class CrisisDataPoint(BaseModel):
    timestamp: str
    risk_score: float
    risk_level: str
    volume_1h: int
    negative_ratio_1h: float


class CrisisHistoryResponse(BaseModel):
    brand: str
    total_data_points: int
    history: List[CrisisDataPoint]


# --- Alerts Schemas ---
class AlertTestRequest(BaseModel):
    brand: str = Field("TechCorp", description="Brand name")
    severity: str = Field("WARNING", description="Severity: INFO, WARNING, CRITICAL")
    crisis_score: float = Field(65.0, ge=0.0, le=100.0)
    message: str = Field("Simulated alert test triggered from API")
    keywords: Optional[List[str]] = Field(default=["test", "outage", "api"])


class AlertTestResponse(BaseModel):
    success: bool
    status: str
    alert_record: Optional[Dict[str, Any]] = None
