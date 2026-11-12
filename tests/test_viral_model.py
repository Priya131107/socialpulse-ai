"""
Unit tests for virality prediction feature pipeline and inference engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engagement_features import EngagementFeatureExtractor
from src.features.temporal_features import TemporalFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.models.viral_model import CompositeFeaturePipeline, ViralityPredictor
from src.utils.config import settings


@pytest.fixture
def mini_posts_df() -> pd.DataFrame:
    """Fixture providing minimal dataframe for pipeline validation."""
    return pd.DataFrame([
        {
            "text": "Exciting AI launch! 🚀 #Tech #AI @TechCorp",
            "cleaned_text": "exciting ai launch :rocket: #tech #ai",
            "platform": "Twitter",
            "follower_count": 5000,
            "has_media": 1,
            "timestamp": "2026-10-15 14:00:00",
        },
        {
            "text": "Server outage today. Service is slow.",
            "cleaned_text": "server outage today service is slow",
            "platform": "LinkedIn",
            "follower_count": 250,
            "has_media": 0,
            "timestamp": "2026-10-16 09:30:00",
        },
    ])


def test_text_feature_extractor(mini_posts_df):
    extractor = TextFeatureExtractor(max_tfidf_features=50)
    extractor.fit(mini_posts_df)
    features = extractor.transform(mini_posts_df)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 2
    assert features.shape[1] == len(extractor.get_feature_names_out())
    assert "vader_compound" in extractor.get_feature_names_out()


def test_temporal_feature_extractor(mini_posts_df):
    extractor = TemporalFeatureExtractor()
    extractor.fit(mini_posts_df)
    features = extractor.transform(mini_posts_df)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 2
    assert features.shape[1] == len(extractor.get_feature_names_out())
    assert "hour_sin" in extractor.get_feature_names_out()


def test_engagement_feature_extractor(mini_posts_df):
    extractor = EngagementFeatureExtractor()
    extractor.fit(mini_posts_df)
    features = extractor.transform(mini_posts_df)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 2
    assert features.shape[1] == len(extractor.get_feature_names_out())
    assert "log_followers" in extractor.get_feature_names_out()


def test_composite_feature_pipeline(mini_posts_df):
    pipeline = CompositeFeaturePipeline(max_tfidf_features=50)
    pipeline.fit(mini_posts_df)
    matrix = pipeline.transform(mini_posts_df)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 20
    assert len(pipeline.get_feature_names_out()) == matrix.shape[1]


def test_virality_predictor_live_inference():
    """Verify inference engine returns valid predictions and probabilities."""
    if not settings.VIRAL_MODEL_PATH.exists():
        pytest.skip("Viral model not yet trained.")

    predictor = ViralityPredictor()
    post = {
        "text": "Join our live webinar on scalable deep learning architectures! #AI #ML 🚀",
        "platform": "LinkedIn",
        "follower_count": 15000,
        "has_media": 1,
        "timestamp": "2026-10-20 18:30:00",
    }
    result = predictor.predict(post)

    assert "prediction" in result
    assert result["prediction"] in ["Low", "Moderate", "Viral"]
    assert "probabilities" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

    # Probabilities should sum to approximately 1.0
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-2

    # Suggestions should be a non-empty list of strings
    assert isinstance(result["suggestions"], list)

