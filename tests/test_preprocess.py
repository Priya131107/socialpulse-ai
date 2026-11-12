"""
Unit tests for data preprocessing and cleaning pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader, REQUIRED_RAW_COLUMNS, REQUIRED_PROCESSED_COLUMNS
from src.data.preprocess import TextPreprocessor


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture providing raw posts dataframe."""
    return pd.DataFrame([
        {
            "post_id": f"POST_{i:04d}",
            "text": f"Check out this update! 🚀 https://example.com/test @Brand #AI #{i}",
            "platform": "Twitter",
            "timestamp": "2026-10-15 14:30:00",
            "hashtags_count": 2,
            "mentions_count": 1,
            "has_media": 1,
            "follower_count": 1000 + (i * 10),
            "likes": 50 + (i * 5),
            "shares": 10 + i,
            "comments": 5 + i,
            "brand": "TechCorp",
            "sentiment_label": "Positive",
        }
        for i in range(100)
    ])


def test_clean_text():
    preprocessor = TextPreprocessor()
    raw_text = "BREAKING NEWS!!! Check this out: https://t.co/abc123xyz @TechCorp 🚀🔥"
    cleaned = preprocessor.clean_text(raw_text)

    # URL should be removed
    assert "https://" not in cleaned
    assert "t.co" not in cleaned

    # Emojis should be demojized into words
    assert "rocket" in cleaned or "fire" in cleaned

    # Punctuation redundancy normalized
    assert "!!!" not in cleaned

    # Lowercase
    assert "breaking news" in cleaned


def test_engagement_score_calculation(sample_raw_data):
    preprocessor = TextPreprocessor()
    scores = preprocessor.calculate_engagement_score(sample_raw_data)

    assert len(scores) == len(sample_raw_data)
    assert (scores >= 0).all()

    # Manual verification for first row:
    # likes=50, shares=10, comments=5, followers=1000
    # weighted = 50 + 2*10 + 1.5*5 = 50 + 20 + 7.5 = 77.5
    # score = (77.5 / (1000 + 100)) * 1000 = (77.5 / 1100) * 1000 = ~70.4545
    expected_first = (77.5 / 1100.0) * 1000.0
    assert abs(scores.iloc[0] - expected_first) < 1e-4


def test_engagement_labeling(sample_raw_data):
    preprocessor = TextPreprocessor(percentiles=(0.50, 0.85))
    df_labeled, cutoffs = preprocessor.assign_engagement_labels(sample_raw_data)

    assert "engagement_label" in df_labeled.columns
    unique_labels = set(df_labeled["engagement_label"].unique())
    assert unique_labels == {"Low", "Moderate", "Viral"}

    # Low should be roughly 50%
    low_pct = (df_labeled["engagement_label"] == "Low").mean()
    assert 0.40 <= low_pct <= 0.60


def test_stratified_split(sample_raw_data):
    preprocessor = TextPreprocessor()
    df_processed, _ = preprocessor.preprocess_dataset(sample_raw_data)
    train_df, val_df, test_df = preprocessor.split_data(df_processed, 0.70, 0.15, 0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(sample_raw_data)

    # Check no ID leakage
    train_ids = set(train_df["post_id"])
    val_ids = set(val_df["post_id"])
    test_ids = set(test_df["post_id"])

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


def test_schema_validation(sample_raw_data):
    loader = DataLoader()
    assert loader.validate_schema(sample_raw_data, REQUIRED_RAW_COLUMNS) is True

    # Intentionally missing column should raise ValueError
    broken_df = sample_raw_data.drop(columns=["likes"])
    with pytest.raises(ValueError):
        loader.validate_schema(broken_df, REQUIRED_RAW_COLUMNS)

