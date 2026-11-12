"""
Pre-publish engagement & metadata feature extractor for SocialPulse AI.
Extracts account reach, media attachment indicators, and platform one-hot encodings.
Strictly excludes raw response metrics (likes, shares, comments) to prevent data leakage.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PLATFORMS = ["Twitter", "Instagram", "LinkedIn", "Reddit"]

class EngagementFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts pre-publish reach signals and media metadata."""

    def __init__(self):
        self.feature_names_: List[str] = [
            "log_followers",
            "has_media",
            "is_micro_creator",
            "is_macro_creator",
        ] + [f"platform_{p.lower()}" for p in PLATFORMS]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        followers = X["follower_count"].fillna(0).values
        log_followers = np.log1p(np.maximum(0, followers))

        has_media = X["has_media"].fillna(0).astype(float).values

        # Creator tier thresholds
        is_micro = (followers < 10_000).astype(float)
        is_macro = (followers >= 100_000).astype(float)

        # One-hot platform encodings
        platform_series = X["platform"].fillna("Twitter").astype(str)
        platform_cols = [
            (platform_series == p).astype(float).values
            for p in PLATFORMS
        ]

        features = np.column_stack([
            log_followers,
            has_media,
            is_micro,
            is_macro,
            *platform_cols,
        ])
        return features

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_

