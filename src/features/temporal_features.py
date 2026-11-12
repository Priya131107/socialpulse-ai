"""
Temporal feature extractor for SocialPulse AI.
Extracts posting hour, cyclical sine/cosine hour encodings, day of week,
and prime-time peak interaction flags.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transforms timestamp columns into cyclical and calendar features."""

    def __init__(self):
        self.feature_names_ = [
            "hour",
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "is_weekend",
            "is_lunch_peak",
            "is_evening_peak",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(X["timestamp"])

        hour = timestamps.dt.hour.values
        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)

        day_of_week = timestamps.dt.dayofweek.values
        is_weekend = (day_of_week >= 5).astype(float)

        is_lunch_peak = ((hour >= 12) & (hour <= 14)).astype(float)
        is_evening_peak = ((hour >= 18) & (hour <= 22)).astype(float)

        features = np.column_stack([
            hour.astype(float),
            hour_sin,
            hour_cos,
            day_of_week.astype(float),
            is_weekend,
            is_lunch_peak,
            is_evening_peak,
        ])
        return features

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_

