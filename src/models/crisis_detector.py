"""
SocialPulse AI — Simple Crisis Detector.
Uses rule-based thresholds on rolling negative-sentiment ratio.
No IsolationForest, no Z-score, no complex aggregation.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import settings
from src.utils.logger import logger


def compute_crisis_status(df: pd.DataFrame, brand: str) -> dict:
    """
    Compute crisis risk for a brand from a dataframe of posts.
    df must have columns: brand, sentiment_label, timestamp, likes, shares, comments.
    """
    brand_df = df[df["brand"] == brand].copy()
    if brand_df.empty:
        return {"brand": brand, "risk_level": "Normal", "crisis_risk_score": 0,
                "negative_ratio": 0.0, "total_posts": 0}

    total = len(brand_df)
    neg   = (brand_df["sentiment_label"].str.lower() == "negative").sum()
    neg_ratio = neg / total if total > 0 else 0.0

    # Simple score: 0–100
    score = int(min(100, neg_ratio * 200))

    if score >= 60:
        level = "Critical"
    elif score >= 30:
        level = "Warning"
    else:
        level = "Normal"

    return {
        "brand":             brand,
        "risk_level":        level,
        "crisis_risk_score": score,
        "negative_ratio":    round(neg_ratio, 4),
        "total_posts":       int(total),
        "negative_posts":    int(neg),
    }


def load_crisis_detector():
    """Return a simple callable wrapper for crisis detection."""
    return compute_crisis_status
