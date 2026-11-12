"""
Inference module for SocialPulse AI.
Exposes `predict_engagement(text, metadata)` to predict engagement class,
class probabilities, and top contributing features for any new post.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.preprocess import clean_text
from src.features import (
    extract_text_statistics,
    extract_sentiment_scores,
    parse_timestamp_features,
    build_single_instance_df,
)

# Global in-memory cache for artifacts to ensure low latency
_CACHED_PIPELINE = None
_CACHED_MODEL = None
_CACHED_METADATA = None


def get_cached_pipeline_and_model():
    """
    Loads and caches the fitted feature pipeline, best model, and metadata.
    """
    global _CACHED_PIPELINE, _CACHED_MODEL, _CACHED_METADATA

    if _CACHED_PIPELINE is None:
        if not config.FEATURE_PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Feature pipeline not found at {config.FEATURE_PIPELINE_PATH}. Run src/train.py first."
            )
        _CACHED_PIPELINE = joblib.load(config.FEATURE_PIPELINE_PATH)

    if _CACHED_MODEL is None:
        if not config.BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Best model not found at {config.BEST_MODEL_PATH}. Run src/train.py first."
            )
        _CACHED_MODEL = joblib.load(config.BEST_MODEL_PATH)

    if _CACHED_METADATA is None:
        if config.METADATA_PATH.exists():
            _CACHED_METADATA = json.loads(config.METADATA_PATH.read_text())
        else:
            _CACHED_METADATA = {
                "best_model_name": "XGBoost",
                "class_mapping": config.CLASS_MAPPING,
            }

    return _CACHED_PIPELINE, _CACHED_MODEL, _CACHED_METADATA


def explain_instance_predictions(
    model,
    pipeline,
    X_inst,
    text: str,
    metadata: Dict[str, Any],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Identifies the top contributing features for a single prediction instance.
    Analyzes active non-zero features and weights.
    """
    feature_names = pipeline.get_feature_names_out()
    inst_dense = X_inst.toarray()[0]

    # Non-zero active features in this post
    active_indices = np.where(inst_dense != 0)[0]

    contributions = []

    # Get model importance vector if available
    if hasattr(model, "feature_importances_"):
        global_imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        global_imp = np.mean(np.abs(model.coef_), axis=0)
    else:
        global_imp = np.ones(len(feature_names))

    for idx in active_indices:
        feat_name = feature_names[idx]
        val = inst_dense[idx]
        weight = float(global_imp[idx]) if idx < len(global_imp) else 0.0
        impact_score = abs(val) * weight

        # Human-readable description
        desc = feat_name
        if feat_name.startswith("tfidf_"):
            token = feat_name.replace("tfidf_", "")
            desc = f"Keyword/Phrase: '{token}'"
        elif feat_name == "media_video":
            desc = "Video attachment (high visual retention)"
        elif feat_name == "media_image":
            desc = "Image attachment (increased click-through)"
        elif feat_name == "log_followers":
            desc = f"Follower reach ({metadata.get('followers', 1000):,} followers)"
        elif feat_name == "vader_compound":
            desc = f"Sentiment polarity (Score: {metadata.get('vader_compound', 0.0):.2f})"
        elif feat_name == "question_count":
            desc = "Discussion question hook (drives replies)"
        elif feat_name == "hashtag_count":
            desc = f"Hashtag strategy ({metadata.get('hashtag_count', 0)} tags)"
        elif feat_name == "hour_of_day":
            desc = f"Posting schedule (Hour {metadata.get('hour_of_day', 12)}:00)"

        contributions.append({
            "feature": feat_name,
            "description": desc,
            "raw_value": round(float(val), 4),
            "importance_score": round(float(impact_score), 4),
        })

    # Sort by impact score
    contributions.sort(key=lambda x: x["importance_score"], reverse=True)
    return contributions[:top_k]


def generate_virality_advice(
    predicted_class: str,
    stats: Dict[str, Any],
    sentiment: Dict[str, Any],
    time_feat: Dict[str, int],
    media_type: str,
    followers: int,
) -> List[str]:
    """
    Generates actionable ML-backed recommendations to maximize post engagement.
    """
    advice = []

    if media_type == "none":
        advice.append("Attach high-quality media (video or infographic) to increase visibility by 40-70%.")

    if stats["hashtag_count"] == 0:
        advice.append("Include 2-3 focused, trending hashtags to enhance algorithmic discovery.")
    elif stats["hashtag_count"] > 5:
        advice.append("Reduce hashtags to 2-4 focused tags to avoid spam penalties.")

    if stats["question_count"] == 0:
        advice.append("Add a closing question or poll prompt to spark discussion in the replies.")

    hour = time_feat["hour_of_day"]
    if not (12 <= hour <= 14 or 18 <= hour <= 22):
        advice.append("Schedule this post during peak engagement windows (12:00-14:00 or 18:00-21:00).")

    if abs(sentiment["vader_compound"]) < 0.20:
        advice.append("Strengthen your hook or thesis; posts with confident sentiment drive higher shares.")

    if stats["word_count"] < 10:
        advice.append("Expand on your point with context, a data point, or a mini-breakdown.")

    if not advice:
        advice.append("Post is well-optimized across media, timing, hashtags, and engagement hooks!")

    return advice


def predict_engagement(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Predicts social media engagement class for a given post.

    Parameters:
    -----------
    text : str
        The raw post text.
    metadata : dict, optional
        Dictionary containing:
        - 'followers': int (account follower count, default 1000)
        - 'timestamp': str or datetime (default now)
        - 'media_type': 'none', 'image', or 'video' (default 'none')

    Returns:
    --------
    dict containing:
        - predicted_class: 'Low' | 'Moderate' | 'Viral'
        - predicted_target: int (0, 1, 2)
        - probabilities: dict of class -> probability
        - confidence: float
        - top_features: list of top contributing features
        - metadata_summary: computed post characteristics
        - virality_advice: recommendations for improvement
    """
    if metadata is None:
        metadata = {}

    pipeline, model, meta_info = get_cached_pipeline_and_model()

    # Fallback defaults
    followers = int(metadata.get("followers", 1000))
    timestamp = metadata.get("timestamp", datetime.now().isoformat())
    media_type = str(metadata.get("media_type", "none")).lower()
    if media_type not in config.VALID_MEDIA_TYPES:
        media_type = "none"

    # Extract internal metadata
    stats = extract_text_statistics(text)
    sentiment = extract_sentiment_scores(text)
    time_feat = parse_timestamp_features(timestamp)

    full_meta = {
        "followers": followers,
        "timestamp": timestamp,
        "media_type": media_type,
        "clean_text": clean_text(text),
        **stats,
        **sentiment,
        **time_feat,
    }

    # Build DataFrame and transform features
    inst_df = build_single_instance_df(text, full_meta)
    X_inst = pipeline.transform(inst_df)

    # Predict
    pred_idx = int(model.predict(X_inst)[0])
    pred_class = config.CLASS_MAPPING.get(pred_idx, "Moderate")

    # Probabilities
    if hasattr(model, "predict_proba"):
        prob_vec = model.predict_proba(X_inst)[0]
    else:
        # Logistic / linear fallback
        prob_vec = [1/3, 1/3, 1/3]

    probabilities = {
        cls_name: round(float(prob_vec[i]), 4)
        for i, cls_name in enumerate(config.TARGET_CLASSES)
    }
    confidence = float(prob_vec[pred_idx])

    # Top contributing features
    top_feats = explain_instance_predictions(model, pipeline, X_inst, text, full_meta, top_k=5)

    # Actionable advice
    advice = generate_virality_advice(pred_class, stats, sentiment, time_feat, media_type, followers)

    return {
        "predicted_class": pred_class,
        "predicted_target": pred_idx,
        "probabilities": probabilities,
        "confidence": round(confidence, 4),
        "top_features": top_feats,
        "metadata_summary": full_meta,
        "virality_advice": advice,
    }


if __name__ == "__main__":
    # Test sample prediction
    sample_text = (
        "Just open-sourced our multi-agent reasoning framework in Python 3.14! "
        "Benchmarks show a 4x latency drop on edge hardware. 🚀🔥 #AI #MachineLearning #Python"
    )
    sample_meta = {
        "followers": 25000,
        "media_type": "video",
        "timestamp": "2026-10-05 19:30:00",
    }
    result = predict_engagement(sample_text, sample_meta)
    print("Inference Test Result:")
    print(json.dumps(result, indent=2))
