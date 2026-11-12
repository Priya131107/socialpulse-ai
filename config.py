"""
Configuration module for SocialPulse AI - Post Engagement & Virality Prediction.
Provides centralized path management, hyperparameter configurations,
and engagement threshold settings.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Base paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

# File paths
RAW_DATASET_PATH: Path = RAW_DATA_DIR / "social_engagement_data.csv"
PROCESSED_DATASET_PATH: Path = PROCESSED_DATA_DIR / "processed_engagement_data.csv"
TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH: Path = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"

# Saved Model & Artifact Paths
BEST_MODEL_PATH: Path = MODELS_DIR / "best_engagement_model.joblib"
FEATURE_PIPELINE_PATH: Path = MODELS_DIR / "feature_pipeline.joblib"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.joblib"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_CSV: Path = REPORTS_DIR / "model_comparison.csv"

# Ensure all directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED: int = 42

# Engagement Target Configuration
# Thresholds: Bottom 50% -> Low, Next 35% -> Moderate, Top 15% -> Viral
ENGAGEMENT_PERCENTILES: Tuple[float, float] = (0.50, 0.85)

# Target Classes
TARGET_CLASSES: List[str] = ["Low", "Moderate", "Viral"]
CLASS_MAPPING: Dict[int, str] = {0: "Low", 1: "Moderate", 2: "Viral"}
INVERSE_CLASS_MAPPING: Dict[str, int] = {"Low": 0, "Moderate": 1, "Viral": 2}

# Engagement Score Calculation Mode:
# 'raw': likes + 2 * shares + comments
# 'rate': (likes + 2 * shares + comments) / (followers + 100) * 1000
ENGAGEMENT_SCORE_MODE: str = "rate"

# Train / Validation / Test Splits (70% / 15% / 15%)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Feature Engineering Settings
TFIDF_MAX_FEATURES: int = 800
TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)
TFIDF_MIN_DF: int = 2
TFIDF_MAX_DF: float = 0.95

# Leakage columns that MUST NEVER be used as model input features
LEAKAGE_COLUMNS: List[str] = [
    "likes",
    "shares",
    "retweets",
    "comments",
    "engagement_score",
    "engagement_class",
    "target",
    "post_id",
]

# Tabular feature names expected by the pipeline
TEXT_STAT_COLS: List[str] = [
    "text_length",
    "word_count",
    "hashtag_count",
    "mention_count",
    "url_present",
    "emoji_count",
    "question_count",
    "exclamation_count",
    "uppercase_ratio",
]

SENTIMENT_COLS: List[str] = [
    "vader_compound",
    "vader_pos",
    "vader_neg",
    "vader_neu",
]

TIME_COLS: List[str] = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

ACCOUNT_COLS: List[str] = [
    "log_followers",
]

CATEGORICAL_COLS: List[str] = [
    "media_type",
]

VALID_MEDIA_TYPES: List[str] = ["none", "image", "video"]
