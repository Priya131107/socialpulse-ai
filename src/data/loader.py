"""
Data loader utility for SocialPulse AI.
Handles loading raw, split, and sample datasets with schema validation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.config import settings
from src.utils.logger import logger

REQUIRED_RAW_COLUMNS = [
    "post_id",
    "text",
    "platform",
    "timestamp",
    "hashtags_count",
    "mentions_count",
    "has_media",
    "follower_count",
    "likes",
    "shares",
    "comments",
    "brand",
    "sentiment_label",
]

REQUIRED_PROCESSED_COLUMNS = REQUIRED_RAW_COLUMNS + [
    "cleaned_text",
    "engagement_score",
    "engagement_label",
]


class DataLoader:
    """Manages loading and integrity validation of datasets."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR

    @staticmethod
    def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Verify DataFrame contains all required columns."""
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        return True

    def load_raw(self) -> pd.DataFrame:
        """Load raw dataset."""
        path = settings.RAW_DATA_PATH
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found at: {path}")
        df = pd.read_csv(path)
        self.validate_schema(df, REQUIRED_RAW_COLUMNS)
        return df

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, val, test splits."""
        for p in [settings.TRAIN_DATA_PATH, settings.VAL_DATA_PATH, settings.TEST_DATA_PATH]:
            if not p.exists():
                raise FileNotFoundError(f"Processed split not found at: {p}")

        train_df = pd.read_csv(settings.TRAIN_DATA_PATH)
        val_df = pd.read_csv(settings.VAL_DATA_PATH)
        test_df = pd.read_csv(settings.TEST_DATA_PATH)

        self.validate_schema(train_df, REQUIRED_PROCESSED_COLUMNS)
        self.validate_schema(val_df, REQUIRED_PROCESSED_COLUMNS)
        self.validate_schema(test_df, REQUIRED_PROCESSED_COLUMNS)

        return train_df, val_df, test_df

    def load_sample(self, n: int = 500) -> pd.DataFrame:
        """Load deterministic sample dataset for testing."""
        if settings.SAMPLE_DATA_PATH.exists():
            df = pd.read_csv(settings.SAMPLE_DATA_PATH)
            return df.head(n)
        train_df, _, _ = self.load_splits()
        return train_df.head(n)

