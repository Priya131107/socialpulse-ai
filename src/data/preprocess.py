"""
Text cleaning, emoji normalization, and engagement labeling pipeline.
Ensures strict prevention of data leakage between target metrics and input features.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import emoji
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import settings
from src.utils.logger import logger


class TextPreprocessor:
    """Preprocesses raw social media text and prepares target labels."""

    def __init__(
        self,
        percentiles: Tuple[float, float] = (0.50, 0.85),
        random_seed: int = 42,
    ):
        self.percentiles = percentiles
        self.random_seed = random_seed
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.whitespace_pattern = re.compile(r"\s+")
        self.hashtag_pattern = re.compile(r"#(\w+)")

    def clean_text(self, text: str, demojize: bool = True) -> str:
        """
        Clean text:
        - Handle emojis by converting to textual description (e.g. :fire:)
        - Remove URLs
        - Standardize whitespace and lowercase
        - Keep hashtags as tokens
        """
        if not isinstance(text, str):
            return ""

        # Demojize emojis
        if demojize:
            text = emoji.demojize(text, delimiters=(" ", " "))

        # Remove URLs
        text = self.url_pattern.sub(" ", text)

        # Normalize redundant punctuation
        text = re.sub(r"([!?.]){2,}", r"\1", text)

        # Remove excess whitespace
        text = self.whitespace_pattern.sub(" ", text).strip().lower()

        return text

    def calculate_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute normalized engagement rate:
        (likes + 2.0*shares + 1.5*comments) / (follower_count + 100) * 1000
        """
        likes = df["likes"].fillna(0)
        shares = df["shares"].fillna(0)
        comments = df["comments"].fillna(0)
        followers = df["follower_count"].fillna(0)

        weighted_interactions = likes + (2.0 * shares) + (1.5 * comments)
        normalized_score = (weighted_interactions / (followers + 100)) * 1000.0
        return normalized_score

    def assign_engagement_labels(
        self,
        df: pd.DataFrame,
        cutoffs: Optional[Tuple[float, float]] = None,
    ) -> Tuple[pd.DataFrame, Tuple[float, float]]:
        """
        Discretize continuous engagement scores into Low, Moderate, Viral tiers.
        """
        df = df.copy()
        if "engagement_score" not in df.columns:
            df["engagement_score"] = self.calculate_engagement_score(df)

        if cutoffs is None:
            p_low, p_viral = self.percentiles
            cutoff_low = float(df["engagement_score"].quantile(p_low))
            cutoff_viral = float(df["engagement_score"].quantile(p_viral))
            cutoffs = (cutoff_low, cutoff_viral)
        else:
            cutoff_low, cutoff_viral = cutoffs

        def _label_tier(score: float) -> str:
            if score < cutoff_low:
                return "Low"
            elif score < cutoff_viral:
                return "Moderate"
            else:
                return "Viral"

        df["engagement_label"] = df["engagement_score"].apply(_label_tier)
        logger.info(
            f"Assigned engagement labels using cutoffs: Low < {cutoff_low:.3f} <= Moderate < {cutoff_viral:.3f} <= Viral"
        )
        return df, cutoffs

    def preprocess_dataset(
        self,
        df: pd.DataFrame,
        cutoffs: Optional[Tuple[float, float]] = None,
    ) -> Tuple[pd.DataFrame, Tuple[float, float]]:
        """Run full preprocessing pipeline on DataFrame."""
        df = df.copy()

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Clean text column
        df["cleaned_text"] = df["text"].apply(self.clean_text)

        # Assign engagement labels
        df, calculated_cutoffs = self.assign_engagement_labels(df, cutoffs)

        return df, calculated_cutoffs

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train / validation / test split based on engagement_label.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5

        # First split into train and temp (val + test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=df["engagement_label"],
            random_state=self.random_seed,
        )

        # Split temp into val and test
        val_relative_ratio = val_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_relative_ratio),
            stratify=temp_df["engagement_label"],
            random_state=self.random_seed,
        )

        logger.info(
            f"Stratified split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )
        return train_df, val_df, test_df


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute preprocessing and generate processed splits from raw data."""
    raw_path = settings.RAW_DATA_PATH
    if not raw_path.exists():
        logger.warning(f"Raw data not found at {raw_path}. Generating now...")
        from src.data.synthetic_generator import generate_social_dataset
        generate_social_dataset(num_records=settings.NUM_SYNTHETIC_RECORDS)

    df_raw = pd.read_csv(raw_path)
    preprocessor = TextPreprocessor(
        percentiles=tuple(settings.PERCENTILES),
        random_seed=settings.RANDOM_SEED,
    )

    df_processed, cutoffs = preprocessor.preprocess_dataset(df_raw)
    train_df, val_df, test_df = preprocessor.split_data(df_processed)

    # Save processed splits
    train_df.to_csv(settings.TRAIN_DATA_PATH, index=False)
    val_df.to_csv(settings.VAL_DATA_PATH, index=False)
    test_df.to_csv(settings.TEST_DATA_PATH, index=False)

    # Save a small deterministic sample for fast testing & API sanity checks
    sample_df = df_processed.sample(n=min(500, len(df_processed)), random_state=settings.RANDOM_SEED)
    sample_df.to_csv(settings.SAMPLE_DATA_PATH, index=False)

    logger.info(f"Saved processed datasets to {settings.PROCESSED_DATA_DIR}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()

