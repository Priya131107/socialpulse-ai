"""
SocialPulse AI — Simple Preprocessing Pipeline.
Text cleaning + train/val/test split.
"""

import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.config import settings
from src.utils.logger import logger


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs, remove noise."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)              # remove mentions
    text = re.sub(r"[^a-z0-9\s#]", "", text)     # keep alphanumeric + hashtags
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_pipeline() -> tuple:
    """Load raw data, clean text, split into train/val/test, save CSVs."""
    raw_path = settings.RAW_DATA_PATH
    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}. Run synthetic_generator first.")
        return None, None, None

    df = pd.read_csv(raw_path)
    df["clean_text"] = df["text"].apply(clean_text)

    # Drop rows with missing labels
    df = df.dropna(subset=["engagement_label", "sentiment_label"])

    # 70/15/15 split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42,
                                         stratify=df["engagement_label"])
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42,
                                         stratify=temp_df["engagement_label"])

    # Save splits
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(settings.TRAIN_DATA_PATH, index=False)
    val_df.to_csv(settings.VAL_DATA_PATH,     index=False)
    test_df.to_csv(settings.TEST_DATA_PATH,   index=False)

    # Sample
    settings.SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.sample(n=min(200, len(df)), random_state=42).to_csv(settings.SAMPLE_DATA_PATH, index=False)

    logger.info(f"Splits saved — train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()
