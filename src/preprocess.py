"""
Preprocessing module for SocialPulse AI.
Handles text cleaning, noise removal, tokenization, stopword removal,
lemmatization, and target label creation based on configurable engagement thresholds.
"""

import re
import string
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union
from pathlib import Path
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Initialize NLTK resources safely
try:
    NLTK_STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    NLTK_STOPWORDS = set(stopwords.words("english"))

# Retain negation and intensity words that convey sentiment and engagement impact
PRESERVED_WORDS = {"not", "no", "nor", "never", "very", "most", "more", "why", "how", "what"}
STOPWORDS = NLTK_STOPWORDS - PRESERVED_WORDS

try:
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize("testing")
except Exception:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    LEMMATIZER = WordNetLemmatizer()

# Regex patterns
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
MENTION_REGEX = re.compile(r"@\w+")
HASHTAG_REGEX = re.compile(r"#(\w+)")
PUNCT_NOISE_REGEX = re.compile(r"[^\w\s#@!?]")
MULTI_SPACE_REGEX = re.compile(r"\s+")


def clean_text(
    text: Union[str, float],
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    replace_entities: bool = True,
) -> str:
    """
    Cleans raw social media text for NLP modeling.

    Steps:
    1. Handle nulls / non-strings
    2. Convert emojis to descriptive text tokens (e.g. :fire:)
    3. Replace URLs and mentions with standardized tokens or whitespace
    4. Remove noise / unusual punctuation (preserving # for hashtags and !? for feature extraction)
    5. Lowercase and tokenize
    6. Remove general stopwords (preserving negations)
    7. Lemmatize tokens
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    cleaned = text.strip()

    # Demojize emojis to retain semantic context (e.g. 🔥 -> :fire:)
    cleaned = emoji.demojize(cleaned, delimiters=(" ", " "))

    # Replace URLs and mentions
    if replace_entities:
        cleaned = URL_REGEX.sub(" ", cleaned)
        cleaned = MENTION_REGEX.sub(" ", cleaned)
    else:
        cleaned = URL_REGEX.sub(" __url__ ", cleaned)
        cleaned = MENTION_REGEX.sub(" __mention__ ", cleaned)

    # Remove hashtags symbol but keep the tag word (e.g. #MachineLearning -> MachineLearning)
    cleaned = re.sub(r"#(\w+)", r"\1", cleaned)

    # Lowercase
    cleaned = cleaned.lower()

    # Remove unwanted punctuation and numbers/noise
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # Split into word tokens
    tokens = cleaned.split()

    # Stopword filtering
    if remove_stopwords:
        tokens = [token for token in tokens if (token not in STOPWORDS and len(token) > 1)]

    # Lemmatization
    if lemmatize:
        try:
            tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
        except Exception:
            pass

    return " ".join(tokens)


def compute_engagement_score(
    df: pd.DataFrame,
    mode: str = config.ENGAGEMENT_SCORE_MODE,
) -> pd.Series:
    """
    Computes post engagement score from interaction metrics.

    Modes:
    - 'raw': likes + 2 * shares + comments
    - 'rate': (likes + 2 * shares + comments) / (followers + 100) * 1000
    """
    required_cols = ["likes", "shares", "comments"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required metric column: {col}")

    raw_score = df["likes"] + 2.0 * df["shares"] + df["comments"]

    if mode == "rate":
        followers = df["followers"] if "followers" in df.columns else 1000
        # Smoothed rate per 1,000 followers
        score = (raw_score / (followers + 100.0)) * 1000.0
    else:
        score = raw_score

    return score.astype(float)


def assign_engagement_class(
    scores: pd.Series,
    percentiles: Tuple[float, float] = config.ENGAGEMENT_PERCENTILES,
) -> Tuple[pd.Series, pd.Series, Tuple[float, float]]:
    """
    Assigns engagement classes based on configurable percentiles:
    - Bottom 50% -> Low (0)
    - Next 35%   -> Moderate (1)
    - Top 15%    -> Viral (2)

    Returns:
    --------
    class_labels : pd.Series (strings: 'Low', 'Moderate', 'Viral')
    target_ids : pd.Series (integers: 0, 1, 2)
    thresholds : Tuple[float, float] (low_threshold, high_threshold)
    """
    q_low = float(scores.quantile(percentiles[0]))
    q_high = float(scores.quantile(percentiles[1]))

    # Categorize
    def _categorize(score: float) -> str:
        if score < q_low:
            return "Low"
        elif score < q_high:
            return "Moderate"
        else:
            return "Viral"

    class_labels = scores.apply(_categorize)
    target_ids = class_labels.map(config.INVERSE_CLASS_MAPPING)

    return class_labels, target_ids, (q_low, q_high)


def load_and_preprocess_dataset(
    input_path: Path = config.RAW_DATASET_PATH,
    output_path: Optional[Path] = config.PROCESSED_DATASET_PATH,
    engagement_mode: str = config.ENGAGEMENT_SCORE_MODE,
) -> pd.DataFrame:
    """
    Loads raw dataset, performs data validation and deduplication,
    calculates engagement score and target labels, cleans text, and saves
    the processed dataset.
    """
    print(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path)

    initial_len = len(df)
    print(f"Initial row count: {initial_len}")

    # Remove duplicates
    if "post_id" in df.columns:
        df = df.drop_duplicates(subset=["post_id"])
    df = df.drop_duplicates(subset=["text"])
    print(f"Rows after removing duplicates: {len(df)}")

    # Drop nulls in essential columns
    df = df.dropna(subset=["text", "followers"])
    print(f"Rows after dropping missing values: {len(df)}")

    # Ensure followers are non-negative
    df["followers"] = df["followers"].clip(lower=1)

    # Ensure media_type is standardized
    if "media_type" in df.columns:
        df["media_type"] = df["media_type"].fillna("none").str.lower()
        df["media_type"] = df["media_type"].apply(
            lambda x: x if x in config.VALID_MEDIA_TYPES else "none"
        )
    else:
        df["media_type"] = "none"

    # Compute Engagement Score and Target Labels
    df["engagement_score"] = compute_engagement_score(df, mode=engagement_mode)
    labels, targets, thresholds = assign_engagement_class(df["engagement_score"])
    df["engagement_class"] = labels
    df["target"] = targets

    print(f"Target thresholds: Low < {thresholds[0]:.3f} <= Moderate < {thresholds[1]:.3f} <= Viral")
    print("Class distribution:")
    print(df["engagement_class"].value_counts(normalize=True).round(3))

    # Clean text for NLP features (keep original text for regex feature extraction)
    print("Cleaning text for NLP features...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Save processed dataset
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed dataset saved to {output_path}")

    return df


if __name__ == "__main__":
    load_and_preprocess_dataset()
