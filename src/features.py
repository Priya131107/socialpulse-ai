"""
Feature Engineering module for SocialPulse AI.
Extracts text statistics, VADER sentiment scores, temporal attributes,
account metadata, and TF-IDF n-grams without any target leakage.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix, issparse

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.preprocess import clean_text

# Initialize VADER sentiment analyzer
VADER_ANALYZER = SentimentIntensityAnalyzer()

# Regex compiled patterns for fast stats extraction
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")


def extract_text_statistics(text: str) -> Dict[str, Union[int, float]]:
    """
    Computes lexical, syntactic, and structural statistics from the raw text.
    """
    if pd.isna(text) or not isinstance(text, str):
        text = ""

    text_len = len(text)
    words = text.split()
    word_count = len(words)

    hashtag_count = len(HASHTAG_PATTERN.findall(text))
    mention_count = len(MENTION_PATTERN.findall(text))
    url_present = 1 if URL_PATTERN.search(text) else 0
    emoji_cnt = emoji.emoji_count(text)

    question_count = text.count("?")
    exclamation_count = text.count("!")

    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) > 0:
        uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / float(len(alpha_chars))
    else:
        uppercase_ratio = 0.0

    return {
        "text_length": text_len,
        "word_count": word_count,
        "hashtag_count": hashtag_count,
        "mention_count": mention_count,
        "url_present": url_present,
        "emoji_count": emoji_cnt,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
    }


def extract_sentiment_scores(text: str) -> Dict[str, float]:
    """
    Calculates VADER sentiment polarities (compound, pos, neg, neu).
    """
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return {
            "vader_compound": 0.0,
            "vader_pos": 0.0,
            "vader_neg": 0.0,
            "vader_neu": 1.0,
        }

    scores = VADER_ANALYZER.polarity_scores(text)
    return {
        "vader_compound": scores["compound"],
        "vader_pos": scores["pos"],
        "vader_neg": scores["neg"],
        "vader_neu": scores["neu"],
    }


def parse_timestamp_features(timestamp_val: Any) -> Dict[str, int]:
    """
    Extracts hour_of_day, day_of_week, and is_weekend from a timestamp string or object.
    """
    if pd.isna(timestamp_val):
        now = datetime.now()
        return {
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0,
        }

    if isinstance(timestamp_val, str):
        try:
            dt = datetime.fromisoformat(timestamp_val)
        except ValueError:
            try:
                dt = datetime.strptime(timestamp_val, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt = datetime.now()
    elif isinstance(timestamp_val, (pd.Timestamp, datetime)):
        dt = timestamp_val
    else:
        dt = datetime.now()

    hour = dt.hour
    day = dt.weekday()
    is_weekend = 1 if day >= 5 else 0

    return {
        "hour_of_day": hour,
        "day_of_week": day,
        "is_weekend": is_weekend,
    }


class SocialPulseFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    End-to-End Feature Engineering Pipeline for Social Media Engagement.
    Combines:
    1. TF-IDF unigram & bigram representations of cleaned text.
    2. Scaled numerical features (text stats, VADER sentiment, follower count, time attributes).
    3. One-hot encoded media attributes.
    """

    def __init__(
        self,
        max_tfidf_features: int = config.TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = config.TFIDF_NGRAM_RANGE,
    ):
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=self.ngram_range,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.media_encoder = OneHotEncoder(
            categories=[config.VALID_MEDIA_TYPES],
            handle_unknown="ignore",
            sparse_output=False,
        )

        self.feature_names_: List[str] = []
        self.tabular_feature_names_: List[str] = []

    def _extract_tabular_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts all tabular numerical & categorical columns from input DataFrame.
        """
        records = []
        for _, row in df.iterrows():
            text = row.get("text", "")
            stats = extract_text_statistics(text)
            sentiment = extract_sentiment_scores(text)
            time_feat = parse_timestamp_features(row.get("timestamp", None))

            followers = float(row.get("followers", 1000))
            log_followers = float(np.log1p(max(0.0, followers)))

            media = str(row.get("media_type", "none")).lower()
            if media not in config.VALID_MEDIA_TYPES:
                media = "none"

            combined_record = {
                **stats,
                **sentiment,
                **time_feat,
                "log_followers": log_followers,
                "media_type": media,
            }
            records.append(combined_record)

        return pd.DataFrame(records)

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the TF-IDF vectorizer, standard scaler, and media encoder on training data.
        """
        # 1. Clean text for TF-IDF if clean_text not present
        if "clean_text" in X.columns:
            clean_texts = X["clean_text"].fillna("").astype(str).tolist()
        else:
            clean_texts = [clean_text(t) for t in X["text"].fillna("").astype(str)]

        # Adapt min_df / max_df for small document collections
        n_docs = len(clean_texts)
        if n_docs < 10:
            self.tfidf_vectorizer.set_params(min_df=1, max_df=1.0)

        self.tfidf_vectorizer.fit(clean_texts)
        tfidf_names = [f"tfidf_{w}" for w in self.tfidf_vectorizer.get_feature_names_out()]

        # 2. Extract tabular attributes
        tab_df = self._extract_tabular_df(X)

        # Separate numerical and media_type
        num_cols = (
            config.TEXT_STAT_COLS
            + config.SENTIMENT_COLS
            + config.TIME_COLS
            + config.ACCOUNT_COLS
        )
        self.scaler.fit(tab_df[num_cols])

        # Media OneHot
        media_input = tab_df[["media_type"]].values
        self.media_encoder.fit(media_input)
        media_names = [f"media_{cat}" for cat in config.VALID_MEDIA_TYPES]

        self.tabular_feature_names_ = num_cols + media_names
        self.feature_names_ = num_cols + media_names + tfidf_names

        return self

    def transform(self, X: pd.DataFrame) -> csr_matrix:
        """
        Transforms input DataFrame into a combined sparse feature matrix.
        """
        # 1. TF-IDF features
        if "clean_text" in X.columns:
            clean_texts = X["clean_text"].fillna("").astype(str).tolist()
        else:
            clean_texts = [clean_text(t) for t in X["text"].fillna("").astype(str)]

        tfidf_sparse = self.tfidf_vectorizer.transform(clean_texts)

        # 2. Tabular features
        tab_df = self._extract_tabular_df(X)
        num_cols = (
            config.TEXT_STAT_COLS
            + config.SENTIMENT_COLS
            + config.TIME_COLS
            + config.ACCOUNT_COLS
        )
        scaled_num = self.scaler.transform(tab_df[num_cols])

        media_input = tab_df[["media_type"]].values
        encoded_media = self.media_encoder.transform(media_input)

        dense_tab = np.hstack([scaled_num, encoded_media])
        dense_tab_sparse = csr_matrix(dense_tab)

        # Combine tabular + TF-IDF
        combined = hstack([dense_tab_sparse, tfidf_sparse]).tocsr()
        return combined

    def get_feature_names_out(self) -> List[str]:
        """
        Returns complete list of feature names corresponding to matrix columns.
        """
        return self.feature_names_


def build_single_instance_df(text: str, metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Utility to format a single inference request into a DataFrame
    compatible with SocialPulseFeaturePipeline.
    """
    if metadata is None:
        metadata = {}

    followers = metadata.get("followers", 1000)
    timestamp = metadata.get("timestamp", datetime.now().isoformat())
    media_type = metadata.get("media_type", "none")

    row = {
        "text": text,
        "clean_text": clean_text(text),
        "followers": followers,
        "timestamp": timestamp,
        "media_type": media_type,
    }

    return pd.DataFrame([row])


if __name__ == "__main__":
    # Test feature pipeline on processed data
    print("Testing Feature Pipeline...")
    sample_df = pd.read_csv(config.PROCESSED_DATASET_PATH).head(20)
    pipeline = SocialPulseFeaturePipeline()
    pipeline.fit(sample_df)
    features = pipeline.transform(sample_df)
    print(f"Features shape: {features.shape}")
    print(f"Total feature names: {len(pipeline.get_feature_names_out())}")
    print("Top 10 feature names:", pipeline.get_feature_names_out()[:10])
