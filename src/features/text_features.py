"""
Text feature extractor for SocialPulse AI.
Extracts lexical statistics, structural flags, VADER sentiment polarities,
and TF-IDF unigram/bigram token representations.
"""

import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts lexical metrics, sentiment signals, and TF-IDF representations."""

    def __init__(
        self,
        max_tfidf_features: int = 600,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self.vader = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r"(?u)\b\w+\b|#\w+",
        )
        self.feature_names_: List[str] = []

    def _extract_lexical_row(self, raw_text: str, cleaned_text: str) -> Dict[str, float]:
        """Extract lexical stats and VADER scores for a single text instance."""
        raw_text = raw_text or ""
        cleaned_text = cleaned_text or ""

        char_len = len(raw_text)
        words = raw_text.split()
        word_count = len(words)

        # Uppercase ratio
        upper_count = sum(1 for c in raw_text if c.isupper())
        upper_ratio = (upper_count / max(1, char_len))

        # Punctuation flags & counts
        question_count = raw_text.count("?")
        exclamation_count = raw_text.count("!")
        has_question = 1.0 if question_count > 0 else 0.0
        has_exclamation = 1.0 if exclamation_count > 0 else 0.0

        # Emojis count
        emoji_count = emoji.emoji_count(raw_text)

        # Hashtags and Mentions
        hashtag_count = len(re.findall(r"#\w+", raw_text))
        mention_count = raw_text.count("@")
        has_url = 1.0 if bool(re.search(r"https?://|www\.", raw_text)) else 0.0

        # VADER Sentiment analysis
        vader_scores = self.vader.polarity_scores(raw_text)

        return {
            "text_length": float(char_len),
            "word_count": float(word_count),
            "uppercase_ratio": float(upper_ratio),
            "question_count": float(question_count),
            "exclamation_count": float(exclamation_count),
            "has_question": has_question,
            "has_exclamation": has_exclamation,
            "emoji_count": float(emoji_count),
            "hashtag_count": float(hashtag_count),
            "mention_count": float(mention_count),
            "has_url": has_url,
            "vader_compound": float(vader_scores["compound"]),
            "vader_pos": float(vader_scores["pos"]),
            "vader_neg": float(vader_scores["neg"]),
            "vader_neu": float(vader_scores["neu"]),
        }

    def fit(self, X: pd.DataFrame, y=None):
        """Fit TF-IDF vectorizer on cleaned_text."""
        texts = X["cleaned_text"].fillna("").astype(str)
        n_samples = len(texts)
        effective_min_df = 1 if n_samples < 5 else self.min_df
        effective_max_df = 1.0 if n_samples < 10 else self.max_df
        self.tfidf = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=self.ngram_range,
            min_df=effective_min_df,
            max_df=effective_max_df,
            token_pattern=r"(?u)\b\w+\b|#\w+",
        )
        self.tfidf.fit(texts)
        
        # Build list of feature names
        lexical_sample = self._extract_lexical_row("", "")
        self.lexical_feature_names_ = list(lexical_sample.keys())
        self.tfidf_feature_names_ = [f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()]
        self.feature_names_ = self.lexical_feature_names_ + self.tfidf_feature_names_
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract lexical features and TF-IDF matrix, returning stacked numpy array."""
        raw_texts = X["text"].fillna("").astype(str).tolist()
        cleaned_texts = X["cleaned_text"].fillna("").astype(str).tolist() if "cleaned_text" in X.columns else raw_texts

        lexical_rows = [
            self._extract_lexical_row(r, c)
            for r, c in zip(raw_texts, cleaned_texts)
        ]
        lexical_df = pd.DataFrame(lexical_rows)
        lexical_matrix = lexical_df[self.lexical_feature_names_].values

        tfidf_matrix = self.tfidf.transform(cleaned_texts).toarray()
        
        return np.hstack([lexical_matrix, tfidf_matrix])

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_
