"""
Sentiment analysis subsystem for SocialPulse AI.
Provides a unified dual-engine architecture:
1. Fast lexicon baseline (VADER)
2. Contextual Transformer (Twitter-RoBERTa / DistilBERT) with CPU/GPU auto-detection & fallback
"""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.config import settings
from src.utils.logger import logger


class VaderSentimentEngine:
    """Fast, lexicon-based sentiment analysis using VADER."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment polarity for a single text string."""
        scores = self.analyzer.polarity_scores(text or "")
        compound = scores["compound"]

        # VADER standard threshold convention
        if compound >= 0.05:
            label = "Positive"
            confidence = min(1.0, 0.5 + (compound / 2.0))
        elif compound <= -0.05:
            label = "Negative"
            confidence = min(1.0, 0.5 + (abs(compound) / 2.0))
        else:
            label = "Neutral"
            confidence = scores["neu"]

        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "scores": {
                "pos": round(float(scores["pos"]), 4),
                "neu": round(float(scores["neu"]), 4),
                "neg": round(float(scores["neg"]), 4),
                "compound": round(float(compound), 4),
            },
            "engine": "vader",
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for a batch of texts."""
        return [self.predict(t) for t in texts]


class TransformerSentimentEngine:
    """Deep learning sentiment analysis using HuggingFace Transformers."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.pipeline = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy load pipeline to avoid startup delay."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() and settings.USE_GPU else -1
            logger.info(f"Initializing Transformer sentiment model: {self.model_name} (device={device})...")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                top_k=None,
            )
            self._initialized = True
            logger.info("Transformer pipeline ready.")
        except Exception as e:
            logger.warning(
                f"Could not load HuggingFace model '{self.model_name}' ({e}). Falling back to VADER."
            )
            self.pipeline = None
            self._initialized = True

    def predict(self, text: str) -> Dict[str, Any]:
        """Run Transformer inference on text."""
        self._lazy_init()
        if self.pipeline is None:
            # Fallback to VADER
            return VaderSentimentEngine().predict(text)

        text_input = (text or "")[:512]  # Truncate to maximum sequence length
        raw_outputs = self.pipeline(text_input)[0]

        # Standardize labels (e.g., negative / neutral / positive or LABEL_0/1/2)
        score_map = {}
        for entry in raw_outputs:
            raw_label = entry["label"].lower()
            val = float(entry["score"])
            if "pos" in raw_label or raw_label == "label_2":
                score_map["pos"] = val
            elif "neg" in raw_label or raw_label == "label_0":
                score_map["neg"] = val
            else:
                score_map["neu"] = val

        pos = score_map.get("pos", 0.0)
        neg = score_map.get("neg", 0.0)
        neu = score_map.get("neu", 0.0)

        # Pick winning class
        if pos >= neg and pos >= neu:
            label = "Positive"
            confidence = pos
        elif neg >= pos and neg >= neu:
            label = "Negative"
            confidence = neg
        else:
            label = "Neutral"
            confidence = neu

        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "scores": {
                "pos": round(pos, 4),
                "neu": round(neu, 4),
                "neg": round(neg, 4),
            },
            "engine": "transformer",
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        self._lazy_init()
        if self.pipeline is None:
            return VaderSentimentEngine().predict_batch(texts)

        truncated = [(t or "")[:512] for t in texts]
        results = []
        # Batch inference
        batch_outputs = self.pipeline(truncated, batch_size=32)
        for out in batch_outputs:
            score_map = {}
            for entry in out:
                raw_label = entry["label"].lower()
                val = float(entry["score"])
                if "pos" in raw_label or raw_label == "label_2":
                    score_map["pos"] = val
                elif "neg" in raw_label or raw_label == "label_0":
                    score_map["neg"] = val
                else:
                    score_map["neu"] = val

            pos = score_map.get("pos", 0.0)
            neg = score_map.get("neg", 0.0)
            neu = score_map.get("neu", 0.0)

            if pos >= neg and pos >= neu:
                label = "Positive"
                conf = pos
            elif neg >= pos and neg >= neu:
                label = "Negative"
                conf = neg
            else:
                label = "Neutral"
                conf = neu

            results.append({
                "label": label,
                "confidence": round(float(conf), 4),
                "scores": {"pos": round(pos, 4), "neu": round(neu, 4), "neg": round(neg, 4)},
                "engine": "transformer",
            })
        return results


# Global engine instances
_vader_instance = VaderSentimentEngine()
_transformer_instance = TransformerSentimentEngine()


def predict_sentiment(
    text: str,
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public unified sentiment analysis function.
    Returns: {"label": str, "confidence": float, "scores": dict, "engine": str}
    """
    selected_engine = (engine or settings.SENTIMENT_ENGINE).lower()
    if selected_engine == "transformer":
        return _transformer_instance.predict(text)
    return _vader_instance.predict(text)


def predict_sentiment_batch(
    texts: List[str],
    engine: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Batch sentiment prediction."""
    selected_engine = (engine or settings.SENTIMENT_ENGINE).lower()
    if selected_engine == "transformer":
        return _transformer_instance.predict_batch(texts)
    return _vader_instance.predict_batch(texts)

