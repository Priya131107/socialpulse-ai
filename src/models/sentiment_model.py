"""
SocialPulse AI — Sentiment Model (VADER only, no transformers).
Fast, simple, no GPU required.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for a single text string.
    Returns label (Positive/Neutral/Negative), compound score, and all scores.
    """
    scores = _analyzer.polarity_scores(str(text))
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label":      label,
        "confidence": round(abs(compound), 4),
        "scores":     {k: round(v, 4) for k, v in scores.items()},
    }


def predict_sentiment_batch(texts: list) -> list:
    """Predict sentiment for a list of texts."""
    return [predict_sentiment(t) for t in texts]
