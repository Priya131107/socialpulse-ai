"""
Unit tests for sentiment analysis subsystem (VADER baseline and unified interface).
"""

import pytest
from src.models.sentiment_model import (
    VaderSentimentEngine,
    predict_sentiment,
    predict_sentiment_batch,
)


def test_vader_positive_sentiment():
    engine = VaderSentimentEngine()
    text = "We are absolutely thrilled with this phenomenal product! Incredible service! 🚀🔥"
    res = engine.predict(text)

    assert res["label"] == "Positive"
    assert res["confidence"] > 0.60
    assert res["scores"]["pos"] > res["scores"]["neg"]
    assert res["engine"] == "vader"


def test_vader_negative_sentiment():
    engine = VaderSentimentEngine()
    text = "Absolute disaster! Broken app, lost my data, customer desk refused refund! 😡"
    res = engine.predict(text)

    assert res["label"] == "Negative"
    assert res["confidence"] > 0.60
    assert res["scores"]["neg"] > res["scores"]["pos"]


def test_vader_neutral_sentiment():
    engine = VaderSentimentEngine()
    text = "The quarterly investor meeting will take place at 10 AM on Monday via teleconference."
    res = engine.predict(text)

    assert res["label"] in ["Neutral", "Positive"]
    assert "scores" in res


def test_unified_predict_sentiment():
    res = predict_sentiment("A great experience today!")
    assert isinstance(res, dict)
    assert "label" in res
    assert "confidence" in res
    assert "scores" in res
    assert res["label"] in ["Positive", "Neutral", "Negative"]


def test_batch_sentiment_prediction():
    texts = [
        "Love this app!",
        "Regular routine update.",
        "Worst customer support ever.",
    ]
    results = predict_sentiment_batch(texts)

    assert len(results) == 3
    assert results[0]["label"] == "Positive"
    assert results[2]["label"] == "Negative"

