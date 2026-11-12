"""
Sentiment analysis API route.
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    SentimentItemResponse,
    SentimentRequest,
    SentimentResponse,
    SentimentScores,
)
from src.models.sentiment_model import predict_sentiment, predict_sentiment_batch
from src.utils.logger import logger

router = APIRouter(prefix="/analyze", tags=["Sentiment Analysis"])


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    summary="Analyze Social Post Polarity",
)
def analyze_sentiment(payload: SentimentRequest):
    """
    Accepts a single `text` string or batch list of `texts`, returning:
    - **Polarity Label**: `Positive`, `Neutral`, or `Negative`
    - **Confidence**: Model certainty
    - **Polarity Scores**: Raw positive, neutral, negative probabilities
    - **Engine**: Engine utilized (`vader` or `transformer`)
    """
    engine = payload.engine or "vader"
    items_to_process = []

    if payload.text:
        items_to_process.append(payload.text)
    if payload.texts:
        items_to_process.extend(payload.texts)

    if not items_to_process:
        raise HTTPException(status_code=400, detail="Must provide either 'text' or 'texts' in request.")

    try:
        raw_predictions = predict_sentiment_batch(items_to_process, engine=engine)
        formatted_results = []
        for p in raw_predictions:
            scores_dict = p["scores"]
            formatted_results.append(
                SentimentItemResponse(
                    label=p["label"],
                    confidence=p["confidence"],
                    scores=SentimentScores(
                        pos=scores_dict["pos"],
                        neu=scores_dict["neu"],
                        neg=scores_dict["neg"],
                        compound=scores_dict.get("compound"),
                    ),
                    engine=p["engine"],
                )
            )
        return SentimentResponse(results=formatted_results)
    except Exception as e:
        logger.error(f"Sentiment analysis error in /analyze/sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
