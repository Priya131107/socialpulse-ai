"""
Engagement & Virality prediction API route.
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas import FeatureImpact, PostEngagementRequest, PostEngagementResponse
from src.models.viral_model import ViralityPredictor
from src.utils.logger import logger

router = APIRouter(prefix="/predict", tags=["Engagement Prediction"])

# Lazy-loaded predictor singleton
_predictor_instance = None

def get_predictor() -> ViralityPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        try:
            _predictor_instance = ViralityPredictor()
        except Exception as e:
            logger.error(f"Failed to load virality predictor: {e}")
            raise HTTPException(
                status_code=503,
                detail="Virality prediction model not initialized or trained.",
            )
    return _predictor_instance


@router.post(
    "/engagement",
    response_model=PostEngagementResponse,
    summary="Predict Social Post Virality Tier",
)
def predict_engagement(payload: PostEngagementRequest):
    """
    Accepts text content and pre-publish metadata, returning:
    - **Engagement Tier**: `Low`, `Moderate`, or `Viral`
    - **Probabilities**: Class probability distribution
    - **Top Contributing Features**: Features pushing the prediction higher/lower
    - **Suggestions**: Actionable advice to boost post performance
    """
    predictor = get_predictor()
    try:
        raw_result = predictor.predict(payload.model_dump())
        feature_impacts = [
            FeatureImpact(
                feature=item["feature"],
                impact=item["impact"],
                direction=item["direction"],
            )
            for item in raw_result.get("top_contributing_features", [])
        ]

        return PostEngagementResponse(
            prediction=raw_result["prediction"],
            probabilities=raw_result["probabilities"],
            confidence=raw_result["confidence"],
            top_contributing_features=feature_impacts,
            suggestions=raw_result["suggestions"],
            model_used=raw_result["model_used"],
        )
    except Exception as e:
        logger.error(f"Inference error in /predict/engagement: {e}")
        raise HTTPException(status_code=500, detail=str(e))
