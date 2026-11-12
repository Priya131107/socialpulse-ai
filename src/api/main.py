"""
Main FastAPI Application Entrypoint for SocialPulse AI.
Assembles API routes, CORS policies, error handlers, and health monitoring.
"""

from datetime import datetime
import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import alerts, crisis, predict, sentiment
from src.api.schemas import HealthResponse
from src.utils.config import settings
from src.utils.logger import logger

START_TIME = time.time()

app = FastAPI(
    title="SocialPulse AI — Machine Learning API",
    description=(
        "Production-style REST API for social media intelligence:\n"
        "- **Post Virality Prediction** (Low, Moderate, Viral)\n"
        "- **Sentiment Analysis** (VADER baseline + HuggingFace Transformer)\n"
        "- **Brand Crisis Early Warning** (Z-Score + Isolation Forest anomalies)\n"
        "- **Multi-channel Alerting** (Cooldown throttling + SQLite audit)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit dashboard and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(predict.router)
app.include_router(sentiment.router)
app.include_router(crisis.router)
app.include_router(alerts.router)


@app.on_event("startup")
async def startup_event():
    logger.info("SocialPulse AI REST API initialized and ready to accept traffic.")


@app.get("/", tags=["General"])
def root():
    return {
        "app": "SocialPulse AI",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health",
        "endpoints": [
            "POST /predict/engagement",
            "POST /analyze/sentiment",
            "GET  /crisis/status?brand={brand}",
            "GET  /crisis/history?brand={brand}",
            "POST /alerts/test",
            "GET  /alerts/recent",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """System health check and dependency readiness."""
    viral_ready = settings.VIRAL_MODEL_PATH.exists()
    crisis_ready = settings.CRISIS_MODEL_PATH.exists()

    db_ready = False
    try:
        import sqlite3
        with sqlite3.connect(settings.ALERTS_DB_PATH) as conn:
            conn.execute("SELECT 1")
            db_ready = True
    except Exception:
        db_ready = False

    return HealthResponse(
        status="ok" if (viral_ready and crisis_ready and db_ready) else "degraded",
        app_name="SocialPulse AI",
        version="1.0.0",
        models_loaded={
            "viral_prediction_model": viral_ready,
            "crisis_anomaly_detector": crisis_ready,
        },
        database_connected=db_ready,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception at {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred", "error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
