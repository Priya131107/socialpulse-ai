"""
Central configuration loader for SocialPulse AI.
Loads settings from config.yaml and environment variables (.env).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

# Base Directory Resolution
CURRENT_FILE = Path(__file__).resolve()
# src/utils/config.py -> parent: src/utils -> parent: src -> parent: socialpulse_ai
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"

def _load_yaml() -> Dict[str, Any]:
    if CONFIG_YAML_PATH.exists():
        with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

_CONFIG: Dict[str, Any] = _load_yaml()

def get_config() -> Dict[str, Any]:
    """Retrieve raw config dictionary."""
    return _CONFIG

class Settings:
    """Config settings helper object."""
    
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
    MODELS_DIR: Path = PROJECT_ROOT / "models_saved"
    NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
    
    # Ensure directory existence
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        
    RAW_DATA_PATH: Path = RAW_DATA_DIR / "posts_dataset.csv"
    TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
    VAL_DATA_PATH: Path = PROCESSED_DATA_DIR / "val.csv"
    TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"
    SAMPLE_DATA_PATH: Path = SAMPLE_DATA_DIR / "sample_posts.csv"
    
    VIRAL_MODEL_PATH: Path = MODELS_DIR / "best_viral_model.joblib"
    CRISIS_MODEL_PATH: Path = MODELS_DIR / "crisis_detector.joblib"
    
    RANDOM_SEED: int = int(_CONFIG.get("data", {}).get("random_seed", 42))
    NUM_SYNTHETIC_RECORDS: int = int(_CONFIG.get("data", {}).get("synthetic_num_records", 20000))
    
    ENGAGEMENT_CLASSES: List[str] = _CONFIG.get("engagement", {}).get("classes", ["Low", "Moderate", "Viral"])
    CLASS_MAPPING: Dict[int, str] = {0: "Low", 1: "Moderate", 2: "Viral"}
    INVERSE_CLASS_MAPPING: Dict[str, int] = {"Low": 0, "Moderate": 1, "Viral": 2}
    
    PERCENTILES: List[float] = [
        float(_CONFIG.get("engagement", {}).get("percentile_low_cutoff", 0.50)),
        float(_CONFIG.get("engagement", {}).get("percentile_viral_cutoff", 0.85))
    ]
    
    # Environmental overrides
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")
    SENTIMENT_ENGINE: str = os.getenv("SENTIMENT_ENGINE", "vader").lower()
    ALERTS_DB_PATH: str = str(PROJECT_ROOT / _CONFIG.get("alerts", {}).get("db_path", "alerts.db"))

settings = Settings()

