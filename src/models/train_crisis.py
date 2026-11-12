"""
Crisis detector model training and historical anomaly validation.
Fits the multi-window Isolation Forest and generates notebooks/04_crisis.ipynb.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.models.crisis_detector import CrisisDetector, save_crisis_detector
from src.utils.config import get_config, settings
from src.utils.logger import logger

def run_crisis_training():
    loader = DataLoader()
    raw_df = loader.load_raw()

    config = get_config()
    detector = CrisisDetector(
        contamination=float(config.get("crisis", {}).get("isolation_forest_contamination", 0.05)),
        z_threshold=float(config.get("crisis", {}).get("z_score_threshold", 2.5)),
        rule_neg_ratio=float(config.get("crisis", {}).get("threshold_negative_ratio", 0.40)),
        rule_velocity_mult=float(config.get("crisis", {}).get("threshold_volume_growth_multiplier", 2.0)),
        random_seed=settings.RANDOM_SEED,
    )

    logger.info("Fitting Crisis Detector...")
    detector.fit(raw_df)
    save_crisis_detector(detector)

    # Historical timeline validation for SwiftAir (which has the Day 10-12 crisis)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    swift_df = raw_df[raw_df["brand"] == "SwiftAir"].sort_values("timestamp")

    timestamps = []
    risk_scores = []
    levels = []

    # Sample daily points
    min_t = swift_df["timestamp"].min() + timedelta(hours=24)
    max_t = swift_df["timestamp"].max()
    curr = min_t
    while curr <= max_t:
        sub = swift_df[swift_df["timestamp"] <= curr]
        res = detector.analyze_brand("SwiftAir", sub, curr)
        timestamps.append(curr)
        risk_scores.append(res["crisis_risk_score"])
        levels.append(res["risk_level"])
        curr += timedelta(hours=12)

    # Plot crisis trajectory
    figures_dir = settings.PROJECT_ROOT / "reports" / "figures" / "crisis"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 5))
    plt.plot(timestamps, risk_scores, label="Crisis Risk Score (0-100)", color="#e74c3c", linewidth=2.5)
    plt.axhline(70, color="red", linestyle="--", alpha=0.7, label="Critical Alarm Threshold (70)")
    plt.axhline(40, color="orange", linestyle=":", alpha=0.7, label="Warning Threshold (40)")
    plt.fill_between(timestamps, 70, 100, color="red", alpha=0.08)
    plt.fill_between(timestamps, 40, 70, color="orange", alpha=0.08)

    plt.title("SwiftAir — Longitudinal Crisis Risk Trajectory & Anomaly Spike Detection", fontsize=13, weight="bold")
    plt.xlabel("Timeline", fontsize=11)
    plt.ylabel("Risk Score (0 - 100)", fontsize=11)
    plt.ylim(0, 100)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "crisis_trajectory_swiftair.png", dpi=200)
    plt.close()

    logger.info(f"Crisis evaluation plot saved to {figures_dir}")

    # Create 04_crisis.ipynb
    notebook_path = settings.NOTEBOOKS_DIR / "04_crisis.ipynb"
    nb_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SocialPulse AI — 04: Brand Crisis Early Warning System\n",
                    "\n",
                    "Demonstrating multi-window rolling aggregation, statistical Z-score tripwires,\n",
                    "and Isolation Forest anomaly detection."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.data.loader import DataLoader\n",
                    "from src.models.crisis_detector import load_crisis_detector\n",
                    "\n",
                    "loader = DataLoader()\n",
                    "raw_df = loader.load_raw()\n",
                    "detector = load_crisis_detector()\n",
                    "\n",
                    "# Analyze SwiftAir status\n",
                    "swift_df = raw_df[raw_df['brand'] == 'SwiftAir']\n",
                    "status = detector.analyze_brand('SwiftAir', swift_df)\n",
                    "print(f'Brand: {status[\"brand\"]}')\n",
                    "print(f'Risk Score: {status[\"crisis_risk_score\"]} ({status[\"risk_level\"]})')\n",
                    "print(f'Signals: {status[\"signals\"]}')\n",
                    "print(f'Top Diagnostic Keywords: {status[\"diagnostic_keywords\"]}')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)

    logger.info(f"Saved crisis notebook {notebook_path}")

if __name__ == "__main__":
    run_crisis_training()
