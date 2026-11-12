"""
Training entrypoint for virality prediction models.
Trains candidate classifiers, saves benchmark metrics, generates evaluation figures,
and creates notebooks/02_viral_model.ipynb.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data.loader import DataLoader
from src.models.viral_model import ViralityModelTrainer, ViralityPredictor
from src.utils.config import settings
from src.utils.logger import logger

def run_training():
    loader = DataLoader()
    logger.info("Loading training and validation splits...")
    train_df, val_df, test_df = loader.load_splits()

    trainer = ViralityModelTrainer(random_seed=settings.RANDOM_SEED)
    bundle = trainer.train_and_benchmark(train_df, val_df, max_tfidf=600)

    # Save benchmark comparison figure
    figures_dir = settings.PROJECT_ROOT / "reports" / "figures" / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = bundle["metrics"]
    models = list(metrics.keys())
    accuracies = [metrics[m]["accuracy"] for m in models]
    macro_f1s = [metrics[m]["macro_f1"] for m in models]
    roc_aucs = [metrics[m]["roc_auc_ovr"] for m in models]

    df_metrics = pd.DataFrame({
        "Model": models,
        "Accuracy": accuracies,
        "Macro F1": macro_f1s,
        "ROC-AUC (OvR)": roc_aucs,
    })

    plt.figure(figsize=(9, 4.5))
    bar_width = 0.25
    x = np.arange(len(models))

    plt.bar(x - bar_width, accuracies, width=bar_width, label="Accuracy", color="#3498db")
    plt.bar(x, macro_f1s, width=bar_width, label="Macro F1", color="#2ecc71")
    plt.bar(x + bar_width, roc_aucs, width=bar_width, label="ROC-AUC (OvR)", color="#e74c3c")

    plt.xlabel("Candidate Model Architecture", fontsize=11, weight="bold")
    plt.ylabel("Score", fontsize=11, weight="bold")
    plt.title("Engagement Model Benchmark Comparison", fontsize=13, weight="bold")
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0.4, 1.0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_benchmark.png", dpi=200)
    plt.close()

    # Save Confusion Matrix of winning model
    best_name = bundle["model_name"]
    best_cm = np.array(metrics[best_name]["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        best_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=settings.ENGAGEMENT_CLASSES,
        yticklabels=settings.ENGAGEMENT_CLASSES,
    )
    plt.title(f"Confusion Matrix — {best_name}", fontsize=12, weight="bold")
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "best_confusion_matrix.png", dpi=200)
    plt.close()

    logger.info(f"Evaluation figures saved to {figures_dir}")

    # Generate 02_viral_model.ipynb
    notebook_path = settings.NOTEBOOKS_DIR / "02_viral_model.ipynb"
    nb_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SocialPulse AI — 02: Virality Prediction Model\n",
                    "\n",
                    "This notebook benchmarks **Logistic Regression**, **Random Forest**, **XGBoost**, and **LightGBM**\n",
                    "on multi-class engagement tier prediction (Low / Moderate / Viral)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.models.viral_model import ViralityPredictor\n",
                    "predictor = ViralityPredictor()\n",
                    "sample_post = {\n",
                    "    'text': 'Excited to announce our new multimodal AI agents! Open-source and free for all developers: https://pulse.io #AI #OpenSource 🚀',\n",
                    "    'platform': 'Twitter',\n",
                    "    'follower_count': 25000,\n",
                    "    'has_media': 1\n",
                    "}\n",
                    "res = predictor.predict(sample_post)\n",
                    "print(f'Predicted Tier: {res[\"prediction\"]} (Confidence: {res[\"confidence\"]})')\n",
                    "print('Probabilities:', res['probabilities'])\n",
                    "print('Optimization Tips:', res['suggestions'])"
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

    logger.info(f"Saved notebook {notebook_path}")

    # Test single inference
    predictor = ViralityPredictor()
    sample_result = predictor.predict({
        "text": "Revolutionary AI architecture breakthrough announced today! #Tech #AI 🚀",
        "platform": "Twitter",
        "follower_count": 50000,
        "has_media": 1,
    })
    logger.info(f"Sample prediction verification: {sample_result['prediction']} ({sample_result['probabilities']})")

if __name__ == "__main__":
    run_training()

