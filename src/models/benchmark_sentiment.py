"""
Sentiment Analysis Benchmarking Engine for SocialPulse AI.
Compares VADER (baseline) against Transformer on accuracy, macro F1, and latency.
Generates benchmark reports, charts, and notebooks/03_sentiment.ipynb.
"""

import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.loader import DataLoader
from src.models.sentiment_model import (
    VaderSentimentEngine,
    TransformerSentimentEngine,
    predict_sentiment,
)
from src.utils.config import settings
from src.utils.logger import logger

def run_sentiment_benchmark(sample_size: int = 300):
    loader = DataLoader()
    _, val_df, _ = loader.load_splits()
    
    eval_df = val_df.sample(n=min(sample_size, len(val_df)), random_state=settings.RANDOM_SEED).copy()
    texts = eval_df["text"].tolist()
    y_true = eval_df["sentiment_label"].tolist()

    logger.info(f"Benchmarking Sentiment Engines on {len(texts)} validation samples...")

    # 1. Benchmark VADER
    vader = VaderSentimentEngine()
    t0 = time.perf_counter()
    vader_preds = [vader.predict(t) for t in texts]
    vader_time = time.perf_counter() - t0
    y_pred_vader = [p["label"] for p in vader_preds]

    vader_acc = accuracy_score(y_true, y_pred_vader)
    vader_f1 = f1_score(y_true, y_pred_vader, average="macro")
    vader_latency = (vader_time / len(texts)) * 1000.0  # ms per sample

    logger.info(
        f"VADER -> Accuracy: {vader_acc:.4f} | Macro F1: {vader_f1:.4f} | Latency: {vader_latency:.2f} ms/sample"
    )

    # 2. Benchmark Transformer (or fast fallback)
    transformer = TransformerSentimentEngine()
    t0 = time.perf_counter()
    transformer_preds = transformer.predict_batch(texts)
    trans_time = time.perf_counter() - t0
    y_pred_trans = [p["label"] for p in transformer_preds]

    trans_acc = accuracy_score(y_true, y_pred_trans)
    trans_f1 = f1_score(y_true, y_pred_trans, average="macro")
    trans_latency = (trans_time / len(texts)) * 1000.0  # ms per sample

    logger.info(
        f"Transformer -> Accuracy: {trans_acc:.4f} | Macro F1: {trans_f1:.4f} | Latency: {trans_latency:.2f} ms/sample"
    )

    results = {
        "VADER": {
            "accuracy": round(float(vader_acc), 4),
            "macro_f1": round(float(vader_f1), 4),
            "latency_ms": round(float(vader_latency), 2),
            "throughput_samples_per_sec": round(float(len(texts) / max(0.001, vader_time)), 1),
        },
        "Transformer": {
            "accuracy": round(float(trans_acc), 4),
            "macro_f1": round(float(trans_f1), 4),
            "latency_ms": round(float(trans_latency), 2),
            "throughput_samples_per_sec": round(float(len(texts) / max(0.001, trans_time)), 1),
        },
    }

    # Save benchmark json
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bench_file = settings.MODELS_DIR / "sentiment_benchmark.json"
    with open(bench_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot comparison chart
    figures_dir = settings.PROJECT_ROOT / "reports" / "figures" / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    models = ["VADER (Baseline)", "Transformer (RoBERTa)"]
    f1_scores = [vader_f1, trans_f1]
    latencies = [vader_latency, trans_latency]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, f1_scores, width, label="Macro F1", color="#2ecc71")
    ax1.set_ylabel("Macro F1 Score", color="#27ae60", fontsize=11, weight="bold")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, weight="bold")

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, latencies, width, label="Latency (ms)", color="#e67e22")
    ax2.set_ylabel("Inference Latency (ms/sample)", color="#d35400", fontsize=11, weight="bold")

    plt.title("Sentiment Model Benchmark: Accuracy vs Latency Tradeoff", fontsize=12, weight="bold")
    fig.tight_layout()
    plt.savefig(figures_dir / "sentiment_benchmark.png", dpi=200)
    plt.close()

    # Generate 03_sentiment.ipynb
    notebook_path = settings.NOTEBOOKS_DIR / "03_sentiment.ipynb"
    nb_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SocialPulse AI — 03: Sentiment Analysis Subsystem\n",
                    "\n",
                    "Benchmarking **VADER Lexicon** vs **HuggingFace Twitter-RoBERTa**\n",
                    "for social media polarity classification."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.models.sentiment_model import predict_sentiment\n",
                    "\n",
                    "texts = [\n",
                    "    'Loving the new ultra-fast cloud SDK! Total game changer! 🚀',\n",
                    "    'Scheduled server maintenance announced for Tuesday night.',\n",
                    "    'Flight cancelled 6 times, customer service hangs up! Boycott this airline! 😡'\n",
                    "]\n",
                    "\n",
                    "for t in texts:\n",
                    "    res = predict_sentiment(t)\n",
                    "    print(f'Text: {t[:40]}... -> Label: {res[\"label\"]} (Confidence: {res[\"confidence\"]})')"
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

    logger.info(f"Sentiment benchmark completed and saved to {bench_file}")
    return results

if __name__ == "__main__":
    run_sentiment_benchmark()

