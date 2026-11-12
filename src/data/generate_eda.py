"""
EDA Notebook Generator & Visual Insight Producer for SocialPulse AI.
Executes Exploratory Data Analysis on the 20,000 posts dataset,
generates publication-ready figures, and formats notebooks/01_eda.ipynb.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils.config import settings
from src.utils.logger import logger

def run_eda():
    figures_dir = settings.PROJECT_ROOT / "reports" / "figures" / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(settings.RAW_DATA_PATH)
    train_df = pd.read_csv(settings.TRAIN_DATA_PATH)
    
    logger.info("Generating EDA figures...")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Class Balance Bar Chart
    plt.figure(figsize=(7, 4.5))
    ax = sns.countplot(
        data=train_df, 
        x="engagement_label", 
        order=["Low", "Moderate", "Viral"],
        palette=["#3498db", "#2ecc71", "#e74c3c"]
    )
    plt.title("Engagement Tier Distribution (Train Set - 14,000 Posts)", fontsize=13, weight="bold")
    plt.xlabel("Engagement Tier", fontsize=11)
    plt.ylabel("Number of Posts", fontsize=11)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:,.0f} ({height/len(train_df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha="center", va="center", color="white", weight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "class_balance.png", dpi=200)
    plt.close()
    
    # 2. Hour of Day vs Engagement Rate
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    train_df["hour"] = train_df["timestamp"].dt.hour
    hourly_eng = train_df.groupby("hour")["engagement_score"].mean().reset_index()
    
    plt.figure(figsize=(10, 4.5))
    sns.lineplot(data=hourly_eng, x="hour", y="engagement_score", marker="o", color="#8e44ad", linewidth=2.5)
    plt.axvspan(12, 14, color="#f39c12", alpha=0.2, label="Lunch Peak (12:00-14:00)")
    plt.axvspan(18, 22, color="#e67e22", alpha=0.2, label="Evening Prime Time (18:00-22:00)")
    plt.title("Hourly Engagement Velocity (Mean Engagement Rate)", fontsize=13, weight="bold")
    plt.xlabel("Posting Hour (24-Hour UTC)", fontsize=11)
    plt.ylabel("Mean Engagement Score", fontsize=11)
    plt.xticks(range(0, 24))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "hour_vs_engagement.png", dpi=200)
    plt.close()
    
    # 3. Sentiment Distribution Across Platforms
    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=train_df,
        x="platform",
        hue="sentiment_label",
        palette={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
    )
    plt.title("Sentiment Polarity Distribution by Social Platform", fontsize=13, weight="bold")
    plt.xlabel("Platform", fontsize=11)
    plt.ylabel("Post Count", fontsize=11)
    plt.legend(title="Sentiment", loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "sentiment_by_platform.png", dpi=200)
    plt.close()
    
    # 4. Brand Volume and Negative Ratio (Crisis Anomaly Footprint)
    brand_stats = train_df.groupby("brand")["sentiment_label"].value_counts(normalize=True).unstack().fillna(0)
    brand_stats["Negative_pct"] = brand_stats["Negative"] * 100
    brand_stats = brand_stats.sort_values(by="Negative_pct", ascending=False)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=brand_stats.index, y=brand_stats["Negative_pct"], palette="Reds_r")
    plt.title("Brand Sentiment Vulnerability (% Negative Posts)", fontsize=13, weight="bold")
    plt.xlabel("Brand", fontsize=11)
    plt.ylabel("% Negative Posts", fontsize=11)
    plt.axhline(35, color="red", linestyle="--", label="Warning Anomaly Threshold (35%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "brand_negative_ratios.png", dpi=200)
    plt.close()
    
    logger.info(f"All EDA figures successfully saved to {figures_dir}")
    
    # Generate 01_eda.ipynb notebook
    notebook_path = settings.NOTEBOOKS_DIR / "01_eda.ipynb"
    nb_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SocialPulse AI — 01: Exploratory Data Analysis\n",
                "\n",
                "This notebook analyzes the **SocialPulse AI** dataset (20,000 social posts) to uncover:\n",
                "1. **Engagement Tier Imbalance** (Low vs Moderate vs Viral)\n",
                "2. **Temporal Dynamics** (Hour-of-day vs normalized engagement)\n",
                "3. **Platform & Brand Sentiment Breakdown**\n",
                "4. **Brand Crisis Outliers & Anomaly Episodes**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "\n",
                "# Load processed training data\n",
                "df = pd.read_csv('../data/processed/train.csv')\n",
                "print(f'Train shape: {df.shape}')\n",
                "df.head(3)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Class Distribution Analysis\n",
                "Examining class proportions across Low (50%), Moderate (35%), and Viral (15%)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "class_counts = df['engagement_label'].value_counts(normalize=True) * 100\n",
                "print(class_counts.to_string())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Temporal Engagement Patterns\n",
                "Investigating how time of posting correlates with viral probability."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['timestamp'] = pd.to_datetime(df['timestamp'])\n",
                "df['hour'] = df['timestamp'].dt.hour\n",
                "hourly = df.groupby(['hour', 'engagement_label']).size().unstack().fillna(0)\n",
                "hourly.plot(kind='bar', stacked=True, figsize=(12, 5), colormap='viridis')\n",
                "plt.title('Hourly Post Counts by Engagement Tier')\n",
                "plt.xlabel('Hour of Day')\n",
                "plt.ylabel('Posts Count')\n",
                "plt.show()"
            ]
        }
    ]
    
    nb_dict = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)
        
    logger.info(f"Created EDA notebook at {notebook_path}")

if __name__ == "__main__":
    run_eda()

