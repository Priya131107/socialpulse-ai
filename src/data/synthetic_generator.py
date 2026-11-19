"""
SocialPulse AI — Synthetic Data Generator (Simplified).
Generates a realistic dataset of social media posts.
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.utils.config import settings
from src.utils.logger import logger

BRANDS = ["TechCorp", "FoodieApp", "SwiftAir", "NovaPay", "GreenWear", "SportsPro"]
PLATFORMS = ["Twitter", "Instagram", "Facebook", "LinkedIn"]
POSITIVE_WORDS = ["love", "amazing", "great", "excellent", "fantastic", "awesome", "brilliant"]
NEGATIVE_WORDS = ["terrible", "awful", "horrible", "worst", "hate", "bad", "broken", "failed"]
NEUTRAL_WORDS  = ["update", "new", "announcement", "feature", "release", "version", "today"]


def generate_social_dataset(n_posts: int = 5000, output_path: Path = None, random_seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic social-media dataset and save it to CSV."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    records = []
    start_date = datetime(2024, 1, 1)

    for i in range(n_posts):
        brand     = random.choice(BRANDS)
        platform  = random.choice(PLATFORMS)
        sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.5, 0.2, 0.3])[0]
        timestamp = start_date + timedelta(hours=random.randint(0, 60 * 24))

        # Build text
        if sentiment == "positive":
            words = random.sample(POSITIVE_WORDS, k=2)
        elif sentiment == "negative":
            words = random.sample(NEGATIVE_WORDS, k=2)
        else:
            words = random.sample(NEUTRAL_WORDS, k=2)
        text = f"{words[0].capitalize()} {words[1]} experience with {brand}! #{brand} #social"

        hashtags_count = random.randint(0, 5)
        mentions_count = random.randint(0, 3)
        has_media      = random.choice([0, 1])
        follower_count = max(1, int(np.random.lognormal(7, 1.5)))

        # Engagement correlated with sentiment + media
        base       = follower_count * 0.01
        multiplier = 1.5 if sentiment == "positive" else (0.6 if sentiment == "negative" else 1.0)
        likes      = max(0, int(np.random.poisson(base * multiplier * (1.3 if has_media else 1.0))))
        shares     = max(0, int(likes * random.uniform(0.1, 0.4)))
        comments   = max(0, int(likes * random.uniform(0.05, 0.3)))

        # Engagement label
        score = (likes + 2 * shares + 1.5 * comments) / max(follower_count, 1)
        if score < 0.01:
            engagement_label = "Low"
        elif score < 0.05:
            engagement_label = "Moderate"
        else:
            engagement_label = "Viral"

        records.append({
            "post_id":          f"P{i:05d}",
            "text":             text,
            "platform":         platform,
            "timestamp":        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "hashtags_count":   hashtags_count,
            "mentions_count":   mentions_count,
            "has_media":        has_media,
            "follower_count":   follower_count,
            "likes":            likes,
            "shares":           shares,
            "comments":         comments,
            "brand":            brand,
            "sentiment_label":  sentiment,
            "engagement_label": engagement_label,
        })

    df = pd.DataFrame(records)
    save_path = output_path or settings.RAW_DATA_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"Generated {len(df)} posts → {save_path}")
    return df


if __name__ == "__main__":
    generate_social_dataset()
