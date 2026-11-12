"""
Live Brand Crisis Injection & Detection Simulator for SocialPulse AI.
Allows demonstrating real-time crisis escalation on any chosen brand.
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from src.data.loader import DataLoader
from src.models.crisis_detector import load_crisis_detector
from src.utils.config import settings
from src.utils.logger import logger

CRISIS_COMPLAINT_TEMPLATES = [
    "CRITICAL BUG: @{brand} updated their software and deleted all user databases! Everything gone! #Boycott{brand} 😡🚨",
    "Security vulnerability in @{brand}! User passwords and credit card details leaked online on dark web forums! Fix this now! #DataBreach",
    "Server outage on @{brand} for 6 consecutive hours. Millions lost by businesses dependent on their API! Disgraceful incompetence!",
    "Customer service at @{brand} hangs up when asked about refunds for defective products! Absolute scam! Never using @{brand} again! 🤬",
    "Breaking news: Regulatory authorities launch probe into @{brand} fraudulent billing practices! Stock plunging! #Boycott{brand}",
]

def simulate_crisis(brand: str = "TechCorp", num_posts: int = 60) -> dict:
    loader = DataLoader()
    raw_df = loader.load_raw()
    detector = load_crisis_detector()

    brand_df = raw_df[raw_df["brand"] == brand].copy()
    brand_df["timestamp"] = pd.to_datetime(brand_df["timestamp"])
    latest_time = brand_df["timestamp"].max()

    # 1. Baseline analysis
    baseline_result = detector.analyze_brand(brand, brand_df, latest_time)
    print("\n" + "=" * 65)
    print(f"  [1] PRE-CRISIS BASELINE STATE — BRAND: {brand.upper()}")
    print("=" * 65)
    print(f"  Timestamp:               {baseline_result['timestamp']}")
    print(f"  Crisis Risk Score:       {baseline_result['crisis_risk_score']} / 100")
    print(f"  Status Level:            {baseline_result['risk_level']}")
    print(f"  1h Negative Ratio:       {baseline_result['signals']['negative_ratio_1h']*100:.1f}%")
    print(f"  Hourly Volume:           {baseline_result['metrics']['volume_1h']} posts")
    print("=" * 65)

    # 2. Inject acute negative viral flood within the last 45 minutes
    print(f"\n>> INJECTING {num_posts} SIMULATED CRISIS COMPLAINTS INTO LIVE STREAM...\n")
    new_records = []
    for i in range(num_posts):
        offset_mins = random.randint(1, 45)
        post_time = latest_time - timedelta(minutes=offset_mins)
        template = random.choice(CRISIS_COMPLAINT_TEMPLATES)
        text = template.format(brand=brand)

        new_records.append({
            "post_id": f"SIM_CRISIS_{i+1:04d}",
            "text": text,
            "platform": random.choice(["Twitter", "Reddit"]),
            "timestamp": post_time,
            "hashtags_count": 2,
            "mentions_count": 1,
            "has_media": random.choice([0, 1]),
            "follower_count": random.randint(2000, 85000),
            "likes": random.randint(40, 600),
            "shares": random.randint(30, 450),
            "comments": random.randint(20, 300),
            "brand": brand,
            "sentiment_label": "Negative",
        })

    injected_df = pd.concat([brand_df, pd.DataFrame(new_records)], ignore_index=True)
    injected_df["timestamp"] = pd.to_datetime(injected_df["timestamp"])

    # 3. Post-injection analysis
    post_crisis_result = detector.analyze_brand(brand, injected_df, latest_time)
    print("=" * 65)
    print(f"  [2] POST-INJECTION CRISIS STATE — BRAND: {brand.upper()}")
    print("=" * 65)
    print(f"  Timestamp:               {post_crisis_result['timestamp']}")
    print(f"  Crisis Risk Score:       {post_crisis_result['crisis_risk_score']} / 100")
    print(f"  Status Level:            {post_crisis_result['risk_level']}")
    print(f"  1h Negative Ratio:       {post_crisis_result['signals']['negative_ratio_1h']*100:.1f}%")
    print(f"  Volume Velocity Surge:   {post_crisis_result['signals']['volume_velocity_multiplier']:.1f}x baseline")
    print(f"  Z-Score Spike:           {post_crisis_result['signals']['z_score']} sigma")
    print(f"  Rule Tripwire Triggered: {post_crisis_result['signals']['rule_tripwire_triggered']}")
    print(f"  ML Anomaly Triggered:    {post_crisis_result['signals']['isolation_forest_anomaly_triggered']}")
    print(f"  Top Diagnostic Keywords: {', '.join(post_crisis_result['diagnostic_keywords'])}")
    print("=" * 65 + "\n")

    return {
        "baseline": baseline_result,
        "crisis": post_crisis_result,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate live brand crisis")
    parser.add_argument("--brand", type=str, default="TechCorp", help="Brand name to target")
    parser.add_argument("--num-posts", type=int, default=60, help="Number of negative posts to inject")
    args = parser.parse_args()
    simulate_crisis(brand=args.brand, num_posts=args.num_posts)
