"""
Realistic Social Media Dataset Generator for SocialPulse AI.
Generates synthetic social media posts across platforms and brands with
statistically realistic interaction dynamics, follower distributions,
temporal patterns, and crisis anomaly episodes.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.logger import logger

BRANDS = [
    "TechCorp",
    "StreamFlix",
    "SwiftAir",
    "NovaPay",
    "EcoWear",
    "CryptoZen",
    "FoodieHub",
    "AutoDrive",
]

PLATFORMS = ["Twitter", "Instagram", "LinkedIn", "Reddit"]
PLATFORM_WEIGHTS = [0.45, 0.25, 0.15, 0.15]

# Domain-specific templates with positive, neutral, and negative variants
TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "TechCorp": {
        "Positive": [
            "Just upgraded to the new @TechCorp AI cloud suite! Inference speed increased 4x with zero downtime. Game changer! {emojis} {hashtags}",
            "Huge shoutout to @TechCorp support team for resolving our enterprise migration ticket in under 15 minutes! Amazing service. {emojis} {hashtags}",
            "The new @TechCorp developer docs and SDK are so clean and developer-friendly. Loving the developer experience! {emojis} {hashtags}",
            "Our engineering team just deployed @TechCorp microservices architecture to production. Latency down by 40%! {emojis} {hashtags}",
        ],
        "Neutral": [
            "@TechCorp announced their Q3 developer summit dates and keynote speaker lineup today. Check agenda: {url} {hashtags}",
            "Exploring the trade-offs between @TechCorp cloud database and self-hosted open source solutions for our startup. {hashtags}",
            "Here is a summary of the latest @TechCorp API release notes and deprecation schedule: {url} {hashtags}",
            "Does anyone have benchmarks comparing @TechCorp inference latency versus local model deployments? {hashtags}",
        ],
        "Negative": [
            "@TechCorp server outage right in the middle of our product launch! Entire dashboard is throwing 502 errors. Unacceptable! {emojis} {hashtags}",
            "Shocking data privacy glitch reported on @TechCorp developer portal. API keys exposed in plain text logs! Fix this now! {emojis} {hashtags}",
            "Terrible update from @TechCorp. Cloud subscription costs jumped 60% with zero prior notice. Looking for alternatives. {emojis} {hashtags}",
            "Customer support at @TechCorp has completely ghosted us for 48 hours while our production database is down. Worst experience ever. {emojis} {hashtags}",
        ],
    },
    "StreamFlix": {
        "Positive": [
            "The new sci-fi original series on @StreamFlix is absolute cinema! Binge-watched all 8 episodes in one night! {emojis} {hashtags}",
            "Thank you @StreamFlix for finally bringing 4K HDR and spatial audio to the family plan without extra fees! {emojis} {hashtags}",
            "Can we talk about the incredible soundtrack of the latest @StreamFlix documentary? Pure masterpiece. {emojis} {hashtags}",
            "Best weekend ever thanks to @StreamFlix movie recommendations algorithm. Hit after hit! {emojis} {hashtags}",
        ],
        "Neutral": [
            "@StreamFlix releases complete list of movies and TV shows arriving next month: {url} {hashtags}",
            "Comparing video bitrate and compression algorithms between @StreamFlix and rival platforms. Interesting findings: {url} {hashtags}",
            "What should I watch next on @StreamFlix? Open to thriller or drama recommendations! {hashtags}",
            "@StreamFlix announced a new international co-production partnership today. Details: {url} {hashtags}",
        ],
        "Negative": [
            "Why is @StreamFlix canceling every single great show after just one season?! Canceling my subscription right now! {emojis} {hashtags}",
            "@StreamFlix streaming quality is constantly buffering in 480p despite my 1Gbps fiber internet connection. Horrible app! {emojis} {hashtags}",
            "Another unjustified price hike from @StreamFlix while removing half the classic catalog? Greedy and anti-consumer. {emojis} {hashtags}",
            "The new @StreamFlix UI update is completely broken on smart TVs. Constant crashes and freezes. {emojis} {hashtags}",
        ],
    },
    "SwiftAir": {
        "Positive": [
            "Flawless flight with @SwiftAir today! On-time departure, spacious legroom, and super friendly cabin crew. 10/10! {emojis} {hashtags}",
            "Lost my iPad on @SwiftAir flight 402 and the ground staff tracked it down and returned it within an hour! Outstanding service! {emojis} {hashtags}",
            "The new @SwiftAir premium economy seats are totally worth the upgrade. Landed completely refreshed! {emojis} {hashtags}",
            "Kudos to the pilot on @SwiftAir for that butter-smooth landing in turbulent weather! {emojis} {hashtags}",
        ],
        "Neutral": [
            "@SwiftAir expands direct non-stop routes between transatlantic hubs starting this winter: {url} {hashtags}",
            "Checking in for my @SwiftAir flight using the mobile boarding pass. Anyone know if terminal 4 has lounge access? {hashtags}",
            "@SwiftAir announced their annual fleet modernizing plan with new fuel-efficient aircraft. {url} {hashtags}",
            "Comparing baggage policies between @SwiftAir and budget domestic carriers. {url} {hashtags}",
        ],
        "Negative": [
            "Absolute disaster with @SwiftAir! Stranded at terminal for 14 hours with zero food, water, or updates after canceled flight! #SwiftAirCrisis {emojis} {hashtags}",
            "@SwiftAir lost all checked luggage for our entire family on an international flight and customer desk refused to help! Incompetent! {emojis} {hashtags}",
            "Overbooked flight by @SwiftAir kicked confirmed passengers off the plane at the gate! Boycott @SwiftAir! #BoycottSwiftAir {emojis} {hashtags}",
            "Flight delayed 6 times, no refund offered, call center hangs up after 2 hours on hold. Never flying @SwiftAir again! {emojis} {hashtags}",
        ],
    },
    "NovaPay": {
        "Positive": [
            "Sending cross-border payments with @NovaPay settled in literally 3 seconds with minimal fees! Future of fintech! {emojis} {hashtags}",
            "The new @NovaPay virtual credit card with auto-rotating security codes is pure genius for online shopping security. {emojis} {hashtags}",
            "Resolved an unauthorized vendor charge through @NovaPay fraud dispute in under 10 minutes. Impressive security! {emojis} {hashtags}",
            "Loving the high-yield savings vault feature on @NovaPay! Interest deposited daily with complete liquidity. {emojis} {hashtags}",
        ],
        "Neutral": [
            "@NovaPay rolls out tap-to-pay support for merchant terminals across regional retailers: {url} {hashtags}",
            "A technical breakdown of @NovaPay real-time settlement engine architecture: {url} {hashtags}",
            "Are @NovaPay merchant processing fees competitive compared to legacy gateway providers? Let's analyze: {url} {hashtags}",
            "@NovaPay announces strategic partnership with major banking networks today. {hashtags}",
        ],
        "Negative": [
            "CRITICAL ALERT: @NovaPay systems are down worldwide! Transactions failing, bank balances displaying zero! Where is our money?! #NovaPayDown {emojis} {hashtags}",
            "@NovaPay locked my verified business account with $45,000 inside for 3 weeks with zero explanation. Payroll is frozen! Criminal behavior! {emojis} {hashtags}",
            "Security vulnerability in @NovaPay mobile app allowing unauthorized transaction withdrawals without 2FA! Fix this immediately! {emojis} {hashtags}",
            "Customer service at @NovaPay is automated bot loops that solve nothing while users lose funds. Worst fintech service! {emojis} {hashtags}",
        ],
    },
    "EcoWear": {
        "Positive": [
            "So proud to wear @EcoWear jacket made from 100% recycled ocean plastic! Incredible warmth and style. {emojis} {hashtags}",
            "The durability of @EcoWear boots is unmatched after hiking 200 miles through rugged trails. Great brand! {emojis} {hashtags}",
            "Love @EcoWear zero-waste packaging initiative. Setting a real benchmark for ethical fashion! {emojis} {hashtags}",
            "Top quality organic cotton shirts from @EcoWear. Breathable, comfortable, and ethically manufactured. {emojis} {hashtags}",
        ],
        "Neutral": [
            "@EcoWear publishes their annual sustainability audit and carbon neutrality progress report: {url} {hashtags}",
            "Can anyone recommend sizing tips for @EcoWear winter coats? Do they run true to size? {hashtags}",
            "Exploring the supply chain logistics behind @EcoWear circular textile recycling program: {url} {hashtags}",
            "@EcoWear opens new flagship eco-conscious retail store in downtown. Details: {url} {hashtags}",
        ],
        "Negative": [
            "Greenwashing exposed: Independent investigation reveals @EcoWear synthetic fibers are sourced from non-recycled factories! Disgraceful! {emojis} {hashtags}",
            "@EcoWear jacket ripped at the seam on the very first day. Paid $250 for low-tier cheap quality! Return denied! {emojis} {hashtags}",
            "Customer support at @EcoWear refuses to refund defective items sent in wrong sizes. Scam brand! {emojis} {hashtags}",
            "Toxic dye chemicals found in recent @EcoWear clothing batch according to consumer safety reports! Disgusting! {emojis} {hashtags}",
        ],
    },
    "CryptoZen": {
        "Positive": [
            "@CryptoZen cold storage wallet security setup is bulletproof. Easiest self-custody experience available! {emojis} {hashtags}",
            "Staking yields on @CryptoZen decentralized protocol just crossed 8.5% APY with on-chain transparency! Bullish! {emojis} {hashtags}",
            "Instant fiat on-ramp on @CryptoZen without exorbitant exchange slippage. Game changer for crypto adoption! {emojis} {hashtags}",
            "The new portfolio tracker on @CryptoZen makes tax reporting so effortless and clean. {emojis} {hashtags}",
        ],
        "Neutral": [
            "@CryptoZen protocol completes comprehensive smart contract audit with zero critical vulnerabilities reported: {url} {hashtags}",
            "Comparing gas optimization strategies between @CryptoZen and competing Layer-2 rollups: {url} {hashtags}",
            "What are your thoughts on @CryptoZen newly proposed governance token distribution model? {hashtags}",
            "Weekly volume on @CryptoZen automated liquidity pools reached $120M: {url} {hashtags}",
        ],
        "Negative": [
            "EMERGENCY: @CryptoZen liquidity bridge exploited for $30M! Withdrawals paused indefinitely! Rug pull suspicion! #CryptoZenExploit {emojis} {hashtags}",
            "Cannot withdraw my funds from @CryptoZen for 48 hours. Server errors everywhere! Is this another exchange collapse?! {emojis} {hashtags}",
            "Phishing links surfaced inside the official @CryptoZen discord community due to compromised admin credentials! Stay away! {emojis} {hashtags}",
            "Worst exchange spread on @CryptoZen hidden under misleading fee structures. Lost 7% on execution slippage! {emojis} {hashtags}",
        ],
    },
    "FoodieHub": {
        "Positive": [
            "Best meal delivery of my life thanks to @FoodieHub! Arrived in 20 minutes piping hot with eco packaging! {emojis} {hashtags}",
            "The artisan chef collaboration on @FoodieHub this week is sensational! Michelin-star dining at home. {emojis} {hashtags}",
            "Amazing driver from @FoodieHub braved torrential rain to deliver birthday dinner on time. Tipped extra! {emojis} {hashtags}",
            "Loving the local bakery discovery section on @FoodieHub! Hidden gems everywhere! {emojis} {hashtags}",
        ],
        "Neutral": [
            "@FoodieHub announces partnership with local organic grocery cooperatives for 1-hour pantry delivery: {url} {hashtags}",
            "Comparing commission rates for small family restaurants on @FoodieHub versus competing platforms: {url} {hashtags}",
            "Looking for authentic regional curry spots available on @FoodieHub in the metro area. Any suggestions? {hashtags}",
            "@FoodieHub releases annual dining trends report showing surge in plant-based orders. {url} {hashtags}",
        ],
        "Negative": [
            "Food poisoning from @FoodieHub delivered meal! Delivery was 2 hours late and ice cold! Health department notified! {emojis} {hashtags}",
            "@FoodieHub delivery driver stole half the items from our sealed family order and app refused to refund! Terrible service! {emojis} {hashtags}",
            "Hidden service fees on @FoodieHub doubled the order total from $28 to $62 at checkout! Deceptive pricing! {emojis} {hashtags}",
            "App charged my card 3 times for one order that never arrived. Support bot is completely useless! Boycott @FoodieHub! {emojis} {hashtags}",
        ],
    },
    "AutoDrive": {
        "Positive": [
            "Took @AutoDrive autonomous taxi through downtown rush hour traffic today. Smooth navigation, zero interventions, totally futuristic! {emojis} {hashtags}",
            "The new @AutoDrive EV battery architecture delivered 450 miles on a single fast charge! Range anxiety officially gone! {emojis} {hashtags}",
            "Love the over-the-air software updates on @AutoDrive! Woke up this morning to a brand new UI and faster acceleration! {emojis} {hashtags}",
            "Safety ratings for @AutoDrive vehicle series earned 5 stars across every crash test category. Impressive engineering! {emojis} {hashtags}",
        ],
        "Neutral": [
            "@AutoDrive submits regulatory filings for commercial driverless robotaxi operations in three new metropolitan zones: {url} {hashtags}",
            "An engineer's breakdown of @AutoDrive sensor fusion combining lidar, radar, and vision cameras: {url} {hashtags}",
            "Comparing long-term battery degradation statistics of @AutoDrive vehicles over 100,000 miles: {url} {hashtags}",
            "@AutoDrive announces construction of new gigafactory for next-generation electric drivetrains: {url} {hashtags}",
        ],
        "Negative": [
            "SAFETY HAZARD: @AutoDrive autonomous mode phantom-braked violently on highway with no obstacles ahead! Almost caused massive collision! {emojis} {hashtags}",
            "@AutoDrive recalls 80,000 electric vehicles due to critical steering column software flaw. Massive defect! #AutoDriveRecall {emojis} {hashtags}",
            "Customer service at @AutoDrive quoted 4 months wait time for a simple replacement door latch on a brand new car. Shocking! {emojis} {hashtags}",
            "Battery on @AutoDrive vehicle died completely after 18 months, company refusing warranty replacement! Total lemon! {emojis} {hashtags}",
        ],
    },
}

HASHTAG_BANK = {
    "Positive": ["#Innovation", "#LoveIt", "#TechLife", "#GameChanger", "#Awesome", "#Quality", "#MustHave", "#TopTier"],
    "Neutral": ["#Update", "#IndustryNews", "#Review", "#Discussion", "#Insights", "#Analysis", "#TechTrends"],
    "Negative": ["#CustomerServiceFail", "#Disappointed", "#Boycott", "#TerribleService", "#Scam", "#Broken", "#Unacceptable", "#WorstEver"],
}

EMOJI_BANK = {
    "Positive": ["🚀", "🔥", "💡", "🙌", "✨", "❤️", "👏", "🎉", "💯", "⭐"],
    "Neutral": ["📊", "🔍", "📌", "🌐", "📱", "💼", "👀", "📝"],
    "Negative": ["😡", "🤬", "🤮", "🚨", "⚠️", "👎", "💔", "🤦‍♂️", "😤", "📉"],
}


def generate_social_dataset(
    num_records: int = 20000,
    random_seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate synthetic social media dataset with realistic distributions,
    temporal features, and crisis anomaly episodes.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    logger.info(f"Generating synthetic social dataset ({num_records} records)...")

    # Time frame: 30 days prior to today
    base_time = datetime(2026, 10, 1, 0, 0, 0)
    total_seconds = 30 * 24 * 3600

    # Lognormal follower distribution: median ~ 3,500, range ~ 100 to 1,500,000
    followers_pool = np.random.lognormal(mean=8.2, sigma=1.3, size=num_records)
    followers_pool = np.clip(followers_pool, 50, 2_000_000).astype(int)

    records = []

    # Crisis injection windows:
    # 1. SwiftAir cancellation crisis on Days 10-12 (seconds: 10*86400 to 12*86400)
    # 2. NovaPay payment gateway outage on Days 20-21 (seconds: 20*86400 to 21*86400)
    crisis_swift_start = 10 * 86400
    crisis_swift_end = 12 * 86400
    crisis_nova_start = 20 * 86400
    crisis_nova_end = 21 * 86400

    for i in range(num_records):
        post_id = f"POST_{i+1:06d}"
        platform = np.random.choice(PLATFORMS, p=PLATFORM_WEIGHTS)
        brand = random.choice(BRANDS)
        
        # Timestamp distribution with peak hours (12-14 lunch, 18-22 evening)
        offset_seconds = random.randint(0, total_seconds)
        timestamp = base_time + timedelta(seconds=offset_seconds)
        hour = timestamp.hour

        # Determine sentiment with crisis anomaly influence
        is_swift_crisis = (brand == "SwiftAir" and crisis_swift_start <= offset_seconds <= crisis_swift_end)
        is_nova_crisis = (brand == "NovaPay" and crisis_nova_start <= offset_seconds <= crisis_nova_end)

        if is_swift_crisis or is_nova_crisis:
            # 85% negative sentiment during crisis episodes
            sentiment_label = np.random.choice(["Positive", "Neutral", "Negative"], p=[0.05, 0.10, 0.85])
        else:
            # Normal baseline: 40% positive, 35% neutral, 25% negative
            sentiment_label = np.random.choice(["Positive", "Neutral", "Negative"], p=[0.40, 0.35, 0.25])

        # Pick text template
        templates_pool = TEMPLATES[brand][sentiment_label]
        template = random.choice(templates_pool)

        # Dynamic template formatting
        num_hashtags = random.choices([1, 2, 3, 4, 5], weights=[0.25, 0.35, 0.25, 0.10, 0.05])[0]
        selected_hashtags = random.sample(HASHTAG_BANK[sentiment_label], min(num_hashtags, len(HASHTAG_BANK[sentiment_label])))
        hashtags_str = " ".join(selected_hashtags)

        num_emojis = random.choices([0, 1, 2, 3], weights=[0.20, 0.40, 0.30, 0.10])[0]
        selected_emojis = "".join(random.sample(EMOJI_BANK[sentiment_label], num_emojis)) if num_emojis > 0 else ""

        url_str = f"https://pulse.io/{brand.lower()}/{random.randint(1000, 9999)}" if "{url}" in template else ""
        
        text = template.format(
            emojis=selected_emojis,
            hashtags=hashtags_str,
            url=url_str,
        ).strip()

        # Count metadata
        hashtags_count = len(selected_hashtags)
        mentions_count = text.count("@")
        has_media = bool(random.random() < (0.65 if platform in ["Instagram", "Twitter"] else 0.40))
        follower_count = int(followers_pool[i])

        # Interaction simulation with statistical correlations:
        # Base engagement rate
        base_rate = np.random.beta(a=2.0, b=45.0)  # typically 2-6% engagement
        
        # Multipliers
        media_mult = 1.6 if has_media else 1.0
        peak_hour_mult = 1.4 if (12 <= hour <= 14 or 18 <= hour <= 22) else 0.85
        
        # Controversy / crisis multiplier (negative crisis posts get massive comment/share spikes)
        if sentiment_label == "Negative" and (is_swift_crisis or is_nova_crisis):
            sentiment_mult = 2.8
            share_boost = 3.5
            comment_boost = 3.0
        elif sentiment_label in ["Positive", "Negative"]:
            sentiment_mult = 1.3
            share_boost = 1.2
            comment_boost = 1.2
        else:
            sentiment_mult = 0.9
            share_boost = 0.8
            comment_boost = 0.8

        expected_interactions = follower_count * base_rate * media_mult * peak_hour_mult * sentiment_mult
        
        # Split into likes, shares, comments
        likes = max(0, int(np.random.poisson(lam=max(1, expected_interactions * 0.70))))
        shares = max(0, int(np.random.poisson(lam=max(0, expected_interactions * 0.20 * share_boost))))
        comments = max(0, int(np.random.poisson(lam=max(0, expected_interactions * 0.10 * comment_boost))))

        # Random outliers (viral breakouts ~ 3% of posts)
        if random.random() < 0.035:
            viral_scale = random.uniform(5.0, 25.0)
            likes = int(likes * viral_scale)
            shares = int(shares * viral_scale * 1.5)
            comments = int(comments * viral_scale * 1.2)

        records.append({
            "post_id": post_id,
            "text": text,
            "platform": platform,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "hashtags_count": hashtags_count,
            "mentions_count": mentions_count,
            "has_media": int(has_media),
            "follower_count": follower_count,
            "likes": likes,
            "shares": shares,
            "comments": comments,
            "brand": brand,
            "sentiment_label": sentiment_label,
        })

    df = pd.DataFrame(records)
    
    # Save if output path specified
    save_path = output_path or settings.RAW_DATA_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8")
    logger.info(f"Successfully generated {len(df)} records and saved to {save_path}")
    
    return df


if __name__ == "__main__":
    generate_social_dataset()

