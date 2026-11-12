"""
Synthetic Social Media Engagement Dataset Generator.
Generates realistic social media post data with correlated interactions,
realistic follower distributions, content metadata, and engagement metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import config


# Templates and lexicons across diverse social media verticals
TOPIC_TEMPLATES = [
    # Tech & AI
    (
        "AI & Tech",
        [
            "Just launched our new open-source agent framework! It automates complex multi-step reasoning in seconds. Check it out: {url} {hashtags} {emojis}",
            "Why is nobody talking about the huge leap in local LLMs? We are running 70B models at 45 tokens/sec on consumer GPUs now. {emojis} {hashtags}",
            "Hot take: Prompt engineering will be obsolete in 12 months. Synthetic data and self-correcting reasoning loops are the real future. Agree or disagree? {emojis} {hashtags}",
            "We spent 6 months fine-tuning transformers on multimodal datasets. Here are 7 critical lessons we learned: 🧵👇 {url} {hashtags}",
            "Breakthrough: New paper shows zero-shot reasoning beats human benchmarks on competitive programming tasks! Thoughts? {emojis} {hashtags}",
            "Python 3.14 features you absolutely need to know. Thread on the new JIT compiler and performance gains: {url} {hashtags}",
            "Building in public update: Crossed 10,000 active developers using our API today! Huge thank you to everyone supporting us! {emojis} {hashtags}",
            "Is AGI closer than we think, or are we hitting a wall with current compute architectures? Let's discuss. {hashtags}",
        ],
        ["#AI", "#MachineLearning", "#TechTrends", "#OpenSource", "#Python", "#DataScience", "#DeepLearning"],
        ["🚀", "🤖", "🔥", "💡", "⚡", "🧠"],
    ),
    # Startups & Business
    (
        "Business & Career",
        [
            "Most founders fail because they build what they think users want instead of talking to 100 customers first. Brutal truth. {emojis} {hashtags}",
            "How we scaled our SaaS from $0 to $1M ARR in 14 months without any VC funding: A complete breakdown {url} {hashtags}",
            "If you want to fast-track your software career in 2026, stop grinding LeetCode and start shipping end-to-end applications with real users. {emojis} {hashtags}",
            "Remote work is not dying—bad management is. Here is how high-performing distributed teams actually communicate: {url} {hashtags}",
            "The best marketing strategy is building a product so absurdly good that customers can't stop tweeting about it. {emojis} {hashtags}",
            "Just raised our Seed Round! Excited to partner with incredible angels to build the next generation of developer tooling. {emojis} {hashtags}",
        ],
        ["#Startups", "#SaaS", "#Entrepreneurship", "#Productivity", "#CareerAdvice", "#TechLeadership"],
        ["💼", "📈", "🎯", "🚀", "✨", "🙌"],
    ),
    # Breaking News & Pop Culture
    (
        "Pop Culture & News",
        [
            "BREAKING: Historic announcement from space agency confirms major discovery on lunar south pole! Full coverage: {url} {emojis} {hashtags}",
            "Did anyone else just see what happened in the championship final?! Unbelievable comeback in the last 10 seconds! 😱 {emojis} {hashtags}",
            "The new sci-fi blockbuster has officially broken worldwide box office records on opening weekend! Did it live up to the hype? {emojis} {hashtags}",
            "Major server outage affecting millions of internet users worldwide right now. Are your services down too? {emojis} {hashtags}",
            "The soundtrack for this new game is pure art. Cannot stop listening on repeat all day! 🎧 {emojis} {hashtags}",
            "Trending worldwide: Millions react to the latest viral video taking over every social feed! What do you think? {url} {hashtags}",
        ],
        ["#BreakingNews", "#PopCulture", "#TrendingNow", "#Championship", "#Cinema", "#Viral"],
        ["🚨", "😱", "🏆", "🎬", "🔥", "👀"],
    ),
    # Finance & Web3
    (
        "Finance & Crypto",
        [
            "Market volatility hits 6-month highs after new economic data release. Here is how smart investors hedge risks: {url} {hashtags}",
            "Bitcoin breaks past resistance levels as institutional inflows hit record highs. The bull run continues! {emojis} {hashtags}",
            "Financial literacy should be taught in high school. 5 basic principles of compounding interest everyone must master: 🧵 {hashtags}",
            "Tech stocks rally 4% today led by semiconductor earnings surprises. Full breakdown here: {url} {hashtags}",
            "Decentralized finance protocols processed over $50B in volume this week alone. The future of global settlements? {emojis} {hashtags}",
        ],
        ["#Finance", "#Crypto", "#Bitcoin", "#Investing", "#StockMarket", "#Economy"],
        ["💰", "📊", "📈", "💎", "⚡", "🪙"],
    ),
    # Lifestyle, Health & Memes
    (
        "Lifestyle & Memes",
        [
            "Day 45 of morning workouts and clean eating. Energy levels are through the roof! Remember: consistency beats intensity every single time. {emojis} {hashtags}",
            "Me: I am going to sleep early tonight.`nAlso me at 2:30 AM reading Wikipedia articles on ancient Roman engineering: 😂 {emojis} {hashtags}",
            "Software engineers explaining why spending 4 hours automating a 5-minute task was totally worth it. {url} {emojis} {hashtags}",
            "The simplest productivity hack is turning off your phone notifications between 9 AM and 1 PM. Try it for 3 days and thank me later. {emojis} {hashtags}",
            "Weekend coffee + crisp morning air + great book = the ultimate recharge recipe. Have a restful Sunday everyone! ☕ {emojis} {hashtags}",
            "Debugging code in production on a Friday afternoon is an extreme sport. Who else survived today? 😅 {emojis} {hashtags}",
        ],
        ["#Lifestyle", "#Fitness", "#TechMemes", "#Humor", "#Productivity", "#Motivation", "#Wellness"],
        ["😂", "☕", "💪", "✨", "🧘", "💯", "😅"],
    ),
]

NOTABLE_MENTIONS = [
    "@elonmusk", "@OpenAI", "@sama", "@Google", "@Microsoft", "@ycombinator",
    "@lexfridman", "@paulg", "@techcrunch", "@wired", "@AndrewYNg", "@hubermanlab"
]

SAMPLE_URLS = [
    "https://t.co/aiNews2026", "https://t.co/techReport", "https://t.co/subStackPost",
    "https://t.co/gitRepoLive", "https://t.co/blogPostV1", "https://t.co/investUpdate"
]


def generate_synthetic_dataset(
    num_samples: int = 8000,
    random_seed: int = 42,
    output_path: Path = config.RAW_DATASET_PATH,
) -> pd.DataFrame:
    """
    Generates a realistic social media engagement dataset.

    Parameters:
    -----------
    num_samples : int
        Number of posts to generate (default 8,000).
    random_seed : int
        Seed for reproducibility.
    output_path : Path
        Where to save the raw CSV file.

    Returns:
    --------
    pd.DataFrame
        The generated dataset.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Generating {num_samples} realistic social media posts...")

    # Follower count follows a realistic log-normal distribution
    # Min ~50, Median ~4,000, Max ~2,500,000
    log_followers = np.random.normal(loc=8.2, scale=1.7, size=num_samples)
    followers = np.clip(np.exp(log_followers), 50, 3_000_000).astype(int)

    # Media type distribution: 45% none, 35% image, 20% video
    media_choices = ["none", "image", "video"]
    media_probs = [0.45, 0.35, 0.20]
    media_types = np.random.choice(media_choices, size=num_samples, p=media_probs)

    # Generate realistic timestamps across the past 365 days
    base_date = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = []
    for _ in range(num_samples):
        day_offset = random.randint(0, 360)
        # Hour distribution: higher traffic during 11:00-14:00 and 18:00-22:00
        hour_weights = [
            1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 12, 14, 13, 10, 9, 10, 12, 15, 16, 14, 11, 6, 3
        ]
        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = base_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
        timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))

    # Generate text & engagement interactions
    records = []

    for i in range(num_samples):
        topic_name, templates, tags_pool, emoji_pool = random.choice(TOPIC_TEMPLATES)
        template = random.choice(templates)

        # Decide hashtag count (0 to 6)
        num_tags = random.choices([0, 1, 2, 3, 4, 5, 7], weights=[0.1, 0.2, 0.35, 0.2, 0.08, 0.05, 0.02])[0]
        selected_tags = " ".join(random.sample(tags_pool, min(num_tags, len(tags_pool)))) if num_tags > 0 else ""

        # Decide emoji count (0 to 4)
        num_emojis = random.choices([0, 1, 2, 3, 4], weights=[0.15, 0.35, 0.3, 0.15, 0.05])[0]
        selected_emojis = "".join(random.choices(emoji_pool, k=num_emojis))

        # Decide URL presence
        url_str = random.choice(SAMPLE_URLS) if ("{url}" in template and random.random() < 0.75) else ""

        # Format text
        text = template.format(
            url=url_str,
            hashtags=selected_tags,
            emojis=selected_emojis
        ).strip()

        # Occasionally add an @mention (25% probability)
        mention_flag = False
        if random.random() < 0.25:
            mention = random.choice(NOTABLE_MENTIONS)
            text = f"{mention} {text}"
            mention_flag = True

        # Clean multiple spaces
        text = " ".join(text.split())

        # Follower count for this sample
        follower_cnt = followers[i]
        media_type = media_types[i]
        dt = datetime.strptime(timestamps[i], "%Y-%m-%d %H:%M:%S")
        hour = dt.hour
        is_weekend = 1 if dt.weekday() >= 5 else 0

        # ---- Calculate Realistic Engagement Multipliers ----
        # Base engagement rate by follower tier:
        # Micro accounts have higher % interaction; massive accounts have lower % but high total
        if follower_cnt < 2000:
            base_rate = np.random.uniform(0.015, 0.050)
        elif follower_cnt < 25000:
            base_rate = np.random.uniform(0.008, 0.025)
        elif follower_cnt < 200000:
            base_rate = np.random.uniform(0.004, 0.012)
        else:
            base_rate = np.random.uniform(0.001, 0.005)

        # Multiplier 1: Media Type
        media_multiplier = 1.0
        if media_type == "image":
            media_multiplier = np.random.uniform(1.3, 1.7)
        elif media_type == "video":
            media_multiplier = np.random.uniform(1.6, 2.3)

        # Multiplier 2: Time of Day (Peak hours: 12-14 and 18-22)
        time_multiplier = 1.0
        if 12 <= hour <= 14 or 18 <= hour <= 22:
            time_multiplier = np.random.uniform(1.25, 1.55)
        elif 1 <= hour <= 6:
            time_multiplier = np.random.uniform(0.55, 0.85)

        # Multiplier 3: Hashtags (optimal 2-4; too many gets spam penalty)
        hashtag_multiplier = 1.0
        if 2 <= num_tags <= 4:
            hashtag_multiplier = np.random.uniform(1.15, 1.40)
        elif num_tags >= 6:
            hashtag_multiplier = np.random.uniform(0.70, 0.90)
        elif num_tags == 0:
            hashtag_multiplier = np.random.uniform(0.85, 1.0)

        # Multiplier 4: Content Engagement Triggers
        content_multiplier = 1.0
        if "?" in text:
            content_multiplier *= np.random.uniform(1.15, 1.35)  # sparks comments/replies
        if "!" in text or any(e in text for e in ["🔥", "🚨", "😱", "🚀"]):
            content_multiplier *= np.random.uniform(1.10, 1.30)  # sparks likes & shares
        if mention_flag:
            content_multiplier *= np.random.uniform(1.05, 1.25)

        # Multiplier 5: Virality stochastic shock (heavy-tailed Pareto / Log-Normal event)
        # Roughly 10-12% of posts experience an algorithmic breakout / viral boost
        viral_boost = 1.0
        viral_roll = np.random.random()
        if viral_roll > 0.88:
            viral_boost = np.random.uniform(2.5, 6.0)
        elif viral_roll > 0.97:
            viral_boost = np.random.uniform(6.0, 15.0)

        # Total expected engagement base
        expected_total = follower_cnt * base_rate * media_multiplier * time_multiplier * hashtag_multiplier * content_multiplier * viral_boost

        # Add realistic noise
        total_interactions = max(1, int(np.random.poisson(max(1.0, expected_total))))

        # Distribute into likes (65-80%), shares (10-20%), comments (5-15%)
        like_ratio = np.random.uniform(0.68, 0.80)
        share_ratio = np.random.uniform(0.10, 0.20)
        comment_ratio = max(0.02, 1.0 - like_ratio - share_ratio)

        # If post has a question, comments ratio increases
        if "?" in text:
            comment_ratio *= 1.6
            like_ratio -= 0.05
            share_ratio -= 0.05

        norm = like_ratio + share_ratio + comment_ratio
        like_ratio /= norm
        share_ratio /= norm
        comment_ratio /= norm

        likes = max(0, int(total_interactions * like_ratio))
        shares = max(0, int(total_interactions * share_ratio))
        comments = max(0, int(total_interactions * comment_ratio))

        records.append({
            "post_id": f"post_{i+1:06d}",
            "text": text,
            "followers": follower_cnt,
            "timestamp": timestamps[i],
            "media_type": media_type,
            "likes": likes,
            "shares": shares,
            "comments": comments,
        })

    df = pd.DataFrame(records)

    # Save raw dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully generated and saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(df.head(3))

    return df


if __name__ == "__main__":
    generate_synthetic_dataset(num_samples=8000, random_seed=config.RANDOM_SEED)
