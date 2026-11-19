"""
SocialPulse AI — Streamlit Dashboard (Simplified).
3 pages: Overview, Post Predictor, Sentiment Analyzer.
No XGBoost, no transformers, no SHAP — just clean and functional.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SocialPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-setup: generate data + train model if missing ───────────────────────
def _ensure_ready():
    raw_path   = PROJECT_ROOT / "data" / "raw"   / "posts_dataset.csv"
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    model_path = PROJECT_ROOT / "models_saved"   / "best_viral_model.joblib"

    if raw_path.exists() and train_path.exists() and model_path.exists():
        return  # all good

    with st.spinner("⚙️ First-run setup (< 30 s) — generating data & training model…"):
        if not raw_path.exists():
            from src.data.synthetic_generator import generate_social_dataset
            generate_social_dataset(n_posts=5000, output_path=raw_path)

        if not train_path.exists():
            from src.data.preprocess import run_pipeline
            run_pipeline()

        if not model_path.exists():
            from src.models.viral_model import run_training
            run_training()

    st.success("✅ Ready!")
    time.sleep(0.8)
    st.rerun()

_ensure_ready()

# ── Load data & model ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_data():
    path = PROJECT_ROOT / "data" / "raw" / "posts_dataset.csv"
    if not path.exists():
        from src.data.synthetic_generator import generate_social_dataset
        df = generate_social_dataset(n_posts=5000, output_path=path)
    else:
        df = pd.read_csv(path)
    
    # Guarantee necessary columns exist with robust fallbacks
    if "engagement_label" not in df.columns:
        likes = df.get("likes", 10)
        shares = df.get("shares", 2)
        comments = df.get("comments", 1)
        followers = df.get("follower_count", 1000).replace(0, 1)
        score = (likes + 2 * shares + 1.5 * comments) / followers
        df["engagement_label"] = pd.cut(
            score, bins=[-float("inf"), 0.01, 0.05, float("inf")], labels=["Low", "Moderate", "Viral"]
        ).fillna("Moderate")
    
    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = "positive"
        
    if "brand" not in df.columns:
        df["brand"] = "TechCorp"
        
    if "platform" not in df.columns:
        df["platform"] = "Twitter"

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_resource
def load_predictor():
    from src.models.viral_model import ViralityPredictor
    return ViralityPredictor()

@st.cache_resource
def get_sentiment_fn():
    from src.models.sentiment_model import predict_sentiment
    return predict_sentiment

df       = load_data()
predictor = load_predictor()
predict_sentiment = get_sentiment_fn()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ SocialPulse AI")
st.sidebar.caption("Social Media Intelligence Platform")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔮 Post Predictor", "💬 Sentiment Analyzer"],
)
st.sidebar.divider()
st.sidebar.info("Built with scikit-learn, VADER, FastAPI, and Streamlit.")

# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================
if page == "📊 Overview":
    st.title("📊 Executive Overview")
    st.caption("Summary statistics across all analysed posts.")

    # KPI row
    total      = len(df)
    viral_pct  = round((df["engagement_label"] == "Viral").mean() * 100, 1)
    neg_pct    = round((df["sentiment_label"]  == "negative").mean() * 100, 1)
    brands     = df["brand"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Posts",      f"{total:,}")
    c2.metric("Viral Posts",      f"{viral_pct}%")
    c3.metric("Negative Sentiment", f"{neg_pct}%")
    c4.metric("Brands Tracked",   brands)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Engagement Distribution")
        eng_counts = df["engagement_label"].value_counts().reset_index()
        eng_counts.columns = ["Label", "Count"]
        fig = px.pie(eng_counts, names="Label", values="Count",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sentiment Distribution")
        sent_counts = df["sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        fig2 = px.bar(sent_counts, x="Sentiment", y="Count",
                      color="Sentiment",
                      color_discrete_map={"positive": "#22c55e", "neutral": "#64748b", "negative": "#ef4444"})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Posts per Platform")
    plat = df.groupby(["platform", "engagement_label"]).size().reset_index(name="count")
    fig3 = px.bar(plat, x="platform", y="count", color="engagement_label", barmode="group",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Brand Crisis Snapshot")
    brands_list = sorted(df["brand"].unique())
    brand_stats = []
    for b in brands_list:
        bdf = df[df["brand"] == b]
        neg_r = (bdf["sentiment_label"] == "negative").mean()
        score = int(min(100, neg_r * 200))
        level = "🔴 Critical" if score >= 60 else ("🟡 Warning" if score >= 30 else "🟢 Normal")
        brand_stats.append({"Brand": b, "Posts": len(bdf), "Neg %": f"{neg_r*100:.1f}%",
                             "Risk Score": score, "Status": level})
    st.dataframe(pd.DataFrame(brand_stats), use_container_width=True)

# =============================================================================
# PAGE 2 — POST PREDICTOR
# =============================================================================
elif page == "🔮 Post Predictor":
    st.title("🔮 Post Virality Predictor")
    st.caption("Enter your post details and predict whether it will go Viral, Moderate, or Low.")

    with st.form("predict_form"):
        text           = st.text_area("Post Text", placeholder="Write your post here…", height=120)
        col1, col2     = st.columns(2)
        platform       = col1.selectbox("Platform", ["Twitter", "Instagram", "Facebook", "LinkedIn"])
        follower_count = col2.number_input("Follower Count", min_value=0, value=5000, step=100)
        col3, col4, col5 = st.columns(3)
        hashtags_count = col3.slider("Hashtags", 0, 10, 2)
        mentions_count = col4.slider("Mentions", 0, 5, 0)
        has_media      = col5.selectbox("Has Media?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        submitted      = st.form_submit_button("⚡ Predict Engagement", use_container_width=True)

    if submitted:
        if not text.strip():
            st.warning("Please enter some post text.")
        else:
            with st.spinner("Predicting…"):
                result = predictor.predict({
                    "text": text, "platform": platform,
                    "follower_count": follower_count,
                    "hashtags_count": hashtags_count,
                    "mentions_count": mentions_count,
                    "has_media": has_media,
                })
            label = result["prediction"]
            proba = result.get("probabilities", {})

            colour = {"Viral": "🔥", "Moderate": "⚡", "Low": "📉"}.get(label, "⚡")
            st.success(f"**Predicted Engagement: {colour} {label}**")

            if proba:
                st.subheader("Confidence Scores")
                prob_df = pd.DataFrame(list(proba.items()), columns=["Label", "Probability"])
                fig = px.bar(prob_df, x="Label", y="Probability", color="Label",
                             color_discrete_sequence=["#22c55e", "#f59e0b", "#ef4444"],
                             range_y=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            # Tips
            st.subheader("💡 Tips to improve engagement")
            tips = []
            if hashtags_count < 2:  tips.append("Add 2–4 relevant hashtags.")
            if not has_media:       tips.append("Include an image or video — media posts get 1.3× more likes.")
            if len(text) < 50:     tips.append("Write a longer, more descriptive post.")
            if mentions_count == 0: tips.append("Tag a relevant account or collaborator.")
            for tip in tips:
                st.info(f"→ {tip}")
            if not tips:
                st.success("Your post is already well-optimised!")

# =============================================================================
# PAGE 3 — SENTIMENT ANALYZER
# =============================================================================
elif page == "💬 Sentiment Analyzer":
    st.title("💬 Sentiment Analyzer")
    st.caption("Analyse the sentiment of any text using VADER.")

    tab1, tab2 = st.tabs(["Single Text", "Batch Analysis"])

    with tab1:
        text = st.text_area("Enter text to analyse", height=150,
                             placeholder="e.g. 'I absolutely love this product!'")
        if st.button("Analyse", use_container_width=True):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                result = predict_sentiment(text)
                label  = result["label"]
                colour = {"Positive": "🟢", "Neutral": "🔵", "Negative": "🔴"}.get(label, "🔵")
                st.success(f"**Sentiment: {colour} {label}** (confidence: {result['confidence']:.2f})")

                scores = result["scores"]
                score_df = pd.DataFrame([
                    {"Component": "Positive", "Score": scores["pos"]},
                    {"Component": "Neutral",  "Score": scores["neu"]},
                    {"Component": "Negative", "Score": scores["neg"]},
                ])
                fig = px.bar(score_df, x="Component", y="Score", color="Component",
                             color_discrete_map={"Positive": "#22c55e", "Neutral": "#64748b", "Negative": "#ef4444"},
                             range_y=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("Paste one post per line:")
        raw = st.text_area("Posts (one per line)", height=200,
                            placeholder="Great product!\nTerrible service.\nJust an update.")
        if st.button("Analyse All", use_container_width=True):
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            if not lines:
                st.warning("No text entered.")
            else:
                results = [{"Text": t, **predict_sentiment(t)} for t in lines]
                rdf = pd.DataFrame(results)
                rdf["label_icon"] = rdf["label"].map({"Positive": "🟢", "Neutral": "🔵", "Negative": "🔴"})
                st.dataframe(rdf[["Text", "label_icon", "label", "confidence"]]
                               .rename(columns={"label_icon": " ", "label": "Sentiment",
                                                "confidence": "Confidence"}),
                             use_container_width=True)

                # Summary donut
                counts = rdf["label"].value_counts().reset_index()
                counts.columns = ["Sentiment", "Count"]
                fig2 = px.pie(counts, names="Sentiment", values="Count", hole=0.5,
                              color_discrete_map={"Positive": "#22c55e", "Neutral": "#64748b", "Negative": "#ef4444"})
                st.plotly_chart(fig2, use_container_width=True)
