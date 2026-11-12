"""
SocialPulse AI — Production-Grade Analytics Dashboard.
Interactive multi-page Streamlit application providing real-time social intelligence,
virality predictions with explainability, brand crisis anomaly tracking, and model benchmarks.
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.alerts.alert_manager import alert_manager
from src.data.loader import DataLoader
from src.models.crisis_detector import load_crisis_detector
from src.models.sentiment_model import predict_sentiment
from src.models.viral_model import ViralityPredictor
from src.utils.config import settings
from src.utils.logger import logger


# ==============================================================================
# AUTO-SETUP: Generate data & train models on first run (Streamlit Cloud safe)
# ==============================================================================
def _auto_setup():
    """Auto-generate data and train models if not present (cloud deployment)."""
    raw_path = PROJECT_ROOT / "data" / "raw" / "posts_dataset.csv"
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    viral_model_path = PROJECT_ROOT / "models_saved" / "best_viral_model.joblib"
    crisis_model_path = PROJECT_ROOT / "models_saved" / "crisis_detector.joblib"

    needs_setup = not raw_path.exists() or not viral_model_path.exists()
    if not needs_setup:
        return  # All good, skip

    st.info("⚙️ **First-run setup detected.** Generating data and training models — this takes ~2–3 minutes on first launch. Grab a coffee ☕")
    progress = st.progress(0, text="Starting setup…")

    try:
        # Step 1: Generate synthetic dataset
        if not raw_path.exists():
            progress.progress(10, text="📊 Generating 20,000 synthetic social media posts…")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            from src.data.synthetic_generator import generate_social_dataset
            generate_social_dataset()

        # Step 2: Preprocess & split
        if not train_path.exists():
            progress.progress(30, text="🔧 Preprocessing data and creating train/val/test splits…")
            train_path.parent.mkdir(parents=True, exist_ok=True)
            from src.data.preprocess import run_pipeline
            run_pipeline()

        # Step 3: Train viral model
        if not viral_model_path.exists():
            progress.progress(50, text="🤖 Training viral engagement prediction model (XGBoost + LightGBM)…")
            viral_model_path.parent.mkdir(parents=True, exist_ok=True)
            from src.models.train_viral import run_training
            run_training()

        # Step 4: Train crisis detector
        if not crisis_model_path.exists():
            progress.progress(80, text="🚨 Training crisis detection model (Isolation Forest)…")
            from src.models.train_crisis import run_crisis_training
            run_crisis_training()

        progress.progress(100, text="✅ Setup complete! Loading dashboard…")
        import time; time.sleep(1)
        progress.empty()
        st.success("✅ Setup complete! Dashboard is ready.")
        st.rerun()

    except Exception as e:
        progress.empty()
        st.error(f"❌ Setup failed: {e}\n\nPlease check the app logs for details.")
        st.stop()


# Page Configuration
st.set_page_config(
    page_title="SocialPulse AI — Social Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-setup on first run (generates data + trains models if missing)
_auto_setup()

# Custom Styling (Dark/Light friendly, modern cards, badge styling)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EDF2F7 100%);
        padding: 1.1rem 1.4rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-normal {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_dashboard_data():
    """Load cached dataset for dashboard analysis."""
    try:
        loader = DataLoader()
        raw_df = loader.load_raw()
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        return raw_df
    except Exception as e:
        logger.error(f"Error loading dashboard data: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_models():
    """Load serialized prediction models."""
    predictor = None
    crisis_detector = None
    try:
        predictor = ViralityPredictor()
    except Exception as e:
        logger.warning(f"Virality predictor unavailable: {e}")
    try:
        crisis_detector = load_crisis_detector()
    except Exception as e:
        logger.warning(f"Crisis detector unavailable: {e}")
    return predictor, crisis_detector


# Load Core State
df_raw = load_dashboard_data()
predictor, crisis_detector = load_models()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/pulse.png", width=64)
st.sidebar.title("SocialPulse AI")
st.sidebar.caption("v1.0.0 | Production-Style ML")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Overview",
        "🔮 Post Virality Predictor",
        "💬 Sentiment Explorer",
        "🚨 Crisis Anomaly Monitor",
        "📈 Model Performance Studio",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.info(
    "💡 **SocialPulse AI** delivers real-time virality forecasting, sentiment signals, "
    "and tri-layer brand crisis tripwires."
)


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if nav_choice == "📊 Executive Overview":
    st.markdown('<div class="main-title">Executive Intelligence Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time health, engagement distribution, and sentiment across monitored social feeds.</div>', unsafe_allow_html=True)

    if df_raw.empty:
        st.warning("No data found. Please run synthetic generation first.")
        st.stop()

    # KPI Metrics
    total_posts = len(df_raw)
    viral_cutoff = df_raw["shares"].quantile(0.85)
    viral_count = (df_raw["shares"] >= viral_cutoff).sum()
    pos_pct = (df_raw["sentiment_label"] == "Positive").mean() * 100
    recent_alerts = alert_manager.get_recent_alerts(limit=5)
    active_alerts = len(recent_alerts)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Monitored Posts</div>
            <div class="metric-val">{total_posts:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Positive Sentiment Index</div>
            <div class="metric-val">{pos_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Viral Breakouts</div>
            <div class="metric-val">{viral_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Recent Active Alerts</div>
            <div class="metric-val">{active_alerts}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Platform Volume Share")
        platform_counts = df_raw["platform"].value_counts().reset_index()
        platform_counts.columns = ["Platform", "Count"]
        fig_donut = px.pie(
            platform_counts,
            names="Platform",
            values="Count",
            hole=0.45,
            color_discrete_sequence=["#3498db", "#9b59b6", "#2ecc71", "#e67e22"],
        )
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.subheader("Sentiment Distribution by Platform")
        plat_sent = df_raw.groupby(["platform", "sentiment_label"]).size().unstack().fillna(0).reset_index()
        fig_bar = px.bar(
            plat_sent,
            x="platform",
            y=["Positive", "Neutral", "Negative"],
            barmode="stack",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, legend_title="Sentiment")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Recent Social Media Live Feed")
    feed_col1, feed_col2 = st.columns([1, 3])
    with feed_col1:
        brand_filter = st.selectbox("Filter Brand", ["All Brands"] + list(df_raw["brand"].unique()))
    with feed_col2:
        sentiment_filter = st.selectbox("Filter Sentiment", ["All Sentiments", "Positive", "Neutral", "Negative"])

    filtered_df = df_raw.copy()
    if brand_filter != "All Brands":
        filtered_df = filtered_df[filtered_df["brand"] == brand_filter]
    if sentiment_filter != "All Sentiments":
        filtered_df = filtered_df[filtered_df["sentiment_label"] == sentiment_filter]

    st.dataframe(
        filtered_df[["timestamp", "platform", "brand", "text", "sentiment_label", "likes", "shares", "comments"]].head(15),
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================================
# PAGE 2: POST VIRALITY PREDICTOR
# ==============================================================================
elif nav_choice == "🔮 Post Virality Predictor":
    st.markdown('<div class="main-title">Pre-Publish Virality & Engagement Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Compose post drafts, test scheduling, and forecast engagement before publishing.</div>', unsafe_allow_html=True)

    if predictor is None:
        st.error("Viral model not initialized. Please train the model in Phase 2.")
        st.stop()

    p_col1, p_col2 = st.columns([1.3, 1])

    with p_col1:
        st.subheader("Post Draft Studio")
        draft_text = st.text_area(
            "Draft Content",
            value="We just released our new open-source multimodal agent framework! Check it out: https://pulse.io #AI #OpenSource #Tech 🚀",
            height=130,
        )

        sub_col1, sub_col2, sub_col3 = st.columns(3)
        with sub_col1:
            platform_choice = st.selectbox("Target Platform", ["Twitter", "Instagram", "LinkedIn", "Reddit"])
        with sub_col2:
            followers_input = st.number_input("Account Followers", min_value=10, max_value=5000000, value=25000, step=500)
        with sub_col3:
            has_media_input = st.checkbox("Media Attached (Image/Video)", value=True)

        sched_hour = st.slider("Scheduled Hour (UTC)", min_value=0, max_value=23, value=14)
        predict_btn = st.button("🔮 Predict Engagement Tier", type="primary", use_container_width=True)

    with p_col2:
        st.subheader("Virality Forecast")
        if predict_btn or draft_text:
            sched_time_str = f"2026-10-20 {sched_hour:02d}:30:00"
            payload = {
                "text": draft_text,
                "platform": platform_choice,
                "follower_count": int(followers_input),
                "has_media": int(has_media_input),
                "timestamp": sched_time_str,
            }
            res = predictor.predict(payload)
            pred_tier = res["prediction"]
            conf = res["confidence"]
            probas = res["probabilities"]

            # Color determination
            color_map = {"Low": "#3498db", "Moderate": "#2ecc71", "Viral": "#e74c3c"}
            tier_color = color_map.get(pred_tier, "#3498db")

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf * 100.0,
                number={"suffix": "%", "font": {"size": 28, "color": tier_color}},
                title={"text": f"Predicted Tier: <b>{pred_tier.upper()}</b>", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": tier_color},
                    "steps": [
                        {"range": [0, 50], "color": "#F1F5F9"},
                        {"range": [50, 80], "color": "#E2E8F0"},
                        {"range": [80, 100], "color": "#CBD5E1"},
                    ],
                }
            ))
            fig_gauge.update_layout(height=240, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Probability Breakdown
            df_prob = pd.DataFrame({
                "Tier": list(probas.keys()),
                "Probability": list(probas.values())
            })
            fig_p = px.bar(
                df_prob,
                x="Tier",
                y="Probability",
                text="Probability",
                color="Tier",
                color_discrete_map=color_map,
            )
            fig_p.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_p.update_layout(yaxis_range=[0, 1.0], height=180, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)

    st.divider()
    tip_col1, tip_col2 = st.columns([1, 1])

    with tip_col1:
        st.subheader("Top Contributing Signals")
        contributions = res.get("top_contributing_features", [])
        if contributions:
            df_c = pd.DataFrame(contributions)
            fig_imp = px.bar(
                df_c,
                x="impact",
                y="feature",
                orientation="h",
                color="direction",
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"},
                title="Feature Impact (SHAP Attribution)",
            )
            fig_imp.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance data ready.")

    with tip_col2:
        st.subheader("💡 Optimization Recommendations")
        suggestions = res.get("suggestions", [])
        for s in suggestions:
            st.markdown(f"- 🚀 {s}")


# ==============================================================================
# PAGE 3: SENTIMENT EXPLORER
# ==============================================================================
elif nav_choice == "💬 Sentiment Explorer":
    st.markdown('<div class="main-title">Sentiment & Conversational Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Longitudinal polarity trajectory, brand breakdowns, and high-engagement post spotlight.</div>', unsafe_allow_html=True)

    if df_raw.empty:
        st.warning("No dataset loaded.")
        st.stop()

    # Time series of sentiment
    df_raw["date"] = df_raw["timestamp"].dt.date
    daily_sentiment = df_raw.groupby(["date", "sentiment_label"]).size().unstack().fillna(0).reset_index()

    st.subheader("Daily Sentiment Polarity Timeline")
    fig_time = px.line(
        daily_sentiment,
        x="date",
        y=["Positive", "Neutral", "Negative"],
        color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
        markers=True,
    )
    fig_time.update_layout(height=340, xaxis_title="Date", yaxis_title="Daily Posts", margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_time, use_container_width=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Brand Sentiment Distribution")
        brand_sent = df_raw.groupby(["brand", "sentiment_label"]).size().unstack().fillna(0)
        brand_sent_pct = brand_sent.div(brand_sent.sum(axis=1), axis=0) * 100.0
        fig_brand = px.bar(
            brand_sent_pct.reset_index(),
            x="brand",
            y=["Positive", "Neutral", "Negative"],
            barmode="stack",
            color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
        )
        fig_brand.update_layout(height=320, yaxis_title="% Share", margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_brand, use_container_width=True)

    with col_s2:
        st.subheader("Quick Text Polarity Inspector")
        custom_text = st.text_input("Enter any sentence or comment to analyze live:", "The new customer update solved everything!")
        if custom_text:
            v_res = predict_sentiment(custom_text, engine="vader")
            lbl = v_res["label"]
            conf = v_res["confidence"]
            scores = v_res["scores"]
            badge_class = "badge-normal" if lbl == "Positive" else ("badge-critical" if lbl == "Negative" else "badge-warning")
            st.markdown(f"**Predicted Polarity:** <span class='{badge_class}'>{lbl} ({conf*100:.1f}%)</span>", unsafe_allow_html=True)
            st.json(scores)

    st.divider()
    st.subheader("Post Spotlight: High Engagement Outliers")
    sp_col1, sp_col2 = st.columns(2)
    with sp_col1:
        st.markdown("##### 🌟 Top Viral Positive Post")
        top_pos = df_raw[df_raw["sentiment_label"] == "Positive"].sort_values("shares", ascending=False).iloc[0]
        st.info(f"**@{top_pos['brand']}** ({top_pos['platform']})\n\n\"{top_pos['text']}\"\n\n❤️ {top_pos['likes']:,} Likes | 🔁 {top_pos['shares']:,} Shares | 💬 {top_pos['comments']:,} Comments")
    with sp_col2:
        st.markdown("##### ⚠️ Top Viral Negative Post (Controversy Spike)")
        top_neg = df_raw[df_raw["sentiment_label"] == "Negative"].sort_values("shares", ascending=False).iloc[0]
        st.error(f"**@{top_neg['brand']}** ({top_neg['platform']})\n\n\"{top_neg['text']}\"\n\n❤️ {top_neg['likes']:,} Likes | 🔁 {top_neg['shares']:,} Shares | 💬 {top_neg['comments']:,} Comments")


# ==============================================================================
# PAGE 4: CRISIS ANOMALY MONITOR
# ==============================================================================
elif nav_choice == "🚨 Crisis Anomaly Monitor":
    st.markdown('<div class="main-title">Brand Crisis & Backlash Early Warning Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-window rolling metrics, statistical Z-score anomalies, Isolation Forest signals, and keyword diagnostics.</div>', unsafe_allow_html=True)

    if crisis_detector is None:
        st.error("Crisis detector not trained. Please train the model in Phase 4.")
        st.stop()

    brands = list(df_raw["brand"].unique()) if not df_raw.empty else ["TechCorp", "SwiftAir", "NovaPay"]
    c_brand = st.selectbox("Select Monitored Brand", brands, index=2 if "SwiftAir" in brands else 0)

    brand_df = df_raw[df_raw["brand"] == c_brand].sort_values("timestamp")
    latest_time = brand_df["timestamp"].max()
    analysis = crisis_detector.analyze_brand(c_brand, brand_df, latest_time)

    risk_score = analysis["crisis_risk_score"]
    risk_level = analysis["risk_level"]
    signals = analysis["signals"]
    metrics = analysis["metrics"]
    diag_kw = analysis["diagnostic_keywords"]

    # Top Gauges & Status
    g_col1, g_col2, g_col3 = st.columns([1.2, 1, 1])

    with g_col1:
        st.subheader("Live Crisis Risk Meter")
        gauge_color = "#e74c3c" if risk_level == "Critical" else ("#f39c12" if risk_level == "Warning" else "#2ecc71")
        fig_c_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "/100", "font": {"size": 28, "color": gauge_color}},
            title={"text": f"Status: <b>{risk_level.upper()}</b>", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 40], "color": "#D1FAE5"},
                    {"range": [40, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 70},
            }
        ))
        fig_c_gauge.update_layout(height=240, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_c_gauge, use_container_width=True)

    with g_col2:
        st.subheader("Active Tri-Layer Tripwires")
        r_trip = signals["rule_tripwire_triggered"]
        z_trip = signals["statistical_anomaly_triggered"]
        iso_trip = signals["isolation_forest_anomaly_triggered"]

        st.markdown(f"**Layer 1 (Rule-Based Thresholds):** {'🚨 TRIGGERED' if r_trip else '✅ Normal'}")
        st.markdown(f"**Layer 2 (Statistical Z-Score):** {'🚨 ANOMALY' if z_trip else '✅ Normal'} ({signals['z_score']}σ)")
        st.markdown(f"**Layer 3 (Isolation Forest ML):** {'🚨 OUTLIER' if iso_trip else '✅ In-Distribution'}")
        st.markdown(f"**Hourly Volume Velocity:** `{signals['volume_velocity_multiplier']:.1f}x` baseline")
        st.markdown(f"**1-Hour Negative Ratio:** `{signals['negative_ratio_1h']*100:.1f}%`")

    with g_col3:
        st.subheader("Diagnostic Root Cause")
        st.caption("Dominant complaint keywords driving negative sentiment surge:")
        for kw in diag_kw:
            st.markdown(f"- 🔴 **{kw}**")

        # Interactive Simulation Trigger Button
        st.write("")
        if st.button(f"⚡ Inject Test Crisis on @{c_brand}", type="secondary", use_container_width=True):
            from scripts.simulate_crisis import simulate_crisis
            sim_res = simulate_crisis(brand=c_brand, num_posts=50)
            st.success(f"Crisis successfully injected! Risk score jumped to {sim_res['crisis']['crisis_risk_score']}/100 ({sim_res['crisis']['risk_level']})")
            st.rerun()

    st.divider()
    st.subheader("Recent Alert Audit Log (SQLite)")
    recent_db_alerts = alert_manager.get_recent_alerts(limit=10, brand=c_brand)
    if recent_db_alerts:
        df_al = pd.DataFrame(recent_db_alerts)
        st.dataframe(df_al[["timestamp", "brand", "severity", "crisis_score", "message", "keywords"]], use_container_width=True, hide_index=True)
    else:
        st.info("No active alerts logged for this brand.")


# ==============================================================================
# PAGE 5: MODEL PERFORMANCE STUDIO
# ==============================================================================
elif nav_choice == "📈 Model Performance Studio":
    st.markdown('<div class="main-title">Machine Learning Performance Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rigorous academic benchmarks, ROC curves, confusion matrices, and explainability artifacts.</div>', unsafe_allow_html=True)

    viral_meta_path = settings.MODELS_DIR / "viral_metadata.json"
    sent_meta_path = settings.MODELS_DIR / "sentiment_benchmark.json"

    # Virality Benchmark Table
    st.subheader("1. Engagement Prediction Benchmark (4-Model Comparison)")
    if viral_meta_path.exists():
        with open(viral_meta_path, "r", encoding="utf-8") as f:
            v_meta = json.load(f)

        metrics_dict = v_meta.get("metrics", {})
        df_bench = pd.DataFrame.from_dict(metrics_dict, orient="index").reset_index()
        df_bench.rename(columns={"index": "Model Architecture"}, inplace=True)
        st.dataframe(
            df_bench[["Model Architecture", "accuracy", "macro_f1", "macro_precision", "macro_recall", "roc_auc_ovr"]],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"🏆 Winning Model Architecture: **{v_meta.get('best_model', 'XGBoost')}**")

        # Confusion Matrix
        best_name = v_meta.get("best_model", "XGBoost")
        best_cm = np.array(metrics_dict[best_name]["confusion_matrix"])
        fig_cm = px.imshow(
            best_cm,
            text_auto=True,
            x=settings.ENGAGEMENT_CLASSES,
            y=settings.ENGAGEMENT_CLASSES,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="True Ground Truth Class", color="Post Count"),
            title=f"Confusion Matrix Heatmap ({best_name})",
        )
        fig_cm.update_layout(height=380, width=500)
        st.plotly_chart(fig_cm, use_container_width=False)
    else:
        st.info("Viral model benchmark data will appear after training.")

    st.divider()

    # Sentiment Benchmark Table
    st.subheader("2. Sentiment Engine Benchmark: VADER vs Contextual Transformer")
    if sent_meta_path.exists():
        with open(sent_meta_path, "r", encoding="utf-8") as f:
            s_meta = json.load(f)
        df_s_bench = pd.DataFrame.from_dict(s_meta, orient="index").reset_index()
        df_s_bench.rename(columns={"index": "Engine"}, inplace=True)
        st.dataframe(
            df_s_bench[["Engine", "accuracy", "macro_f1", "latency_ms", "throughput_samples_per_sec"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Sentiment benchmark data will appear after running benchmark.")
