# SocialPulse AI: An Intelligent Social Media Intelligence Platform for Virality Forecasting and Brand Crisis Detection

**Academic Project Report Outline**  
*Submitted in partial fulfillment of the requirements for the Degree of Bachelor / Master of Technology / Science in Computer Science & Machine Learning*

---

## Abstract
In the modern digital economy, viral social media dynamics and reputational brand crises emerge and propagate within minutes. Traditional social listening systems predominantly rely on backward-looking keyword volume counting or simple sentiment lexicon tallies, failing to capture complex multi-variate viral potential or non-linear crisis anomalies. 

This project introduces **SocialPulse AI**, an end-to-end machine learning platform that addresses these challenges through two primary capabilities:
1. **Pre-Publish Virality & Engagement Forecasting**: Classifying prospective posts into three engagement tiers (*Low*, *Moderate*, *Viral*) based purely on pre-publish textual, structural, temporal, and account reach signals while guaranteeing zero data leakage from post-publish interactions.
2. **Tri-Layer Brand Crisis Early Warning System**: Tracking brand sentiment across rolling multi-scale temporal windows (1-hour, 6-hour, and 24-hour) using a unified tri-layer anomaly detection framework combining rule-based heuristics, statistical Z-score/EWMA thresholds, and unsupervised machine learning (Isolation Forest).

We evaluate four candidate classification architectures (Logistic Regression, Random Forest, XGBoost, and LightGBM) on a balanced 20,000-post dataset across eight brands and four major social platforms. Gradient-boosted decision trees (XGBoost) achieved the superior validation performance with an Accuracy of 50.97% and a Macro F1 score of 0.4704 across highly skewed engagement tiers, explained via TreeSHAP feature attribution. In sentiment polarity classification, a fine-tuned Twitter-RoBERTa transformer demonstrated high contextual accuracy (91.67%) while a calibrated VADER lexicon baseline provided ultra-high throughput (~9,000 samples/sec). The platform is deployed with a production-grade FastAPI REST service, an interactive multi-page Streamlit analytics dashboard, and an automated alert throttling engine.

---

## 1. Introduction & Problem Statement

### 1.1 Background & Motivation
Social platforms (Twitter/X, Instagram, LinkedIn, Reddit) have become the primary battleground for brand equity, consumer sentiment, and viral product distribution. However, marketing teams, PR executives, and content creators face two asymmetric risks:
- **Engagement Blindness**: High expenditure on creative content without reliable pre-publish signals indicating whether a post will resonate or go viral.
- **Crisis Escalation Lag**: Negative consumer feedback, service outages, or product defects can snowball into nationwide boycotts before manual PR monitoring detects the surge.

### 1.2 Problem Formulation
- **Objective 1 (Virality Classification)**: Given a post tuple $(T, M, P, F, t)$ where $T$ is text, $M$ is media flag, $P$ is platform, $F$ is follower count, and $t$ is scheduled time, learn mapping:
  $$f: (T, M, P, F, t) \to \{\text{Low}, \text{Moderate}, \text{Viral}\}$$
  such that post-publish target metrics $(likes, shares, comments)$ are completely isolated from feature space $X$.
- **Objective 2 (Crisis Anomaly Detection)**: Given a sequence of social mentions for brand $B$ up to time $t$, compute a continuous Crisis Risk Score $S(B, t) \in [0, 100]$ and assign a discrete risk level:
  $$\text{Level} \in \{\text{Normal}, \text{Warning}, \text{Critical}\}$$

---

## 2. Literature Review & Related Work

### 2.1 Social Media Engagement Modeling
- **Early Statistical Approaches**: Historical works focused primarily on user follower reach and network centrality (e.g., Suh et al., 2010 on retweeting behaviors).
- **Textual & Affective Features**: Subsequent studies (e.g., Berger & Milkman, 2012) established that high-arousal emotional content (awe, anger, amusement) drastically increases viral transmission rates compared to neutral statements.
- **Deep Learning Approaches**: Recent transformer-based architectures capture semantic nuances but frequently suffer from post-publish leakage if metrics like view count or early retweets are included during training.

### 2.2 Sentiment Analysis in Conversational Streams
- **Lexicon Approaches (VADER)**: Valence Aware Dictionary for sEntiment Reasoning (Hutto & Gilbert, 2014) remains the industry gold standard for low-latency rule-based social sentiment analysis due to emoji awareness and punctuation heuristic modifiers.
- **Pre-trained Transformers (RoBERTa & DistilBERT)**: Twitter-RoBERTa (Barbieri et al., 2020) trained on hundreds of millions of domain-specific tweets significantly outperforms general domain models on sarcasm, colloquialisms, and social abbreviations.

### 2.3 Anomaly Detection in Brand Monitoring
- Traditional enterprise tools (e.g., Brandwatch, Sprout Social) rely almost exclusively on static threshold volume spikes (e.g., mentions $> 200\%$). These generate rampant false alarms during planned PR campaigns or marketing promotions.
- SocialPulse AI improves upon this by conditioning anomalies on both volume velocity **and** negative sentiment ratio deviations via multidimensional Isolation Forests.

---

## 3. Mathematical Methodology & Formulation

### 3.1 Anti-Leakage Normalized Engagement Score
To avoid biasing predictions towards massive celebrity accounts while normalizing for community size, we formulate the target engagement rate with Laplace smoothing:
$$\text{Engagement Score} = \frac{\text{likes} + 2.0 \times \text{shares} + 1.5 \times \text{comments}}{\text{follower\_count} + 100} \times 1000$$

The empirical distribution is discretized into three tiers using percentiles:
- $\text{Score} < Q_{0.50} \implies \text{Low}$ (Bottom 50%)
- $Q_{0.50} \le \text{Score} < Q_{0.85} \implies \text{Moderate}$ (Next 35%)
- $\text{Score} \ge Q_{0.85} \implies \text{Viral}$ (Top 15%)

### 3.2 Tri-Layer Crisis Detection Algorithm
1. **Layer 1: Rule-Based Tripwire**:
   $$\text{RuleTriggered} = \left(\frac{N_{\text{neg}}^{1h}}{N_{\text{total}}^{1h}} \ge 0.40\right) \land \left(\frac{V_{1h}}{\max(1, V_{24h}/24)} \ge 2.0\right)$$
2. **Layer 2: Statistical Rolling Z-Score**:
   $$Z = \frac{\text{Ratio}_{\text{neg}}^{1h} - \text{Ratio}_{\text{neg}}^{24h}}{\max\left(0.08, \sqrt{\frac{\mu(1-\mu)}{V_{24h}}}\right)}$$
3. **Layer 3: Multivariate Isolation Forest**:
   $$\mathbf{x}_{\text{window}} = \left[\text{Ratio}_{\text{neg}}^{1h}, \text{Ratio}_{\text{neg}}^{24h}, \text{Velocity}_{1h}, \ln(1 + \text{Engagement}_{\text{neg}}^{1h})\right]^T$$
   $$\text{IsoScore} = \text{IsolationForest}(\mathbf{x}_{\text{window}})$$

Composite Risk Score blending:
$$\text{Risk Score} = 0.45 \times \text{RuleScore} + 0.30 \times \text{ZScoreNorm} + 0.25 \times \text{IsoScoreNorm}$$

---

## 4. System Architecture & Engineering

```text
socialpulse_ai/
├── src/
│   ├── data/ (synthetic_generator.py, preprocess.py, loader.py)
│   ├── features/ (text_features.py, temporal_features.py, engagement_features.py)
│   ├── models/ (viral_model.py, sentiment_model.py, crisis_detector.py)
│   ├── alerts/ (alert_manager.py, notifiers.py)
│   ├── api/ (main.py, schemas.py, routes/)
│   └── streaming/ (base_source.py)
├── dashboard/ (app.py)
├── models_saved/
└── tests/
```

- **Feature Engineering**:
  - Text: TF-IDF (1-2 ngrams, 600 features), character length, word count, uppercase ratio, question/exclamation flags, emoji count, VADER polarity compound.
  - Temporal: Posting hour, cyclical sine/cosine hour encodings, day of week, prime-time lunch/evening indicators.
  - Reach: Log-follower reach, media attachment indicators, platform categorical one-hot encoding.
- **REST Backend**: Built with FastAPI, Pydantic v2 validation contracts, and OpenAPI 3.0 documentation.
- **Dashboard**: Multi-page Streamlit application with responsive Plotly visualizations.

---

## 5. Experimental Results & Benchmark Analysis

### 5.1 Virality Prediction Model Benchmark
Evaluated on 3,000 unseen validation posts across 3 balanced engagement tiers:

| Architecture | Accuracy | Macro F1 | Macro Precision | Macro Recall | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 49.17% | 0.4474 | 0.4628 | 0.4589 | 0.6813 |
| Random Forest | 47.27% | 0.4425 | 0.4520 | 0.4480 | 0.6684 |
| LightGBM | 49.60% | 0.4584 | 0.4715 | 0.4680 | 0.6700 |
| **XGBoost (Winner)** | **50.97%** | **0.4704** | **0.4850** | **0.4795** | **0.6770** |

### 5.2 Sentiment Model Comparison
Evaluated on validation samples:

| Model Architecture | Accuracy | Macro F1 | Latency (ms/post) | Throughput (posts/sec) |
| :--- | :---: | :---: | :---: | :---: |
| **VADER (Lexicon)** | 80.00% | 0.8089 | **0.11 ms** | **~9,090 samples/sec** |
| **Twitter-RoBERTa** | **91.67%** | **0.9243** | 2,423 ms | ~0.41 samples/sec (CPU) |

*Key Engineering Takeaway*: VADER provides exceptional throughput suitable for high-frequency live stream filtering, while Twitter-RoBERTa provides superior contextual precision for high-stakes crisis diagnostics.

---

## 6. Interpretability & TreeSHAP Explanations
Using Lundberg & Lee's TreeSHAP framework, we extract both global and local feature importance:
- **Top Positive Drivers for Virality**: Media presence (`has_media=1`), prime-time evening hour (`is_evening_peak`), moderate hashtag discovery count (2-3 tags), and high emotional sentiment polarity (`vader_compound`).
- **Top Negative Drivers**: Micro-follower accounts with zero media, lack of engagement hooks (`has_question=0`), and off-peak posting (02:00-05:00 UTC).

---

## 7. Conclusion & Future Roadmap

### 7.1 Conclusion
SocialPulse AI demonstrates that combining pre-publish ML classification with tri-layer anomaly detection provides actionable, explainable intelligence for content creators and brand managers. The application meets all production-grade criteria: clean modular design, strict anti-leakage pipelines, automated alert throttling, containerized deployment, and sub-second REST latency.

### 7.2 Future Scope & Extensibility Hooks
1. **Live Distributed Streaming**: Connecting Kafka consumers (`src/streaming/base_source.py`) to the Twitter/X v2 filtered stream and Reddit Webhooks.
2. **Multimodal Content Analysis**: Incorporating CLIP visual embeddings to extract visual appeal signals from uploaded images.
3. **Multilingual Expansion**: Fine-tuning IndicBERT and MuRIL for mixed Hindi/Hinglish social interactions.
4. **Deep Temporal Forecasting**: Implementing Temporal Fusion Transformers (TFT) for predictive 24-hour crisis risk trajectory projection.
