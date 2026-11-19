"""
SocialPulse AI — Simple Viral Engagement Model.
Uses Logistic Regression with TF-IDF + numeric features.
No XGBoost, no SHAP, no hyperparameter tuning.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

from src.utils.config import settings
from src.utils.logger import logger


class ViralityPredictor:
    """Simple virality predictor using Logistic Regression + TF-IDF."""

    def __init__(self):
        self.model_path = settings.VIRAL_MODEL_PATH
        self.tfidf = None
        self.clf   = None
        self._load()

    def _load(self):
        if self.model_path.exists():
            bundle = joblib.load(self.model_path)
            self.tfidf = bundle["tfidf"]
            self.clf   = bundle["clf"]
            logger.info("Viral model loaded.")
        else:
            logger.warning("Viral model not found — run train_viral.py first.")

    def _numeric_features(self, row: dict) -> np.ndarray:
        return np.array([[
            float(row.get("hashtags_count", 0)),
            float(row.get("mentions_count", 0)),
            float(row.get("has_media", 0)),
            np.log1p(float(row.get("follower_count", 1000))),
        ]])

    def predict(self, post: dict) -> dict:
        """Predict engagement label for a single post dict."""
        if self.tfidf is None or self.clf is None:
            return {"prediction": "Moderate", "probabilities": {}, "error": "Model not loaded"}

        text = str(post.get("text", ""))
        X_text = self.tfidf.transform([text])
        X_num  = csr_matrix(self._numeric_features(post))
        X      = hstack([X_text, X_num])

        label = self.clf.predict(X)[0]
        proba = self.clf.predict_proba(X)[0]
        classes = self.clf.classes_

        return {
            "prediction":    label,
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        }


def run_training():
    """Train and save the virality model."""
    from src.utils.config import settings

    train_path = settings.TRAIN_DATA_PATH
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}. Run preprocess first.")
        return

    df = pd.read_csv(train_path)
    df["clean_text"] = df["text"].fillna("").astype(str)

    X_text = df["clean_text"]
    X_num  = df[["hashtags_count", "mentions_count", "has_media", "follower_count"]].fillna(0).copy()
    X_num["follower_count"] = np.log1p(X_num["follower_count"])
    y = df["engagement_label"]

    # Train/test split
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build TF-IDF
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train = hstack([tfidf.fit_transform(X_text_train), csr_matrix(X_num_train.values)])
    X_test  = hstack([tfidf.transform(X_text_test),      csr_matrix(X_num_test.values)])

    # Train Logistic Regression
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")
    logger.info(f"Viral model — Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Save
    settings.VIRAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tfidf": tfidf, "clf": clf}, settings.VIRAL_MODEL_PATH)
    logger.info(f"Model saved → {settings.VIRAL_MODEL_PATH}")


if __name__ == "__main__":
    run_training()
