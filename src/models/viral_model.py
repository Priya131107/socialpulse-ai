"""
Virality & Engagement Prediction Modeling Engine for SocialPulse AI.
Trains, evaluates, tunes, and explains multi-class engagement models
(Logistic Regression, Random Forest, XGBoost, LightGBM) with TreeSHAP explainability.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.engagement_features import EngagementFeatureExtractor
from src.features.temporal_features import TemporalFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.utils.config import settings
from src.utils.logger import logger


class CompositeFeaturePipeline(BaseEstimator):
    """Combines text, temporal, and engagement feature extractors."""

    def __init__(self, max_tfidf_features: int = 500):
        self.max_tfidf_features = max_tfidf_features
        self.text_extractor = TextFeatureExtractor(max_tfidf_features=max_tfidf_features)
        self.temporal_extractor = TemporalFeatureExtractor()
        self.engagement_extractor = EngagementFeatureExtractor()
        self.scaler = StandardScaler()
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.text_extractor.fit(X, y)
        self.temporal_extractor.fit(X, y)
        self.engagement_extractor.fit(X, y)

        f_text = self.text_extractor.transform(X)
        f_temp = self.temporal_extractor.transform(X)
        f_eng = self.engagement_extractor.transform(X)

        combined = np.hstack([f_text, f_temp, f_eng])
        self.scaler.fit(combined)

        self.feature_names_ = (
            self.text_extractor.get_feature_names_out()
            + self.temporal_extractor.get_feature_names_out()
            + self.engagement_extractor.get_feature_names_out()
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        f_text = self.text_extractor.transform(X)
        f_temp = self.temporal_extractor.transform(X)
        f_eng = self.engagement_extractor.transform(X)
        combined = np.hstack([f_text, f_temp, f_eng])
        return self.scaler.transform(combined)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_


class ViralityModelTrainer:
    """Trains and compares multi-class engagement classifiers."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.class_mapping = settings.CLASS_MAPPING
        self.inverse_mapping = settings.INVERSE_CLASS_MAPPING
        self.target_classes = settings.ENGAGEMENT_CLASSES

    def train_and_benchmark(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        max_tfidf: int = 500,
    ) -> Dict[str, Any]:
        """
        Train and benchmark Logistic Regression, Random Forest, XGBoost, and LightGBM.
        """
        logger.info("Fitting composite feature extraction pipeline...")
        feature_pipeline = CompositeFeaturePipeline(max_tfidf_features=max_tfidf)
        feature_pipeline.fit(train_df)

        X_train = feature_pipeline.transform(train_df)
        y_train = train_df["engagement_label"].map(self.inverse_mapping).values

        X_val = feature_pipeline.transform(val_df)
        y_val = val_df["engagement_label"].map(self.inverse_mapping).values

        # Compute balanced class weights for XGBoost
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        weights_map = {cls: total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
        sample_weights = np.array([weights_map[y] for y in y_train])

        candidate_models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_seed,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight="balanced",
                random_state=self.random_seed,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                objective="multi:softprob",
                num_class=3,
                random_state=self.random_seed,
                eval_metric="mlogloss",
                n_jobs=-1,
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                class_weight="balanced",
                random_state=self.random_seed,
                verbose=-1,
                n_jobs=-1,
            ),
        }

        results: Dict[str, Any] = {}
        best_model_name = None
        best_macro_f1 = -1.0
        best_clf = None

        logger.info("Evaluating candidate models on validation split...")
        for name, clf in candidate_models.items():
            if name == "XGBoost":
                clf.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                clf.fit(X_train, y_train)

            y_pred = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)

            acc = accuracy_score(y_val, y_pred)
            macro_f1 = f1_score(y_val, y_pred, average="macro")
            macro_prec = precision_score(y_val, y_pred, average="macro")
            macro_rec = recall_score(y_val, y_pred, average="macro")
            roc_auc = roc_auc_score(y_val, y_proba, multi_class="ovr")

            cm = confusion_matrix(y_val, y_pred).tolist()

            results[name] = {
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "macro_precision": round(float(macro_prec), 4),
                "macro_recall": round(float(macro_rec), 4),
                "roc_auc_ovr": round(float(roc_auc), 4),
                "confusion_matrix": cm,
            }

            logger.info(
                f"Model [{name:19s}] -> Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}"
            )

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_model_name = name
                best_clf = clf

        logger.info(f"Top performing model: {best_model_name} with Macro F1: {best_macro_f1:.4f}")

        # Compute SHAP explainer on validation sample for best tree model
        shap_explainer = None
        shap_feature_importance = {}
        try:
            if best_model_name in ["XGBoost", "LightGBM", "Random Forest"]:
                logger.info("Computing TreeSHAP explainer values...")
                explainer = shap.TreeExplainer(best_clf)
                shap_sample = X_val[:300]
                shap_values = explainer.shap_values(shap_sample)

                # Format feature importances
                if isinstance(shap_values, list):
                    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                elif len(shap_values.shape) == 3:
                    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
                else:
                    mean_abs = np.abs(shap_values).mean(axis=0)

                top_indices = np.argsort(mean_abs)[::-1][:20]
                feature_names = feature_pipeline.get_feature_names_out()
                shap_feature_importance = {
                    feature_names[i]: round(float(mean_abs[i]), 5)
                    for i in top_indices
                }
                shap_explainer = explainer
        except Exception as e:
            logger.warning(f"SHAP explanation computation notice: {e}")

        # Build full production pipeline container
        pipeline_bundle = {
            "feature_pipeline": feature_pipeline,
            "classifier": best_clf,
            "model_name": best_model_name,
            "feature_names": feature_pipeline.get_feature_names_out(),
            "target_classes": self.target_classes,
            "class_mapping": self.class_mapping,
            "inverse_mapping": self.inverse_mapping,
            "metrics": results,
            "shap_top_features": shap_feature_importance,
        }

        # Save model
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline_bundle, settings.VIRAL_MODEL_PATH)
        logger.info(f"Serialized best model pipeline bundle to {settings.VIRAL_MODEL_PATH}")

        # Save metadata JSON
        metadata_path = settings.MODELS_DIR / "viral_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_model": best_model_name,
                    "metrics": results,
                    "top_shap_features": shap_feature_importance,
                },
                f,
                indent=2,
            )

        return pipeline_bundle


class ViralityPredictor:
    """Inference engine for predicting engagement tiers with actionable recommendations."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.VIRAL_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {self.model_path}. Train the model first."
            )
        self.bundle = joblib.load(self.model_path)
        self.feature_pipeline: CompositeFeaturePipeline = self.bundle["feature_pipeline"]
        self.classifier = self.bundle["classifier"]
        self.model_name = self.bundle["model_name"]
        self.feature_names = self.bundle["feature_names"]
        self.target_classes = self.bundle["target_classes"]
        self.class_mapping = self.bundle["class_mapping"]

    def predict(self, post_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict engagement tier for a single post payload.
        Expected keys: text, platform, follower_count, has_media, timestamp (optional).
        """
        # Build single-row DataFrame
        df_row = pd.DataFrame([{
            "text": post_payload.get("text", ""),
            "cleaned_text": post_payload.get("text", "").lower(),
            "platform": post_payload.get("platform", "Twitter"),
            "follower_count": post_payload.get("follower_count", 5000),
            "has_media": int(post_payload.get("has_media", 0)),
            "timestamp": post_payload.get("timestamp", "2026-10-15 14:00:00"),
        }])

        X_feat = self.feature_pipeline.transform(df_row)
        probas = self.classifier.predict_proba(X_feat)[0]
        pred_idx = int(np.argmax(probas))
        predicted_tier = self.class_mapping[pred_idx]

        probabilities = {
            self.class_mapping[i]: round(float(p), 4)
            for i, p in enumerate(probas)
        }

        # Local feature contributions (SHAP approximation via feature weights / trees)
        contributions = self._explain_local_instance(X_feat[0])

        # Generate rule-based optimization tips
        suggestions = self._generate_suggestions(post_payload, predicted_tier)

        return {
            "prediction": predicted_tier,
            "probabilities": probabilities,
            "confidence": round(float(np.max(probas)), 4),
            "top_contributing_features": contributions,
            "suggestions": suggestions,
            "model_used": self.model_name,
        }

    def _explain_local_instance(self, x_vec: np.ndarray) -> List[Dict[str, Any]]:
        """Identify top contributing features for this prediction."""
        try:
            if hasattr(self.classifier, "feature_importances_"):
                imp = self.classifier.feature_importances_
                impact = x_vec * imp
                top_idx = np.argsort(np.abs(impact))[::-1][:6]
                return [
                    {
                        "feature": self.feature_names[i],
                        "impact": round(float(impact[i]), 4),
                        "direction": "positive" if impact[i] > 0 else "negative",
                    }
                    for i in top_idx
                ]
        except Exception:
            pass
        return []

    def _generate_suggestions(
        self,
        payload: Dict[str, Any],
        predicted_tier: str,
    ) -> List[str]:
        """Generate tactical AI advice to elevate post engagement."""
        suggestions = []
        text = payload.get("text", "")
        has_media = bool(payload.get("has_media", 0))
        hashtags = [w for w in text.split() if w.startswith("#")]
        
        # Check media presence
        if not has_media:
            suggestions.append("Attach an image, infographic, or video clip — visual posts receive ~1.6x higher engagement.")

        # Check hashtag usage
        if len(hashtags) < 2:
            suggestions.append("Include 2-3 targeted niche hashtags (#Tech, #AI, etc.) to enhance discovery beyond existing followers.")
        elif len(hashtags) > 5:
            suggestions.append("Reduce hashtags to 3 or fewer; excessive tags reduce perceived readability.")

        # Check engagement hooks (question/call-to-action)
        if "?" not in text:
            suggestions.append("Add a question or open-ended hook at the end of the post to invite replies and community discussion.")

        # Check length
        if len(text) < 50:
            suggestions.append("Provide a little more context or a concrete insight to encourage shares and bookmarks.")

        if not suggestions:
            suggestions.append("Great post structure! Consider scheduling during peak lunch (12:00-14:00) or evening (18:00-21:00) hours.")

        return suggestions

