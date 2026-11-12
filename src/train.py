"""
Model Training & Hyperparameter Tuning module for SocialPulse AI.
Implements:
1. Stratified 70/15/15 split.
2. Handling class imbalance using balanced class/sample weights.
3. Training & hyperparameter tuning for 3 models:
   - Logistic Regression (Baseline)
   - Random Forest Classifier
   - XGBoost Classifier
4. Validation-based model selection (Macro F1) and artifact serialization with joblib.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import xgboost as xgb

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.features import SocialPulseFeaturePipeline


def split_dataset(
    df: pd.DataFrame,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified 70/15/15 train/val/test split.
    Saves split dataframes to data/processed/.
    """
    print("Splitting dataset into Train (70%), Validation (15%), and Test (15%)...")

    # First split off Test (15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_RATIO,
        random_state=random_seed,
        stratify=df["target"],
    )

    # Next split remaining 85% into Train (70% total) and Val (15% total)
    # val_ratio_relative = 0.15 / 0.85 = ~0.17647
    val_relative_size = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_seed,
        stratify=train_val_df["target"],
    )

    # Save to disk
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    val_df.to_csv(config.VAL_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    print(f"Train set: {train_df.shape[0]} samples")
    print(f"Val set:   {val_df.shape[0]} samples")
    print(f"Test set:  {test_df.shape[0]} samples")

    return train_df, val_df, test_df


def train_and_tune_models(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    random_seed: int = config.RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Trains and tunes Logistic Regression, Random Forest, and XGBoost models.
    Handles class imbalance using 'balanced' weighting.
    Evaluates each on the validation set using Macro F1.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
    trained_models = {}
    val_scores = {}

    # Calculate balanced sample weights for models that don't natively take class_weight='balanced'
    sample_weights_train = compute_sample_weight("balanced", y_train)

    print("\n" + "=" * 60)
    print("1. Tuning Logistic Regression (Baseline)...")
    print("=" * 60)
    lr_params = {
        "C": [0.1, 1.0, 5.0],
        "max_iter": [500],
        "class_weight": ["balanced"],
        "solver": ["lbfgs"],
    }
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=random_seed),
        lr_params,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    y_val_pred_lr = best_lr.predict(X_val)
    f1_lr = f1_score(y_val, y_val_pred_lr, average="macro")
    acc_lr = accuracy_score(y_val, y_val_pred_lr)
    print(f"Logistic Regression Best Params: {lr_grid.best_params_}")
    print(f"Validation Accuracy: {acc_lr:.4f} | Validation Macro F1: {f1_lr:.4f}")

    trained_models["Logistic Regression"] = best_lr
    val_scores["Logistic Regression"] = {
        "macro_f1": f1_lr,
        "accuracy": acc_lr,
        "best_params": lr_grid.best_params_,
    }

    print("\n" + "=" * 60)
    print("2. Tuning Random Forest Classifier...")
    print("=" * 60)
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [12, 20, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=random_seed, n_jobs=-1),
        rf_params,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_val_pred_rf = best_rf.predict(X_val)
    f1_rf = f1_score(y_val, y_val_pred_rf, average="macro")
    acc_rf = accuracy_score(y_val, y_val_pred_rf)
    print(f"Random Forest Best Params: {rf_grid.best_params_}")
    print(f"Validation Accuracy: {acc_rf:.4f} | Validation Macro F1: {f1_rf:.4f}")

    trained_models["Random Forest"] = best_rf
    val_scores["Random Forest"] = {
        "macro_f1": f1_rf,
        "accuracy": acc_rf,
        "best_params": rf_grid.best_params_,
    }

    print("\n" + "=" * 60)
    print("3. Tuning XGBoost Classifier...")
    print("=" * 60)
    # Compute class weights for sample weighting in XGBoost
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sample_weights_train_xgb = compute_sample_weight(
        class_weight={cls: weight for cls, weight in zip(classes, class_weights)},
        y=y_train,
    )

    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=random_seed,
        eval_metric="mlogloss",
        tree_method="hist",
    )
    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    xgb_grid = GridSearchCV(
        xgb_clf,
        xgb_params,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    xgb_grid.fit(X_train, y_train, sample_weight=sample_weights_train_xgb)
    best_xgb = xgb_grid.best_estimator_
    y_val_pred_xgb = best_xgb.predict(X_val)
    f1_xgb = f1_score(y_val, y_val_pred_xgb, average="macro")
    acc_xgb = accuracy_score(y_val, y_val_pred_xgb)
    print(f"XGBoost Best Params: {xgb_grid.best_params_}")
    print(f"Validation Accuracy: {acc_xgb:.4f} | Validation Macro F1: {f1_xgb:.4f}")

    trained_models["XGBoost"] = best_xgb
    val_scores["XGBoost"] = {
        "macro_f1": f1_xgb,
        "accuracy": acc_xgb,
        "best_params": xgb_grid.best_params_,
    }

    # Select Best Model based on Validation Macro F1
    best_model_name = max(val_scores, key=lambda k: val_scores[k]["macro_f1"])
    best_model = trained_models[best_model_name]
    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Best Validation Macro F1: {val_scores[best_model_name]['macro_f1']:.4f}")
    print("=" * 60)

    return {
        "trained_models": trained_models,
        "val_scores": val_scores,
        "best_model_name": best_model_name,
        "best_model": best_model,
    }


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes end-to-end training pipeline:
    1. Loads processed dataset
    2. Performs stratified train/val/test split
    3. Fits FeaturePipeline on Train set
    4. Transforms Train and Validation sets
    5. Tunes & trains 3 models with class weight balancing
    6. Saves best model and pipeline artifacts to models/
    """
    # 1. Load data
    processed_path = config.PROCESSED_DATASET_PATH
    if not processed_path.exists():
        from src.preprocess import load_and_preprocess_dataset
        df = load_and_preprocess_dataset()
    else:
        df = pd.read_csv(processed_path)

    # 2. Split
    train_df, val_df, test_df = split_dataset(df)

    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    # 3. Fit Feature Pipeline ONLY on Train set (prevents leakage!)
    print("\nFitting SocialPulseFeaturePipeline on Training data...")
    feature_pipeline = SocialPulseFeaturePipeline(
        max_tfidf_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    feature_pipeline.fit(train_df)

    # Transform Train, Val, Test
    print("Transforming feature matrices...")
    X_train = feature_pipeline.transform(train_df)
    X_val = feature_pipeline.transform(val_df)
    X_test = feature_pipeline.transform(test_df)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")

    # 4. Train & Tune Models
    results = train_and_tune_models(X_train, y_train, X_val, y_val)

    best_model_name = results["best_model_name"]
    best_model = results["best_model"]
    trained_models = results["trained_models"]

    # 5. Save all models and pipeline to disk with joblib
    print("\nSaving model artifacts to models/...")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save feature pipeline
    joblib.dump(feature_pipeline, config.FEATURE_PIPELINE_PATH)
    print(f"Saved feature pipeline: {config.FEATURE_PIPELINE_PATH}")

    # Save best model
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    print(f"Saved best model ({best_model_name}): {config.BEST_MODEL_PATH}")

    # Save individual models for comparative evaluation
    for name, model in trained_models.items():
        fname = config.MODELS_DIR / f"model_{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, fname)

    # Save metadata JSON
    metadata = {
        "best_model_name": best_model_name,
        "target_classes": config.TARGET_CLASSES,
        "class_mapping": config.CLASS_MAPPING,
        "feature_count": X_train.shape[1],
        "feature_names": feature_pipeline.get_feature_names_out(),
        "validation_scores": {k: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]} for k, v in results["val_scores"].items()},
        "imbalance_strategy": "class_weight='balanced' (Logistic Regression, Random Forest) & balanced sample weights (XGBoost)",
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata: {config.METADATA_PATH}")

    return {
        "feature_pipeline": feature_pipeline,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    run_training_pipeline()
