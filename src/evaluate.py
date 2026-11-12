"""
Evaluation and Explainability module for SocialPulse AI.
Computes:
1. Classification metrics: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 (Macro/Weighted).
2. Confusion Matrices (counts and normalized heatmaps).
3. Multiclass One-vs-Rest ROC Curves and ROC-AUC scores.
4. Top 20 Global Feature Importances.
5. SHAP (SHapley Additive exPlanations) summary plot.
6. Exports model comparison CSV and evaluation report.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from scipy.sparse import issparse

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Set plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def evaluate_single_model(
    model,
    X_test,
    y_test: np.ndarray,
    model_name: str,
    target_names: List[str] = config.TARGET_CLASSES,
) -> Dict[str, Any]:
    """
    Evaluates a single model on test features and returns a metric dictionary.
    """
    y_pred = model.predict(X_test)

    # Class probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        # Softmax of decision function
        df = model.decision_function(X_test)
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        y_prob = exp_df / np.sum(exp_df, axis=1, keepdims=True)
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Multiclass ROC-AUC (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    if y_prob is not None:
        try:
            roc_auc_macro = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
        except Exception:
            roc_auc_macro = np.nan
    else:
        roc_auc_macro = np.nan

    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_ovr": roc_auc_macro,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def plot_confusion_matrices(eval_results: List[Dict[str, Any]], output_dir: Path = config.FIGURES_DIR):
    """
    Plots confusion matrix heatmaps for each model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = config.TARGET_CLASSES

    for res in eval_results:
        model_name = res["model_name"]
        cm = res["confusion_matrix"]
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=axes[0],
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)", fontsize=13, weight="bold")
        axes[0].set_xlabel("Predicted Class", fontsize=11)
        axes[0].set_ylabel("True Class", fontsize=11)

        # Normalized percentages
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=axes[1],
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[1].set_title(f"{model_name} - Normalized Confusion Matrix", fontsize=13, weight="bold")
        axes[1].set_xlabel("Predicted Class", fontsize=11)
        axes[1].set_ylabel("True Class", fontsize=11)

        plt.tight_layout()
        save_path = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved confusion matrix: {save_path}")


def plot_multiclass_roc_curves(eval_results: List[Dict[str, Any]], y_test: np.ndarray, output_dir: Path = config.FIGURES_DIR):
    """
    Plots One-vs-Rest ROC curves for each class across models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = config.TARGET_CLASSES
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    colors = ["#2563EB", "#10B981", "#F59E0B"]

    for res in eval_results:
        model_name = res["model_name"]
        y_prob = res["y_prob"]
        if y_prob is None:
            continue

        plt.figure(figsize=(8, 6))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=colors[i % len(colors)],
                lw=2.2,
                label=f"{classes[i]} (AUC = {roc_auc:.3f})",
            )

        plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Chance (AUC = 0.50)")
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate (Recall)", fontsize=12)
        plt.title(f"Multiclass ROC Curve (One-vs-Rest) - {model_name}", fontsize=13, weight="bold")
        plt.legend(loc="lower right", frameon=True, fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)

        save_path = output_dir / f"roc_curves_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved ROC curves: {save_path}")


def plot_top_features(
    model,
    feature_names: List[str],
    model_name: str,
    top_k: int = 20,
    output_dir: Path = config.FIGURES_DIR,
) -> pd.DataFrame:
    """
    Extracts, displays, and plots the top K features by importance or coefficient magnitude.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        metric_name = "Importance Score"
    elif hasattr(model, "coef_"):
        # For multiclass logistic regression, take average absolute magnitude across classes
        importances = np.mean(np.abs(model.coef_), axis=0)
        metric_name = "Mean Absolute Coefficient"
    else:
        print(f"Model {model_name} does not expose feature importances.")
        return pd.DataFrame()

    feat_df = pd.DataFrame({
        "Feature": feature_names[:len(importances)],
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).head(top_k)

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("mako", n_colors=top_k)[::-1]
    sns.barplot(data=feat_df, y="Feature", x="Importance", palette=palette)
    plt.title(f"Top {top_k} Features - {model_name}", fontsize=14, weight="bold", pad=12)
    plt.xlabel(metric_name, fontsize=12)
    plt.ylabel("Engineered Feature", fontsize=12)
    plt.tight_layout()

    save_path = output_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved top {top_k} feature importance plot: {save_path}")

    return feat_df


def generate_shap_summary(
    model,
    X_test,
    feature_names: List[str],
    output_dir: Path = config.FIGURES_DIR,
    sample_size: int = 200,
):
    """
    Generates and saves SHAP summary explanation for tree-based models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "shap_summary.png"

    try:
        import shap

        print("\nComputing SHAP values for model explainability...")
        # Convert sparse to dense sample if needed
        if issparse(X_test):
            indices = np.random.choice(X_test.shape[0], size=min(sample_size, X_test.shape[0]), replace=False)
            X_sample = X_test[indices].toarray()
        else:
            indices = np.random.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
            X_sample = X_test[indices]

        # Use TreeExplainer for tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Plot summary
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            # Multiclass list of arrays: plot for class 2 (Viral) or all
            # Class 2 is 'Viral'
            shap.summary_plot(
                shap_values[2],
                X_sample,
                feature_names=feature_names,
                max_display=20,
                show=False,
            )
            plt.title("SHAP Summary Plot - Class: Viral (Impact on Virality Prediction)", fontsize=13, weight="bold", pad=15)
        else:
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                max_display=20,
                show=False,
            )
            plt.title("SHAP Summary Plot", fontsize=13, weight="bold", pad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot: {save_path}")
    except Exception as e:
        print(f"Notice: SHAP plot generation skipped or handled gracefully: {e}")


def run_full_evaluation() -> pd.DataFrame:
    """
    Executes full evaluation suite on the Test set:
    - Loads test data and pipeline
    - Evaluates all models
    - Computes and saves comparison table
    - Generates confusion matrices, ROC curves, feature importances, and SHAP plots
    """
    print("\n" + "=" * 60)
    print("RUNNING FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    # 1. Load test data and feature pipeline
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    y_test = test_df["target"].values

    pipeline = joblib.load(config.FEATURE_PIPELINE_PATH)
    feature_names = pipeline.get_feature_names_out()
    X_test = pipeline.transform(test_df)

    # 2. Load trained models
    model_files = {
        "Logistic Regression": config.MODELS_DIR / "model_logistic_regression.joblib",
        "Random Forest": config.MODELS_DIR / "model_random_forest.joblib",
        "XGBoost": config.MODELS_DIR / "model_xgboost.joblib",
    }

    eval_results = []
    comparison_records = []

    for name, path in model_files.items():
        if path.exists():
            model = joblib.load(path)
            res = evaluate_single_model(model, X_test, y_test, name)
            eval_results.append(res)

            comparison_records.append({
                "Model": name,
                "Accuracy": round(res["accuracy"], 4),
                "Precision (Macro)": round(res["precision_macro"], 4),
                "Recall (Macro)": round(res["recall_macro"], 4),
                "F1 Score (Macro)": round(res["f1_macro"], 4),
                "F1 Score (Weighted)": round(res["f1_weighted"], 4),
                "ROC-AUC (OvR)": round(res["roc_auc_ovr"], 4),
            })

    comparison_df = pd.DataFrame(comparison_records).sort_values(by="F1 Score (Macro)", ascending=False)
    comparison_df.to_csv(config.MODEL_COMPARISON_CSV, index=False)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON TABLE (TEST SET)")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

    # 3. Generate Visualizations
    plot_confusion_matrices(eval_results)
    plot_multiclass_roc_curves(eval_results, y_test)

    # 4. Feature Importances for best model and tree model
    best_model = joblib.load(config.BEST_MODEL_PATH)
    metadata = json.loads(config.METADATA_PATH.read_text())
    best_name = metadata.get("best_model_name", "Best Model")

    top_feats = plot_top_features(best_model, feature_names, best_name, top_k=20)
    print("\nTop 20 Features for Best Model:")
    print(top_feats.to_string(index=False))

    # Also plot for XGBoost if present
    if "XGBoost" in model_files and model_files["XGBoost"].exists():
        xgb_m = joblib.load(model_files["XGBoost"])
        plot_top_features(xgb_m, feature_names, "XGBoost", top_k=20)
        # 5. SHAP summary
        generate_shap_summary(xgb_m, X_test, feature_names)

    return comparison_df


if __name__ == "__main__":
    run_full_evaluation()
