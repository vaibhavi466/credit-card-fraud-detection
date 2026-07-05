"""
src/evaluate.py — Model evaluation: metrics, plots, and JSON logging.

Primary metric: AUPRC (Area Under the Precision-Recall Curve).

Why AUPRC over ROC-AUC?
ROC-AUC measures the trade-off between TPR and FPR across all thresholds.
Under severe class imbalance (0.17% fraud), the False Positive Rate stays
very low even for a bad classifier because there are so many negatives — so
ROC-AUC can look great (≥0.95) while the classifier is nearly useless for
finding actual fraud. AUPRC summarises the Precision-Recall curve, which is
directly sensitive to performance on the minority class and doesn't benefit
from the large negative pool. A random classifier scores AUPRC ≈ prevalence
(~0.0017), so improvements are directly meaningful.

Why not accuracy?
With 0.17% fraud, predicting "legit" for everything yields 99.83% accuracy
while catching zero frauds. Accuracy is an actively misleading metric here.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    model_name: str = "",
) -> dict:
    """
    Evaluate a fitted model on the test set.

    Parameters
    ----------
    model     : fitted sklearn-compatible classifier
    X_test    : feature matrix (test set only — never touched during training)
    y_test    : true labels
    threshold : decision threshold (default 0.5, but see cost_analysis.py for
                a principled choice)
    model_name: string label for logging

    Returns
    -------
    dict with keys: precision, recall, f1, roc_auc, auprc,
                    confusion_matrix, threshold, model_name
    """
    # Get probability scores (not binary predictions) for threshold-independent
    # metrics (AUPRC, ROC-AUC) and for threshold sweep experiments.
    y_proba = model.predict_proba(X_test)[:, 1]

    # Apply threshold to get binary predictions
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    # average_precision_score computes AUPRC directly from probabilities —
    # no need to pick a threshold. This is the primary metric.
    auprc = average_precision_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "auprc": round(auprc, 4),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    if model_name:
        print(
            f"[{model_name}] threshold={threshold:.2f} | "
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
            f"AUPRC={auprc:.4f} ROC-AUC={roc_auc:.4f}"
        )
    return metrics


def evaluate_all_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Sweep thresholds from 0.01 to 0.99 and record precision/recall/F1.

    Returns a DataFrame with columns: threshold, precision, recall, f1.
    Used by cost_analysis.py and the Streamlit demo.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in config.THRESHOLD_SWEEP:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })
    return pd.DataFrame(rows)


def log_metrics_json(metrics: dict, tag: str = None) -> None:
    """
    Append (or initialise) a metrics entry to reports/metrics.json.

    The JSON file is a list of dicts so it can hold results from all
    strategy/model combinations in one place.
    """
    config.REPORTS_DIR.mkdir(exist_ok=True)
    path = config.METRICS_PATH

    entry = dict(metrics)
    if tag:
        entry["tag"] = tag

    # Load existing entries or start fresh
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    else:
        existing = []

    # Replace if same tag already exists (for idempotent re-runs)
    key = entry.get("tag") or entry.get("model_name", "unknown")
    existing = [e for e in existing if e.get("tag") != key and e.get("model_name") != key]
    existing.append(entry)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# ── Plotting functions ────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, threshold=0.5, title="Best Model", save=True):
    """Save a styled confusion matrix heatmap."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Legit", "Predicted Fraud"],
        yticklabels=["Actual Legit", "Actual Fraud"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {title}\n(threshold={threshold:.2f})", fontsize=13)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    plt.tight_layout()

    path = config.FIGURES_DIR / "confusion_matrix.png"
    if save:
        config.FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"  Saved -> {path}")
    plt.close(fig)
    return path


def plot_pr_curve(model, X_test, y_test, title="Best Model", save=True):
    """Save a Precision-Recall curve with AUPRC annotated."""
    y_proba = model.predict_proba(X_test)[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_vals, precision_vals, lw=2, color="#2563eb", label=f"AUPRC = {auprc:.4f}")
    # Baseline = random classifier at fraud prevalence
    baseline = y_test.mean()
    ax.axhline(baseline, ls="--", color="gray", alpha=0.7, label=f"Random baseline ({baseline:.4f})")
    ax.set_xlabel("Recall (sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (PPV)", fontsize=11)
    ax.set_title(f"Precision-Recall Curve — {title}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = config.FIGURES_DIR / "pr_curve.png"
    if save:
        config.FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"  Saved -> {path}")
    plt.close(fig)
    return path


def plot_roc_curve(model, X_test, y_test, title="Best Model", save=True):
    """Save a ROC curve with AUC annotated."""
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="#16a34a", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(f"ROC Curve — {title}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = config.FIGURES_DIR / "roc_curve.png"
    if save:
        config.FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"  Saved -> {path}")
    plt.close(fig)
    return path


def plot_threshold_sweep(model, X_test, y_test, optimal_threshold=None, save=True):
    """Save precision/recall/F1 vs threshold plot."""
    df = evaluate_all_thresholds(model, X_test, y_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["precision"], label="Precision", color="#2563eb", lw=2)
    ax.plot(df["threshold"], df["recall"], label="Recall", color="#dc2626", lw=2)
    ax.plot(df["threshold"], df["f1"], label="F1 Score", color="#16a34a", lw=2)

    if optimal_threshold is not None:
        ax.axvline(optimal_threshold, ls="--", color="orange", lw=1.5,
                   label=f"Optimal threshold ({optimal_threshold:.2f})")

    ax.set_xlabel("Classification Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision / Recall / F1 vs. Threshold", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = config.FIGURES_DIR / "threshold_sweep.png"
    if save:
        config.FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"  Saved -> {path}")
    plt.close(fig)
    return path
