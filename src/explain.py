"""
src/explain.py — SHAP-based model explainability.

What is a SHAP value?
SHAP (SHapley Additive exPlanations) assigns each feature a "contribution score"
to a specific prediction. The score for feature i on sample x is the average
marginal contribution of feature i across all possible orderings of features —
grounded in cooperative game theory (Shapley values).

Concretely: if a transaction's SHAP value for V4 is +2.5, it means V4 pushed
the model's fraud log-odds up by 2.5 compared to the average prediction.
Negative values push toward "legit". The final prediction = base value + sum
of all SHAP values.

Why SHAP over simple feature importances?
- Feature importances (Gini, permutation) are global and can't explain individual predictions.
- SHAP is both locally faithful (exact for each sample) and globally consistent
  (sums to the model output).
- TreeExplainer computes exact SHAP values for tree models in O(TLD²) time,
  making it practical for production use.

Interview tip: "Why did the model flag this specific transaction?" is one of the
most common questions. SHAP waterfall plots answer it directly.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving files
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def compute_shap_values(model, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """
    Compute SHAP values using TreeExplainer.

    TreeExplainer is exact (not approximate) for tree-based models including
    XGBoost. It uses the model's tree structure directly rather than sampling.

    Parameters
    ----------
    model        : fitted XGBoost or RandomForest model
    X_background : small sample used to establish the SHAP baseline expectation.
                   We use config.SHAP_BACKGROUND_SAMPLES randomly chosen training
                   samples. More samples = more stable baseline, but slower.
    X_explain    : samples to explain (typically X_test)

    Returns
    -------
    shap_values : np.ndarray, shape (n_samples, n_features)
    explainer   : the TreeExplainer object (needed for waterfall plots)
    """
    print(f"Computing SHAP values for {len(X_explain)} samples …")
    explainer = shap.TreeExplainer(model, X_background)

    # check_additivity=False avoids a slow double-check pass; values are still correct
    shap_values = explainer.shap_values(X_explain, check_additivity=False)

    # For binary XGBoost, shap_values is shape (n, p) — one score per feature per sample.
    # For RandomForest, shap_values can be a list [class0_vals, class1_vals] or a 3D array of shape (n, p, 2).
    # We take the class-1 (fraud) slice.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    print(f"SHAP values computed: {shap_values.shape}")
    return shap_values, explainer


def plot_shap_summary(shap_values: np.ndarray, X_explain: pd.DataFrame, save=True) -> Path:
    """
    Save a SHAP summary bar plot (global feature importance).

    The bar length = mean(|SHAP value|) across all test samples, which is a
    reliable measure of average impact. Features are sorted high → low.

    Interpretation: "V14 has the highest average impact on the fraud prediction.
    A high |SHAP| for V14 means the model's predictions shift the most based on
    that feature, whether towards fraud or away from it."
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(
        shap_values,
        X_explain,
        plot_type="bar",
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance (mean |SHAP value|)", fontsize=13, pad=12)
    plt.tight_layout()

    path = config.FIGURES_DIR / "shap_summary.png"
    if save:
        config.FIGURES_DIR.mkdir(exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close()
    return path


def plot_shap_beeswarm(shap_values: np.ndarray, X_explain: pd.DataFrame, save=True) -> Path:
    """
    Save a SHAP beeswarm plot (shows direction of impact per feature).

    Each dot is one sample. Color = feature value (red=high, blue=low).
    Position on x-axis = SHAP value (right = pushes toward fraud).
    """
    plt.figure(figsize=(8, 7))
    shap.summary_plot(
        shap_values,
        X_explain,
        plot_type="dot",
        max_display=20,
        show=False,
    )
    plt.title("SHAP Beeswarm — Feature Impact Direction", fontsize=13, pad=12)
    plt.tight_layout()

    path = config.FIGURES_DIR / "shap_beeswarm.png"
    if save:
        config.FIGURES_DIR.mkdir(exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close()
    return path


def plot_shap_waterfall(
    model,
    X_background: pd.DataFrame,
    X_sample: pd.DataFrame,
    sample_idx: int = 0,
    save=True,
) -> Path:
    """
    Save a SHAP waterfall plot for a single transaction.

    The waterfall shows how each feature pushes the prediction up or down
    from the model's base rate (expected value) to the final fraud score.

    We pick a correctly-flagged fraud (a True Positive) so the plot answers:
    "Why did the model correctly identify this transaction as fraud?"

    Parameters
    ----------
    sample_idx : index into X_sample (should be a True Positive)
    """
    explainer = shap.TreeExplainer(model, X_background)

    # Compute Explanation object for waterfall (uses shap.Explanation, not raw arrays)
    sample = X_sample.iloc[[sample_idx]]
    explanation = explainer(sample, check_additivity=False)

    # For binary classifiers, take the fraud-class explanation
    if explanation.values.ndim == 3:
        # RandomForest: shape (1, n_features, 2)
        explanation = shap.Explanation(
            values=explanation.values[0, :, 1],
            base_values=explanation.base_values[0, 1],
            data=explanation.data[0],
            feature_names=X_sample.columns.tolist(),
        )
    else:
        # XGBoost: shape (1, n_features)
        explanation = shap.Explanation(
            values=explanation.values[0],
            base_values=explanation.base_values[0],
            data=explanation.data[0],
            feature_names=X_sample.columns.tolist(),
        )

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title("SHAP Waterfall — Single Fraud Transaction", fontsize=13, pad=12)
    plt.tight_layout()

    path = config.FIGURES_DIR / "shap_waterfall.png"
    if save:
        config.FIGURES_DIR.mkdir(exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close()
    return path


def run_explainability(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_test: pd.Series) -> dict:
    """
    Full explainability pipeline: background sample → SHAP values → plots.

    Returns dict with paths to saved figures.
    """
    # Use a random subsample of training data as the SHAP background
    rng = np.random.default_rng(config.RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=min(config.SHAP_BACKGROUND_SAMPLES, len(X_train)),
                        replace=False)
    X_background = X_train.iloc[bg_idx]

    # For the global explanation plots, use a representative subset of 500 test samples
    # to avoid the heavy O(N_samples * N_trees) computation on the full test set.
    # 500 samples is plenty for clean, informative summary and beeswarm plots.
    X_explain = X_test.head(500)

    # Compute SHAP values on the subset
    shap_values, _ = compute_shap_values(model, X_background, X_explain)

    # Global plots
    summary_path = plot_shap_summary(shap_values, X_explain)
    beeswarm_path = plot_shap_beeswarm(shap_values, X_explain)

    # Waterfall for a correctly-flagged fraud (True Positive) from the full test set
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    tp_mask = (y_pred == 1) & (y_test.values == 1)
    if tp_mask.sum() == 0:
        print("  Warning: no True Positives at threshold=0.5; using first fraud instead")
        tp_mask = y_test.values == 1

    tp_indices = np.where(tp_mask)[0]
    # Pick the highest-confidence TP for a clear waterfall
    best_tp_idx = tp_indices[np.argmax(y_proba[tp_indices])]

    # Waterfall only explains a single row, so it is instant
    waterfall_path = plot_shap_waterfall(model, X_background, X_test, sample_idx=best_tp_idx)

    print("\n  SHAP interpretation:")
    top_features = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_explain.columns
    ).sort_values(ascending=False).head(3)
    print(f"  Top 3 features by mean |SHAP|: {top_features.index.tolist()}")
    print(
        "  These features had the largest average influence on fraud predictions. "
        "The waterfall plot shows the exact contribution of each feature for one "
        "specific transaction that the model correctly identified as fraud."
    )

    return {
        "shap_summary": str(summary_path),
        "shap_beeswarm": str(beeswarm_path),
        "shap_waterfall": str(waterfall_path),
        "top_features": top_features.index.tolist(),
    }
