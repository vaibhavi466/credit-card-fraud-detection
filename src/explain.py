"""
src/explain.py — SHAP-based model explainability.

What is a SHAP value?
SHAP (SHapley Additive exPlanations) assigns each feature a contribution score
to a specific prediction. Grounded in cooperative game theory (Shapley values),
SHAP values explain contributions relative to the model's expected output in the
explainer's output space.

Why SHAP over simple feature importances?
- Feature importances (Gini, permutation) are global and can't explain individual predictions.
- SHAP is both locally faithful (exact for each sample) and globally consistent
  (sums to the model output).
- TreeExplainer computes exact SHAP values for tree models in polynomial time,
  making it practical for production use.
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


def _extract_tree_model(model):
    """Extract underlying estimator if passed an imbalanced-learn Pipeline."""
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def _create_tree_explainer(tree_model, X_background: pd.DataFrame):
    """Safely create TreeExplainer with fallback for XGBoost version differences."""
    try:
        return shap.TreeExplainer(tree_model, X_background)
    except Exception:
        return shap.TreeExplainer(tree_model, feature_perturbation="tree_path_dependent")


def compute_shap_values(model, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """
    Compute SHAP values using TreeExplainer.
    """
    tree_model = _extract_tree_model(model)
    print(f"Computing SHAP values for {len(X_explain)} samples …")
    try:
        explainer = _create_tree_explainer(tree_model, X_background)
        shap_values = explainer.shap_values(X_explain, check_additivity=False)
    except Exception:
        explainer = shap.TreeExplainer(tree_model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_explain, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    print(f"SHAP values computed: {shap_values.shape}")
    return shap_values, explainer


def plot_shap_summary(shap_values: np.ndarray, X_explain: pd.DataFrame, save=True) -> Path:
    """Save a SHAP summary bar plot (global feature importance)."""
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
    """Save a SHAP beeswarm plot (shows direction of impact per feature)."""
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
    """Save a SHAP waterfall plot for a single transaction."""
    tree_model = _extract_tree_model(model)
    explainer = _create_tree_explainer(tree_model, X_background)

    sample = X_sample.iloc[[sample_idx]]
    try:
        explanation = explainer(sample, check_additivity=False)
    except Exception:
        explainer = shap.TreeExplainer(tree_model, feature_perturbation="tree_path_dependent")
        explanation = explainer(sample, check_additivity=False)

    if explanation.values.ndim == 3:
        explanation = shap.Explanation(
            values=explanation.values[0, :, 1],
            base_values=explanation.base_values[0, 1],
            data=explanation.data[0],
            feature_names=X_sample.columns.tolist(),
        )
    else:
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


def run_explainability(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    operating_threshold: float = None,
) -> dict:
    """
    Full explainability pipeline: background sample → SHAP values → plots.
    """
    threshold = operating_threshold if operating_threshold is not None else 0.5

    rng = np.random.default_rng(config.RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=min(config.SHAP_BACKGROUND_SAMPLES, len(X_train)), replace=False)
    X_background = X_train.iloc[bg_idx]

    # Deterministic random sample of test set for global explanations
    X_explain = X_test.sample(n=min(500, len(X_test)), random_state=config.RANDOM_STATE)

    shap_values, _ = compute_shap_values(model, X_background, X_explain)

    summary_path = plot_shap_summary(shap_values, X_explain)
    beeswarm_path = plot_shap_beeswarm(shap_values, X_explain)

    # True Positive selection at locked operating threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    tp_mask = (y_pred == 1) & (y_test.values == 1)
    if tp_mask.sum() == 0:
        print(f"  Warning: no True Positives at threshold={threshold:.2f}; using first fraud sample instead")
        tp_mask = y_test.values == 1

    tp_indices = np.where(tp_mask)[0]
    best_tp_idx = tp_indices[np.argmax(y_proba[tp_indices])]

    waterfall_path = plot_shap_waterfall(model, X_background, X_test, sample_idx=best_tp_idx)

    print("\n  SHAP interpretation:")
    top_features = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_explain.columns
    ).sort_values(ascending=False).head(3)
    print(f"  Top 3 features by mean |SHAP|: {top_features.index.tolist()}")

    return {
        "shap_summary": str(summary_path),
        "shap_beeswarm": str(beeswarm_path),
        "shap_waterfall": str(waterfall_path),
        "top_features": top_features.index.tolist(),
    }
