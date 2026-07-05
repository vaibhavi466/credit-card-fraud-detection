"""
run_pipeline.py — Single entrypoint that reproduces all results end-to-end.

Usage:
    python run_pipeline.py                  # full pipeline (Phases 1–7)
    python run_pipeline.py --skip-autoencoder  # skip Phase 7 (faster)
    python run_pipeline.py --phase 3        # run only up to phase N

All outputs (metrics, plots, saved models) are written to:
    reports/metrics.json
    reports/figures/
    models/

Requirements:
    pip install -r requirements.txt
    Dataset at data/raw/creditcard.csv (see README for download instructions)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import config

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
parser.add_argument("--skip-autoencoder", action="store_true",
                    help="Skip Phase 7 (autoencoder) — runs faster")
parser.add_argument("--phase", type=int, default=7,
                    help="Stop after this phase number (1-7, default=7)")
parser.add_argument("--tune", action="store_true",
                    help="Run grid search for best model (adds ~10 min)")
args = parser.parse_args()

skip_autoencoder = args.skip_autoencoder or (args.phase < 7)
max_phase = args.phase

t_start = time.time()


def _print_summary(all_metrics, best_name, best_auprc, cost_res, shap_res):
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nBest model: {best_name}")
    print(f"AUPRC (primary metric): {best_auprc:.4f}")
    print(f"Cost-optimal threshold: {cost_res['optimal_threshold']:.4f}")
    print(f"Mean fraud amount (FN cost): ${cost_res['fn_cost_per_transaction']:.2f}")
    print(f"Top SHAP features: {shap_res['top_features'][:3]}")
    print(f"\nAll metrics saved to: {config.METRICS_PATH}")
    print(f"Plots saved to:       {config.FIGURES_DIR}/")
    print(f"Models saved to:      {config.MODELS_DIR}/")
    print(f"\nTime elapsed: {t_elapsed/60:.1f} min")
    print("\nNext steps:")
    print("  streamlit run app/streamlit_app.py")
    print("  pytest -q tests/")


print("=" * 70)
print("Credit Card Fraud Detection Pipeline")
print("=" * 70)

# ── Phase 0: Verify environment ───────────────────────────────────────────────
print("\n[Phase 0] Verifying environment …")
config.MODELS_DIR.mkdir(exist_ok=True)
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Phase 1: Load + EDA summary ───────────────────────────────────────────────
print("\n[Phase 1] Loading and validating dataset …")
from src.data_loader import load_raw_data, get_features_and_target
import pandas as pd
import numpy as np

df = load_raw_data()
X, y = get_features_and_target(df)

# Quick EDA stats (full EDA in notebooks/01_eda.ipynb)
fraud_count = int(y.sum())
legit_count = int((y == 0).sum())
fraud_pct = fraud_count / len(y) * 100
mean_fraud_amount = df.loc[df["Class"] == 1, "Amount"].mean()
mean_legit_amount = df.loc[df["Class"] == 0, "Amount"].mean()

print(f"  Dataset: {len(df):,} transactions")
print(f"  Fraud:   {fraud_count} ({fraud_pct:.4f}%)")
print(f"  Legit:   {legit_count:,}")
print(f"  Mean fraud amount: ${mean_fraud_amount:.2f}")
print(f"  Mean legit amount: ${mean_legit_amount:.2f}")

# Run EDA (saves plots to reports/figures/)
from src.eda import run_eda
eda_findings = run_eda(df)

if max_phase < 2:
    sys.exit(0)

# ── Phase 2: Preprocessing + Baseline ─────────────────────────────────────────
print("\n[Phase 2] Preprocessing + Logistic Regression baseline …")
from src.preprocessing import preprocess
from src.train import build_logistic_regression, train_model, save_model
from src.evaluate import evaluate_model, log_metrics_json

X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

import joblib
joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")

# Baseline: LR with class_weight='balanced', no resampling
lr_baseline = build_logistic_regression(class_weight="balanced")
lr_baseline = train_model(lr_baseline, X_train, y_train)
save_model(lr_baseline, "lr_baseline")

metrics_baseline = evaluate_model(lr_baseline, X_test, y_test, model_name="LR_baseline")
log_metrics_json(metrics_baseline, tag="LR_baseline")

if max_phase < 3:
    sys.exit(0)

# ── Phase 3: Imbalance strategies × model families ────────────────────────────
print("\n[Phase 3] Training all strategy × model combinations …")
from src.resampling import RESAMPLING_STRATEGIES
from src.train import build_random_forest, build_xgboost

# We'll track the best model by AUPRC
best_auprc = -1
best_model = None
best_model_name = ""
all_metrics = []

# Strategies that need class_weight vs those that don't (because SMOTE balanced the data)
NEEDS_CLASS_WEIGHT = {"class_weight", "undersample"}

for strategy_name, resample_fn in RESAMPLING_STRATEGIES.items():
    print(f"\n  ── Strategy: {strategy_name} ──")
    X_res, y_res = resample_fn(X_train, y_train)
    use_cw = strategy_name in NEEDS_CLASS_WEIGHT

    models_to_train = {
        "LR": build_logistic_regression(class_weight="balanced" if use_cw else None),
        "RF": build_random_forest(class_weight="balanced" if use_cw else None),
        "XGB": build_xgboost(y_train=y_res, use_class_weight=(not use_cw)),
    }

    for model_abbr, model in models_to_train.items():
        tag = f"{strategy_name}_{model_abbr}"
        model = train_model(model, X_res, y_res)
        m = evaluate_model(model, X_test, y_test, model_name=tag)
        log_metrics_json(m, tag=tag)
        all_metrics.append(m)

        if m["auprc"] > best_auprc:
            best_auprc = m["auprc"]
            best_model = model
            best_model_name = tag
            save_model(model, "best_model")
            print(f"  ★ New best model: {tag} (AUPRC={best_auprc:.4f})")

print(f"\nBest model after Phase 3: {best_model_name} (AUPRC={best_auprc:.4f})")

if max_phase < 4:
    sys.exit(0)

# ── Phase 4: Evaluation deep dive ─────────────────────────────────────────────
print(f"\n[Phase 4] Evaluation deep dive for {best_model_name} …")
from src.evaluate import (
    plot_confusion_matrix, plot_pr_curve, plot_roc_curve, plot_threshold_sweep
)

plot_confusion_matrix(best_model, X_test, y_test, title=best_model_name)
plot_pr_curve(best_model, X_test, y_test, title=best_model_name)
plot_roc_curve(best_model, X_test, y_test, title=best_model_name)
plot_threshold_sweep(best_model, X_test, y_test)

if max_phase < 5:
    sys.exit(0)

# ── Phase 5: Cost analysis ────────────────────────────────────────────────────
print("\n[Phase 5] Cost-sensitive threshold selection …")
from src.cost_analysis import run_cost_analysis

cost_results = run_cost_analysis(best_model, X_test, y_test, df)
optimal_threshold = cost_results["optimal_threshold"]
print(f"  Optimal threshold (min cost): {optimal_threshold:.4f}")

# Save a confusion matrix at the cost-optimal threshold too
plot_confusion_matrix(
    best_model, X_test, y_test,
    threshold=optimal_threshold,
    title=f"{best_model_name} (cost-optimal threshold={optimal_threshold:.2f})",
    save=True,
)
config.FIGURES_DIR.mkdir(exist_ok=True)
# Rename to avoid overwriting the default threshold version
import shutil
shutil.copy(
    config.FIGURES_DIR / "confusion_matrix.png",
    config.FIGURES_DIR / "confusion_matrix_optimal.png"
)

# Re-evaluate at optimal threshold
metrics_optimal = evaluate_model(
    best_model, X_test, y_test,
    threshold=optimal_threshold,
    model_name=f"{best_model_name}_optimal_threshold"
)
log_metrics_json(metrics_optimal, tag=f"{best_model_name}_optimal_threshold")
log_metrics_json(
    {**cost_results, "model_name": best_model_name},
    tag="cost_analysis"
)

# Update threshold sweep with optimal marker
plot_threshold_sweep(best_model, X_test, y_test, optimal_threshold=optimal_threshold)

if max_phase < 6:
    sys.exit(0)

# ── Phase 6: SHAP explainability ─────────────────────────────────────────────
print("\n[Phase 6] SHAP explainability …")
from src.explain import run_explainability

shap_results = run_explainability(best_model, X_train, X_test, y_test)
log_metrics_json(
    {"model_name": best_model_name, "top_shap_features": shap_results["top_features"]},
    tag="shap_summary"
)

if max_phase < 7 or skip_autoencoder:
    _print_summary(all_metrics, best_model_name, best_auprc, cost_results, shap_results)
    sys.exit(0)

# ── Phase 7: Autoencoder (stretch) ────────────────────────────────────────────
print("\n[Phase 7] Autoencoder (unsupervised) …")
from src.autoencoder import run_autoencoder_pipeline

ae_metrics, ae, ae_threshold = run_autoencoder_pipeline(X_train, X_test, y_train, y_test)
log_metrics_json(ae_metrics, tag="autoencoder")

# ── Final summary ─────────────────────────────────────────────────────────────
_print_summary(all_metrics, best_model_name, best_auprc, cost_results, shap_results)
