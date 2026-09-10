"""
run_pipeline.py — Single entrypoint that reproduces all results end-to-end.

Methodology Guarantees:
1. 3-Way Stratified Split: Train (60%), Validation (20%), Test (20%).
2. Preprocessing & Scaler fitted ONLY on X_train.
3. Candidate models evaluated & selected using VALIDATION set AUPRC.
4. Operating threshold optimized on VALIDATION set predictions.
5. Final test set evaluated EXACTLY ONCE on locked model + locked threshold.

Usage:
    python run_pipeline.py                  # full pipeline (Phases 1–8)
    python run_pipeline.py --skip-autoencoder  # skip Phase 8 (faster)
    python run_pipeline.py --tune           # perform leak-free CV grid search on selected candidate
    python run_pipeline.py --phase 3        # run only up to phase N
"""

import argparse
import shutil
import sys
import time
import joblib

import config
from src.evaluate import reset_metrics_json, evaluate_model, log_metrics_json

parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
parser.add_argument("--skip-autoencoder", action="store_true",
                    help="Skip Phase 8 (autoencoder) — runs faster")
parser.add_argument("--phase", type=int, default=8,
                    help="Stop after this phase number (1-8, default=8)")
parser.add_argument("--tune", action="store_true",
                    help="Run leak-free grid search CV on selected winning candidate")
args = parser.parse_args()

skip_autoencoder = args.skip_autoencoder or (args.phase < 8)
max_phase = args.phase

t_start = time.time()


def _print_summary(best_name, val_auprc, test_metrics_optimal, cost_res, shap_res):
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE (METHODOLOGICALLY VALIDATED)")
    print("=" * 70)
    print(f"\nValidation-Selected Candidate: {best_name}")
    print(f"Validation AUPRC (selection metric): {val_auprc:.4f}")
    print(f"Locked Operating Threshold:         {cost_res['optimal_threshold']:.4f}")
    print(f"Mean Fraud Amount (FN Cost):        ${cost_res['fn_cost_per_transaction']:.2f}")
    print("\nFINAL UNTOUCHED TEST RESULTS:")
    print(f"  Test AUPRC:     {test_metrics_optimal['auprc']:.4f}")
    print(f"  Test ROC-AUC:   {test_metrics_optimal['roc_auc']:.4f}")
    print(f"  Test Precision: {test_metrics_optimal['precision']:.4f}")
    print(f"  Test Recall:    {test_metrics_optimal['recall']:.4f}")
    print(f"  Test F1 Score:  {test_metrics_optimal['f1']:.4f}")
    print(f"\nTop SHAP features: {shap_res.get('top_features', [])[:3]}")
    print(f"\nAll metrics saved to: {config.METRICS_PATH}")
    print(f"Plots saved to:       {config.FIGURES_DIR}/")
    print(f"Models saved to:      {config.MODELS_DIR}/")
    print(f"\nTime elapsed: {t_elapsed/60:.1f} min")
    print("\nNext steps:")
    print("  streamlit run app/streamlit_app.py")
    print("  pytest -q tests/")


print("=" * 70)
print("Credit Card Fraud Detection Pipeline — Methodologically Hardened")
print("=" * 70)

# ── Phase 0: Verify environment & initialize clean metrics store ─────────────
print("\n[Phase 0] Verifying environment & resetting metrics database ...")
config.MODELS_DIR.mkdir(exist_ok=True)
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
reset_metrics_json()

# ── Phase 1: Load + EDA summary ───────────────────────────────────────────────
print("\n[Phase 1] Loading and validating dataset ...")
from src.data_loader import load_raw_data, get_features_and_target

df = load_raw_data()
X, y = get_features_and_target(df)

fraud_count = int(y.sum())
legit_count = int((y == 0).sum())
fraud_pct = fraud_count / len(y) * 100

print(f"  Dataset: {len(df):,} transactions")
print(f"  Fraud:   {fraud_count} ({fraud_pct:.4f}%)")
print(f"  Legit:   {legit_count:,}")

from src.eda import run_eda
eda_findings = run_eda(df)

if max_phase < 2:
    sys.exit(0)

# ── Phase 2: 3-Way Stratified Split + Scaler (Train only) ────────────────────
print("\n[Phase 2] Preprocessing (60% Train / 20% Val / 20% Test) + Baseline ...")
from src.preprocessing import preprocess_train_val_test
from src.train import build_logistic_regression, train_model, save_model

X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_train_val_test(X, y)
joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")

# Baseline: LR with class_weight='balanced', trained on Train, evaluated on Validation
lr_baseline = build_logistic_regression(class_weight="balanced")
lr_baseline = train_model(lr_baseline, X_train, y_train)
save_model(lr_baseline, "lr_baseline")

metrics_baseline_val = evaluate_model(
    lr_baseline, X_val, y_val,
    model_name="LR_baseline", split="validation", purpose="model_selection"
)
log_metrics_json(metrics_baseline_val, tag="LR_baseline")

if max_phase < 3:
    sys.exit(0)

# ── Phase 3: Imbalance strategies x model families (Validation Selection) ────
print("\n[Phase 3] Training candidates & evaluating on VALIDATION set ...")
from src.resampling import RESAMPLING_STRATEGIES
from src.train import build_random_forest, build_xgboost

best_val_auprc = -1.0
best_model = None
best_model_name = ""
best_strategy_name = ""
best_model_abbr = ""
all_val_metrics = []

NEEDS_CLASS_WEIGHT = {"class_weight", "undersample"}

for strategy_name, resample_fn in RESAMPLING_STRATEGIES.items():
    print(f"\n  === Strategy: {strategy_name} ===")
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
        m_val = evaluate_model(
            model, X_val, y_val,
            model_name=tag, split="validation", purpose="model_selection"
        )
        log_metrics_json(m_val, tag=tag)
        all_val_metrics.append(m_val)

        if m_val["auprc"] > best_val_auprc:
            best_val_auprc = m_val["auprc"]
            best_model = model
            best_model_name = tag
            best_strategy_name = strategy_name
            best_model_abbr = model_abbr
            save_model(model, "best_model")
            print(f"  [BEST VALIDATION] Winning candidate: {tag} (Val AUPRC={best_val_auprc:.4f})")

print(f"\nValidation Selected Model: {best_model_name} (Val AUPRC={best_val_auprc:.4f})")

if max_phase < 4:
    sys.exit(0)

# ── Phase 4: Optional Leak-Free CV Hyperparameter Tuning ────────────────────
if args.tune:
    print(f"\n[Phase 4] Tuning selected candidate ({best_model_name}) via leak-free CV Pipeline ...")
    from src.train import tune_best_model
    best_model = tune_best_model(
        X_train, y_train,
        model_type=best_model_abbr,
        strategy_name=best_strategy_name
    )
    save_model(best_model, "best_model")
    # Re-evaluate tuned model on validation set
    m_val_tuned = evaluate_model(
        best_model, X_val, y_val,
        model_name=f"{best_model_name}_tuned", split="validation", purpose="model_selection"
    )
    log_metrics_json(m_val_tuned, tag=f"{best_model_name}_tuned")
    best_val_auprc = m_val_tuned["auprc"]

if max_phase < 5:
    sys.exit(0)

# ── Phase 5: Cost-Sensitive Threshold Selection (Validation Set) ────────────
print("\n[Phase 5] Cost-sensitive threshold optimization on VALIDATION set ...")
from src.cost_analysis import run_cost_analysis

cost_results = run_cost_analysis(
    best_model, X_val, y_val, df_raw=df, train_indices=X_train.index
)
optimal_threshold = cost_results["optimal_threshold"]
print(f"  Locked Operating Threshold from Validation: {optimal_threshold:.4f}")

log_metrics_json(
    {**cost_results, "model_name": best_model_name},
    tag="cost_analysis"
)

if max_phase < 6:
    sys.exit(0)

# ── Phase 6: Final Untouched Test Set Evaluation ────────────────────────────
print(f"\n[Phase 6] Final evaluation of {best_model_name} on UNTOUCHED TEST SET ...")
from src.evaluate import (
    plot_confusion_matrix, plot_pr_curve, plot_roc_curve, plot_threshold_sweep
)

# Test evaluation at default threshold 0.50
m_test_default = evaluate_model(
    best_model, X_test, y_test,
    threshold=0.50,
    model_name=f"{best_model_name}_test_default",
    split="test",
    purpose="final_evaluation"
)
log_metrics_json(m_test_default, tag=f"{best_model_name}_test_default")

# Test evaluation at locked cost-optimal threshold
m_test_optimal = evaluate_model(
    best_model, X_test, y_test,
    threshold=optimal_threshold,
    model_name=f"{best_model_name}_test_optimal",
    split="test",
    purpose="final_evaluation"
)
log_metrics_json(m_test_optimal, tag=f"{best_model_name}_test_optimal")

# Save figures for untouched test set
plot_confusion_matrix(best_model, X_test, y_test, threshold=0.50, title=f"{best_model_name} (Test @ 0.50)")
plot_pr_curve(best_model, X_test, y_test, title=f"{best_model_name} (Test)")
plot_roc_curve(best_model, X_test, y_test, title=f"{best_model_name} (Test)")
plot_threshold_sweep(best_model, X_test, y_test, optimal_threshold=optimal_threshold)

plot_confusion_matrix(
    best_model, X_test, y_test,
    threshold=optimal_threshold,
    title=f"{best_model_name} (Test @ optimal threshold={optimal_threshold:.2f})",
    save=True
)
shutil.copy(
    config.FIGURES_DIR / "confusion_matrix.png",
    config.FIGURES_DIR / "confusion_matrix_optimal.png"
)

if max_phase < 7:
    sys.exit(0)

# ── Phase 7: SHAP Explainability ─────────────────────────────────────────────
print("\n[Phase 7] SHAP explainability for selected model ...")
from src.explain import run_explainability

shap_results = run_explainability(
    best_model, X_train, X_test, y_test, operating_threshold=optimal_threshold
)
log_metrics_json(
    {"model_name": best_model_name, "top_shap_features": shap_results["top_features"]},
    tag="shap_summary"
)

if max_phase < 8 or skip_autoencoder:
    _print_summary(best_model_name, best_val_auprc, m_test_optimal, cost_results, shap_results)
    sys.exit(0)

# ── Phase 8: Autoencoder (Unsupervised Stretch Goal) ─────────────────────────
print("\n[Phase 8] Autoencoder (unsupervised anomaly detection) ...")
from src.autoencoder import run_autoencoder_pipeline

ae_metrics, ae, ae_threshold = run_autoencoder_pipeline(
    X_train, X_test, y_train, y_test, X_val=X_val, y_val=y_val
)
log_metrics_json(ae_metrics, tag="autoencoder")

# ── Final Summary ─────────────────────────────────────────────────────────────
_print_summary(best_model_name, best_val_auprc, m_test_optimal, cost_results, shap_results)
