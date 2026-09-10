"""
config.py — Central configuration for the Credit Card Fraud Detection pipeline.

All constants that appear in multiple modules live here so they can be changed
in one place and stay consistent everywhere. This is also the authoritative
source for random seeds — using fixed random seeds ensures deterministic
reproducibility across runs on identical software environments.
"""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42  # Fixed seed used by every stochastic component.

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

RAW_CSV = DATA_RAW_DIR / "creditcard.csv"

# ── Dataset expectations (ULB creditcard dataset) ────────────────────────────
EXPECTED_ROW_COUNT = 284_807
EXPECTED_FRAUD_COUNT = 492
TARGET_COL = "Class"
FRAUD_LABEL = 1
LEGIT_LABEL = 0

# ── Train / Validation / Test split ──────────────────────────────────────────
TEST_SIZE = 0.20                      # 20% final held-out test set
VALIDATION_SIZE_WITHIN_TRAIN = 0.25   # 25% of the 80% train_val set (giving 60% Train, 20% Val, 20% Test)
DEFAULT_OPERATING_THRESHOLD = 0.50

# ── Feature engineering ──────────────────────────────────────────────────────
# V1–V28 are already PCA-transformed by the dataset authors; we only scale
# the two raw features (Amount and Time) to put them on the same order of
# magnitude as the PCA components.
COLS_TO_SCALE = ["Amount", "Time"]

# ── Model hyperparameters ────────────────────────────────────────────────────
# Intentionally narrow grids — 5-fold CV on a 227K-row dataset is already
# compute-intensive; exhaustive search adds noise without defensible benefit.
LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["lbfgs"],
    "max_iter": [1000],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10],
    "min_samples_leaf": [1, 4],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

# ── Cost-sensitive analysis ──────────────────────────────────────────────────
# FN cost: mean fraud transaction amount (computed from actual data at runtime,
#          NOT hardcoded here — see src/cost_analysis.py).
# FP cost: $5 illustrative "customer friction" cost.
#   ⚠️  This is a business assumption, not a real figure. In a production system
#   this would be derived from customer churn rates, call-centre costs, etc.
FP_COST_DOLLARS = 5.0

# ── SHAP ─────────────────────────────────────────────────────────────────────
# Number of background samples used by the SHAP TreeExplainer.
# Smaller = faster; 100 is sufficient for a portfolio demo.
SHAP_BACKGROUND_SAMPLES = 100

# ── Threshold sweep ──────────────────────────────────────────────────────────
import numpy as np  # noqa: E402  (import after constants so file is readable top-to-bottom)
THRESHOLD_SWEEP = np.linspace(0.01, 0.99, 99)

# ── Autoencoder (stretch goal) ───────────────────────────────────────────────
AUTOENCODER_HIDDEN_LAYER_SIZES = (32, 16, 8, 16, 32)
AUTOENCODER_MAX_ITER = 20
