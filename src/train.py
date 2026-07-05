"""
src/train.py — Model definitions and training wrappers.

Design decisions:
- class_weight='balanced' is set as default for LR and RF (they natively
  support it). For XGBoost, scale_pos_weight is computed explicitly because
  XGBoost's API doesn't take class_weight; it takes the ratio of neg/pos.
- When SMOTE has already balanced the classes, class_weight is set to None
  so we don't double-correct the imbalance.
- All models use random_state=config.RANDOM_STATE for reproducibility.
- We use cross-validation for hyperparameter selection but report final metrics
  on the held-out test set (not the CV folds) to avoid optimistic bias.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_logistic_regression(class_weight="balanced") -> LogisticRegression:
    """
    Logistic Regression baseline.

    class_weight='balanced' upweights fraud samples in the loss function
    so the model doesn't trivially predict "legit" for everything.
    This is the standard first step before trying fancier techniques.
    """
    return LogisticRegression(
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=1000,
        random_state=config.RANDOM_STATE,
    )


def build_random_forest(class_weight="balanced") -> RandomForestClassifier:
    """
    Random Forest — an ensemble of decision trees that votes by majority.

    class_weight='balanced' propagates to each tree's split criterion.
    n_estimators=100 is a good default; diminishing returns after ~300.
    """
    return RandomForestClassifier(
        n_estimators=100,
        class_weight=class_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,           # use all CPU cores
    )


def build_xgboost(y_train: pd.Series = None, use_class_weight: bool = True) -> XGBClassifier:
    """
    XGBoost — gradient-boosted trees, typically the strongest tabular model.

    XGBoost doesn't take class_weight; instead it uses scale_pos_weight =
    (# negative samples) / (# positive samples). When we've already balanced
    the training set via SMOTE, this should be ~1.0.

    Parameters
    ----------
    y_train : pd.Series, optional
        Training labels used to compute scale_pos_weight. If None or
        use_class_weight is False, scale_pos_weight defaults to 1.
    use_class_weight : bool
        Set to False when SMOTE has already balanced the classes.
    """
    if use_class_weight and y_train is not None:
        neg = int((y_train == config.LEGIT_LABEL).sum())
        pos = int((y_train == config.FRAUD_LABEL).sum())
        scale_pos_weight = neg / pos  # e.g. 227,000 / 394 ≈ 576
    else:
        scale_pos_weight = 1.0

    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fit a model and return it. Simple wrapper for logging/timing.

    This intentionally does NOT run GridSearchCV — we use sensible defaults
    and the model objects themselves encode reasonable hyperparameters.
    For the best model (SMOTE + XGBoost) we run a small grid search separately.
    """
    model_name = type(model).__name__
    print(f"  Training {model_name} on {len(y_train):,} samples …", end=" ", flush=True)
    model.fit(X_train, y_train)
    print("done.")
    return model


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
) -> object:
    """
    Run a small grid search (5-fold stratified CV, scored on AUPRC) on the
    best model type. Returns the best estimator.

    We use AUPRC (average_precision) as the CV scoring metric — the same
    primary metric we use for final evaluation. This ensures the hyperparameters
    are tuned for what we actually care about, not accuracy.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    if model_type == "xgboost":
        base = build_xgboost(y_train=y_train, use_class_weight=False)
        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        }
    elif model_type == "rf":
        base = build_random_forest(class_weight=None)
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [None, 10],
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    gs = GridSearchCV(
        base,
        param_grid,
        cv=cv,
        scoring="average_precision",  # AUPRC — our primary metric
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}  |  CV AUPRC: {gs.best_score_:.4f}")
    return gs.best_estimator_


def save_model(model, name: str) -> Path:
    """Persist a fitted model to models/<name>.joblib."""
    config.MODELS_DIR.mkdir(exist_ok=True)
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"  Saved model -> {path}")
    return path


def load_model(name: str):
    """Load a previously saved model from models/<name>.joblib."""
    path = config.MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)
