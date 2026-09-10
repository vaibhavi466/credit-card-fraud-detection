"""
src/train.py — Model definitions and training wrappers.

Design decisions:
- class_weight='balanced' is set as default for LR and RF (they natively
  support it). For XGBoost, scale_pos_weight is computed explicitly because
  XGBoost's API doesn't take class_weight; it takes the ratio of neg/pos.
- When SMOTE has already balanced the classes, class_weight is set to None
  so we don't double-correct the imbalance.
- All models use random_state=config.RANDOM_STATE for reproducibility.
- Hyperparameter tuning via --tune uses imblearn Pipeline so resampling occurs
  strictly within each cross-validation training fold without data leakage.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_logistic_regression(class_weight="balanced") -> LogisticRegression:
    """
    Logistic Regression baseline.
    """
    return LogisticRegression(
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=1000,
        random_state=config.RANDOM_STATE,
    )


def build_random_forest(class_weight="balanced") -> RandomForestClassifier:
    """
    Random Forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=100,
        class_weight=class_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def build_xgboost(y_train: pd.Series = None, use_class_weight: bool = True) -> XGBClassifier:
    """
    XGBoost classifier with optional scale_pos_weight.
    """
    if use_class_weight and y_train is not None:
        neg = int((y_train == config.LEGIT_LABEL).sum())
        pos = int((y_train == config.FRAUD_LABEL).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0
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
    """
    model_name = type(model).__name__
    print(f"  Training {model_name} on {len(y_train):,} samples …", end=" ", flush=True)
    model.fit(X_train, y_train)
    print("done.")
    return model


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "RF",
    strategy_name: str = "class_weight",
    param_grid: dict = None,
) -> object:
    """
    Run a grid search (5-fold stratified CV, scored on AUPRC) without leakage.

    Uses an imbalanced-learn Pipeline so resampling (if any) occurs independently
    within each cross-validation fold.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    use_cw = strategy_name in {"class_weight", "undersample"}

    # Base model selection
    if model_type in {"XGB", "xgboost"}:
        base = build_xgboost(y_train=y_train, use_class_weight=use_cw)
        default_grid = config.XGB_PARAM_GRID
    elif model_type in {"RF", "rf", "RandomForest"}:
        base = build_random_forest(class_weight="balanced" if use_cw else None)
        default_grid = config.RF_PARAM_GRID
    elif model_type in {"LR", "lr", "LogisticRegression"}:
        base = build_logistic_regression(class_weight="balanced" if use_cw else None)
        default_grid = config.LR_PARAM_GRID
    else:
        base = build_random_forest(class_weight="balanced" if use_cw else None)
        default_grid = config.RF_PARAM_GRID

    grid = param_grid if param_grid is not None else default_grid

    # Resampler selection
    if strategy_name == "undersample":
        resampler = RandomUnderSampler(random_state=config.RANDOM_STATE)
        pipeline = ImbPipeline([("resampler", resampler), ("model", base)])
    elif strategy_name == "smote":
        resampler = SMOTE(random_state=config.RANDOM_STATE)
        pipeline = ImbPipeline([("resampler", resampler), ("model", base)])
    elif strategy_name == "smote_tomek":
        resampler = SMOTETomek(random_state=config.RANDOM_STATE)
        pipeline = ImbPipeline([("resampler", resampler), ("model", base)])
    else:
        pipeline = ImbPipeline([("model", base)])

    # Prefix grid keys with model__
    piped_grid = {f"model__{k}": v for k, v in grid.items()}

    print(f"  Tuning {strategy_name} x {model_type} via 5-fold CV ...")
    gs = GridSearchCV(
        pipeline,
        piped_grid,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"  [Tuned] Best CV AUPRC: {gs.best_score_:.4f} | Best params: {gs.best_params_}")

    # Extract fitted estimator if user needs standalone model, or return fitted pipeline
    # The pipeline implements predict_proba and predict, making it fully compatible.
    return gs.best_estimator_


def save_model(model, name: str) -> Path:
    """Persist a fitted model or pipeline to models/<name>.joblib."""
    config.MODELS_DIR.mkdir(exist_ok=True)
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"  Saved model -> {path}")
    return path


def load_model(name: str):
    """Load a previously saved model from models/<name>.joblib."""
    path = config.MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)
