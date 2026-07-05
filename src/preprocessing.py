"""
src/preprocessing.py — Feature scaling and stratified train/test split.

⚠️  LEAKAGE GUARD — this is the most important design decision in the pipeline:

    The StandardScaler is fitted ONLY on X_train, then applied to both
    X_train and X_test using transform() (not fit_transform()).

    Why does this matter?
    If you fit the scaler on the full dataset (or on X_test), the scaler
    "sees" test-set statistics (mean, std) during training. The model then
    indirectly benefits from future information — this is data leakage.
    On tabular data it often causes an optimistic bias of 1–5 AUPRC points,
    making your cross-validation look better than real-world performance.

    This is one of the most common interview trap questions:
    "Walk me through how you handled the train/test split and preprocessing."
    The correct answer is: fit on train, transform both — never fit on full data.

Columns scaled: only Amount and Time (V1–V28 are already PCA-standardised
by the dataset authors, so re-scaling them changes nothing useful).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def stratified_split(X: pd.DataFrame, y: pd.Series):
    """
    Perform a stratified 80/20 train/test split.

    Stratification ensures the fraud rate (≈0.17%) is preserved in both
    the train and test folds. Without stratification, a random 20% test set
    could easily contain zero fraud cases.

    Parameters
    ----------
    X : pd.DataFrame — feature matrix
    y : pd.Series    — target vector (0/1)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,           # preserve fraud prevalence in both folds
        random_state=config.RANDOM_STATE,
    )
    _log_split_info(y_train, y_test)
    return X_train, X_test, y_train, y_test


def _log_split_info(y_train: pd.Series, y_test: pd.Series) -> None:
    """Print a quick sanity-check summary of the split fraud rates."""
    train_fraud = y_train.sum()
    test_fraud = y_test.sum()
    print(
        f"Split -> train: {len(y_train):,} rows ({train_fraud} fraud, "
        f"{train_fraud/len(y_train)*100:.4f}%) | "
        f"test: {len(y_test):,} rows ({test_fraud} fraud, "
        f"{test_fraud/len(y_test)*100:.4f}%)"
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols_to_scale: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on X_train only, then transform both sets.

    LEAKAGE GUARD: The scaler is fitted exclusively on the training fold.
    Calling .transform() on X_test applies the train statistics without
    exposing test statistics to the scaler.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    cols_to_scale : list, optional
        Column names to scale. Defaults to config.COLS_TO_SCALE (Amount, Time).

    Returns
    -------
    X_train_scaled : pd.DataFrame  (same index as input)
    X_test_scaled  : pd.DataFrame  (same index as input)
    scaler         : fitted StandardScaler (for persistence / inspection)
    """
    if cols_to_scale is None:
        cols_to_scale = config.COLS_TO_SCALE

    scaler = StandardScaler()

    # ── FIT on train only ─────────────────────────────────────────────────────
    scaler.fit(X_train[cols_to_scale])

    # ── TRANSFORM both sets using train statistics ────────────────────────────
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print(
        f"Scaled {cols_to_scale} using train-set mean="
        f"{scaler.mean_.round(2).tolist()}, std={scaler.scale_.round(2).tolist()}"
    )
    return X_train_scaled, X_test_scaled, scaler


def preprocess(X: pd.DataFrame, y: pd.Series):
    """
    Full preprocessing pipeline: split then scale.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    return X_train, X_test, y_train, y_test, scaler
