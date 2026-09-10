"""
src/preprocessing.py — Feature scaling and 3-way stratified train/validation/test split.

⚠️  LEAKAGE GUARD — this is the most important design decision in the pipeline:

    The StandardScaler is fitted ONLY on X_train, then applied to X_train,
    X_val, and X_test using transform() (not fit_transform()).

    Why does this matter?
    If you fit the scaler on the full dataset (or on validation/test sets),
    the scaler "sees" future statistics (mean, std) during training. The model
    then indirectly benefits from future information — this is data leakage.
    On tabular data it often causes an optimistic bias of 1–5 AUPRC points,
    making your validation/test look better than real-world performance.

    This is one of the most common interview trap questions:
    "Walk me through how you handled the train/validation/test split and preprocessing."
    The correct answer is: fit on train fold only, transform all splits — never fit on full data.

Columns scaled: V1–V28 are PCA-transformed/anonymised features. This project
scales the raw Amount and Time features while preserving the provided PCA-transformed
components.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def stratified_train_val_test_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform a 3-way stratified split into Train (60%), Validation (20%), and Test (20%).

    Sequence:
    1. Split complete dataset into train_val (80%) and test (20%).
    2. Split train_val into train (75% of 80% = 60%) and validation (25% of 80% = 20%).

    Stratification ensures fraud prevalence (≈0.17%) is preserved across all three folds.
    Original DataFrame index traceability is preserved.
    """
    # First split: 80% train_val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    # Second split: 75% train (60% overall), 25% val (20% overall)
    val_size = getattr(config, "VALIDATION_SIZE_WITHIN_TRAIN", 0.25)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        stratify=y_train_val,
        random_state=config.RANDOM_STATE,
    )

    _log_3way_split_info(y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _log_3way_split_info(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> None:
    """Print sanity-check summary of 3-way split fraud rates."""
    tr_f, val_f, te_f = y_train.sum(), y_val.sum(), y_test.sum()
    n_tot = len(y_train) + len(y_val) + len(y_test)
    print(
        f"3-Way Split -> Train: {len(y_train):,} ({len(y_train)/n_tot*100:.1f}%, {tr_f} fraud, {tr_f/len(y_train)*100:.4f}%) | "
        f"Val: {len(y_val):,} ({len(y_val)/n_tot*100:.1f}%, {val_f} fraud, {val_f/len(y_val)*100:.4f}%) | "
        f"Test: {len(y_test):,} ({len(y_test)/n_tot*100:.1f}%, {te_f} fraud, {te_f/len(y_test)*100:.4f}%)"
    )


def stratified_split(X: pd.DataFrame, y: pd.Series):
    """
    Backwards-compatible stratified 80/20 train/test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    _log_split_info(y_train, y_test)
    return X_train, X_test, y_train, y_test


def _log_split_info(y_train: pd.Series, y_test: pd.Series) -> None:
    """Print sanity-check summary of 2-way split fraud rates."""
    train_fraud = y_train.sum()
    test_fraud = y_test.sum()
    print(
        f"Split -> train: {len(y_train):,} rows ({train_fraud} fraud, "
        f"{train_fraud/len(y_train)*100:.4f}%) | "
        f"test: {len(y_test):,} rows ({test_fraud} fraud, "
        f"{test_fraud/len(y_test)*100:.4f}%)"
    )


def scale_train_val_test_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cols_to_scale: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on X_train only, then transform X_train, X_val, and X_test.

    LEAKAGE GUARD: The scaler is fitted exclusively on the training fold.
    Calling .transform() on X_val and X_test applies the train statistics without
    exposing validation/test statistics to the scaler.
    """
    if cols_to_scale is None:
        cols_to_scale = config.COLS_TO_SCALE

    scaler = StandardScaler()
    scaler.fit(X_train[cols_to_scale])

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print(
        f"Scaled {cols_to_scale} using train-set mean="
        f"{scaler.mean_.round(2).tolist()}, std={scaler.scale_.round(2).tolist()}"
    )
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols_to_scale: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Backwards-compatible scale_features for 2-way splits.
    Fit StandardScaler on X_train only, then transform both sets.
    """
    if cols_to_scale is None:
        cols_to_scale = config.COLS_TO_SCALE

    scaler = StandardScaler()
    scaler.fit(X_train[cols_to_scale])

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, scaler


def preprocess_train_val_test(X: pd.DataFrame, y: pd.Series):
    """
    Full 3-way preprocessing pipeline: split (60/20/20) then scale.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(X, y)
    X_train, X_val, X_test, scaler = scale_train_val_test_features(X_train, X_val, X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def preprocess(X: pd.DataFrame, y: pd.Series):
    """
    Backwards-compatible 2-way preprocessing pipeline.
    """
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    return X_train, X_test, y_train, y_test, scaler
