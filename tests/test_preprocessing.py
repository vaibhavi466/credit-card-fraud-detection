"""
tests/test_preprocessing.py — Verify leakage guard and split correctness.

The most important thing we test here is that the StandardScaler is fitted
ONLY on training data. This is the no-leakage guarantee.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import (
    stratified_split,
    scale_features,
    stratified_train_val_test_split,
    scale_train_val_test_features,
)
import config


def make_dummy_data(n=1000, fraud_rate=0.01, seed=0):
    """Create a small synthetic dataset with the same structure as creditcard.csv."""
    rng = np.random.default_rng(seed)
    n_fraud = max(4, int(n * fraud_rate))
    n_legit = n - n_fraud

    X = pd.DataFrame(
        rng.standard_normal((n, 30)),
        columns=["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
    )
    y = pd.Series([1] * n_fraud + [0] * n_legit, name="Class")
    idx = rng.permutation(n)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


class TestStratifiedSplit:
    def test_split_sizes(self):
        X, y = make_dummy_data(n=1000)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        assert len(X_train) + len(X_test) == len(X)
        assert abs(len(X_test) / len(X) - config.TEST_SIZE) < 0.01

    def test_3way_split_sizes_and_completeness(self):
        """3-way split must sum to total rows and match ~60/20/20 proportions."""
        X, y = make_dummy_data(n=1000)
        X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_train_val_test_split(X, y)
        assert len(X_tr) + len(X_val) + len(X_te) == len(X)
        assert abs(len(X_te) / len(X) - 0.20) < 0.01
        assert abs(len(X_val) / len(X) - 0.20) < 0.01
        assert abs(len(X_tr) / len(X) - 0.60) < 0.01

    def test_3way_no_overlap_between_splits(self):
        """Train, validation, and test indices must be completely disjoint."""
        X, y = make_dummy_data(n=600)
        X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_train_val_test_split(X, y)
        tr_idx, val_idx, te_idx = set(X_tr.index), set(X_val.index), set(X_te.index)
        assert tr_idx.isdisjoint(val_idx), "Train and validation share indices!"
        assert tr_idx.isdisjoint(te_idx), "Train and test share indices!"
        assert val_idx.isdisjoint(te_idx), "Validation and test share indices!"

    def test_3way_stratification_preserves_fraud_rate(self):
        """All 3 folds should preserve the overall fraud prevalence."""
        X, y = make_dummy_data(n=5000, fraud_rate=0.02)
        X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_train_val_test_split(X, y)
        overall_rate = y.mean()
        assert abs(y_tr.mean() - overall_rate) < 0.01
        assert abs(y_val.mean() - overall_rate) < 0.01
        assert abs(y_te.mean() - overall_rate) < 0.01


class TestScaleFeatures:
    def test_scaler_fitted_on_train_only(self):
        """
        LEAKAGE GUARD TEST: the scaler's mean and std must match the training
        set statistics, not the full dataset statistics.
        """
        X, y = make_dummy_data(n=1000)
        X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_train_val_test_split(X, y)
        X_tr_sc, X_val_sc, X_te_sc, scaler = scale_train_val_test_features(X_tr, X_val, X_te)

        for i, col in enumerate(config.COLS_TO_SCALE):
            expected_mean = X_tr[col].mean()
            actual_mean = scaler.mean_[i]
            assert abs(expected_mean - actual_mean) < 1e-9, (
                f"Scaler mean for {col} doesn't match train mean. Leakage detected!"
            )

    def test_scaler_does_not_use_val_or_test_statistics(self):
        X_train = pd.DataFrame({"Amount": [1.0, 2.0, 3.0], "Time": [10.0, 20.0, 30.0]})
        X_val = pd.DataFrame({"Amount": [50.0, 60.0, 70.0], "Time": [500.0, 600.0, 700.0]})
        X_test = pd.DataFrame({"Amount": [100.0, 200.0, 300.0], "Time": [1000.0, 2000.0, 3000.0]})

        _, X_val_sc, X_test_sc, scaler = scale_train_val_test_features(X_train, X_val, X_test)

        assert abs(scaler.mean_[0] - 2.0) < 0.01
        assert X_val_sc["Amount"].mean() > 20
        assert X_test_sc["Amount"].mean() > 50

    def test_output_shapes_unchanged(self):
        X, y = make_dummy_data(n=300)
        X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_train_val_test_split(X, y)
        X_tr_sc, X_val_sc, X_te_sc, scaler = scale_train_val_test_features(X_tr, X_val, X_te)
        assert X_tr_sc.shape == X_tr.shape
        assert X_val_sc.shape == X_val.shape
        assert X_te_sc.shape == X_te.shape
