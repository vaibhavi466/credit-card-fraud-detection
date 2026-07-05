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
from src.preprocessing import stratified_split, scale_features
import config


def make_dummy_data(n=1000, fraud_rate=0.002, seed=0):
    """Create a small synthetic dataset with the same structure as creditcard.csv."""
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n * fraud_rate))
    n_legit = n - n_fraud

    X = pd.DataFrame(
        rng.standard_normal((n, 30)),
        columns=["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
    )
    y = pd.Series([1] * n_fraud + [0] * n_legit, name="Class")
    # Shuffle
    idx = rng.permutation(n)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


class TestStratifiedSplit:
    def test_split_sizes(self):
        X, y = make_dummy_data(n=1000)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        assert len(X_train) + len(X_test) == len(X)
        assert abs(len(X_test) / len(X) - config.TEST_SIZE) < 0.01

    def test_stratification_preserves_fraud_rate(self):
        """Both folds should have approximately the same fraud rate."""
        X, y = make_dummy_data(n=5000, fraud_rate=0.01)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        overall_rate = y.mean()
        assert abs(y_train.mean() - overall_rate) < 0.005
        assert abs(y_test.mean() - overall_rate) < 0.005

    def test_no_overlap_between_splits(self):
        """Train and test indices must be disjoint."""
        X, y = make_dummy_data(n=500)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(test_idx), "Train and test sets share indices!"


class TestScaleFeatures:
    def test_scaler_fitted_on_train_only(self):
        """
        LEAKAGE GUARD TEST: the scaler's mean and std must match the training
        set statistics, not the full dataset statistics.
        """
        X, y = make_dummy_data(n=1000)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

        # The scaler's mean_ should match the train-set mean exactly
        for i, col in enumerate(config.COLS_TO_SCALE):
            expected_mean = X_train[col].mean()
            actual_mean = scaler.mean_[i]
            assert abs(expected_mean - actual_mean) < 1e-9, (
                f"Scaler mean for {col} ({actual_mean:.6f}) doesn't match "
                f"train mean ({expected_mean:.6f}). Possible leakage!"
            )

    def test_scaler_does_not_use_test_statistics(self):
        """
        If the scaler used test-set statistics, the scaled test mean would be
        ~0. Using train statistics, the scaled test mean can be non-zero.
        We verify the scaler transform was applied with train (not test) params.
        """
        # Create two very different distributions
        X_train = pd.DataFrame({"Amount": [1.0, 2.0, 3.0], "Time": [10.0, 20.0, 30.0]})
        X_test = pd.DataFrame({"Amount": [100.0, 200.0, 300.0], "Time": [1000.0, 2000.0, 3000.0]})

        _, X_test_sc, scaler = scale_features(X_train, X_test)

        # Scaler mean should equal train mean (~2.0 for Amount)
        assert abs(scaler.mean_[0] - 2.0) < 0.01
        # Test set scaled values should be large (since test >> train mean)
        assert X_test_sc["Amount"].mean() > 50, (
            "Test data scaling used wrong statistics — possible leakage!"
        )

    def test_scaled_train_mean_near_zero(self):
        """Scaled training features should have mean ≈ 0."""
        X, y = make_dummy_data(n=500)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        X_train_sc, _, _ = scale_features(X_train, X_test)

        for col in config.COLS_TO_SCALE:
            assert abs(X_train_sc[col].mean()) < 0.01, (
                f"Scaled train mean for {col} is not near zero: {X_train_sc[col].mean()}"
            )

    def test_output_shapes_unchanged(self):
        X, y = make_dummy_data(n=300)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
        assert X_train_sc.shape == X_train.shape
        assert X_test_sc.shape == X_test.shape
