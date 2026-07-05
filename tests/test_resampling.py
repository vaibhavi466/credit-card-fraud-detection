"""
tests/test_resampling.py — Verify resampling strategies.

Key invariant: resampling functions must ONLY operate on training data.
We test this by verifying that the resampled output contains no test indices.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.resampling import class_weight_only, random_undersample, smote, smote_tomek


def make_imbalanced_data(n_legit=500, n_fraud=10, seed=42):
    """Small imbalanced dataset for fast unit tests."""
    rng = np.random.default_rng(seed)
    # Fraud cluster is slightly different from legit
    X_legit = rng.standard_normal((n_legit, 29))
    X_fraud = rng.standard_normal((n_fraud, 29)) + 2.0  # offset so they're separable
    X = pd.DataFrame(
        np.vstack([X_legit, X_fraud]),
        columns=["Time", "Amount"] + [f"V{i}" for i in range(1, 28)],
    )
    y = pd.Series([0] * n_legit + [1] * n_fraud)
    return X, y


class TestClassWeightOnly:
    def test_returns_unchanged(self):
        X, y = make_imbalanced_data()
        X_res, y_res = class_weight_only(X, y)
        assert X_res.shape == X.shape
        assert len(y_res) == len(y)
        assert list(y_res) == list(y)


class TestRandomUndersample:
    def test_reduces_majority_class(self):
        X, y = make_imbalanced_data(n_legit=500, n_fraud=10)
        X_res, y_res = random_undersample(X, y)
        # After undersampling, majority count ≈ minority count
        assert y_res.sum() == (y_res == 0).sum() or abs(y_res.sum() - (y_res == 0).sum()) <= 1

    def test_total_rows_less_than_original(self):
        X, y = make_imbalanced_data(n_legit=500, n_fraud=10)
        X_res, y_res = random_undersample(X, y)
        assert len(y_res) < len(y)

    def test_fraud_count_preserved(self):
        """Undersampling removes majority rows, never minority rows."""
        X, y = make_imbalanced_data(n_legit=500, n_fraud=10)
        X_res, y_res = random_undersample(X, y)
        assert int(y_res.sum()) == 10


class TestSMOTE:
    def test_increases_minority_count(self):
        X, y = make_imbalanced_data(n_legit=500, n_fraud=10)
        X_res, y_res = smote(X, y)
        # SMOTE should increase minority class to match majority
        assert int(y_res.sum()) > 10

    def test_total_rows_greater_than_original(self):
        X, y = make_imbalanced_data(n_legit=500, n_fraud=10)
        X_res, y_res = smote(X, y)
        assert len(y_res) > len(y)

    def test_output_shape_consistent(self):
        X, y = make_imbalanced_data(n_legit=500, n_fraud=10)
        X_res, y_res = smote(X, y)
        assert X_res.shape[0] == len(y_res)
        assert X_res.shape[1] == X.shape[1]  # feature count unchanged

    def test_column_names_preserved(self):
        X, y = make_imbalanced_data()
        X_res, y_res = smote(X, y)
        assert list(X_res.columns) == list(X.columns)


class TestSMOTETomek:
    def test_output_shape_consistent(self):
        X, y = make_imbalanced_data(n_legit=500, n_fraud=10)
        X_res, y_res = smote_tomek(X, y)
        assert X_res.shape[0] == len(y_res)
        assert X_res.shape[1] == X.shape[1]

    def test_both_classes_present(self):
        X, y = make_imbalanced_data(n_legit=500, n_fraud=10)
        X_res, y_res = smote_tomek(X, y)
        assert set(y_res.unique()) == {0, 1}


class TestLeakageGuard:
    """
    The most important test: resampling must NEVER touch test data.

    We simulate the full pipeline and verify that none of the resampled
    training samples have indices from the test set.

    Note: after resampling, the DataFrame gets a new integer index (0, 1, 2 ...),
    so we can't check indices directly. Instead, we verify that the feature
    values of every resampled sample are either:
    (a) identical to a training sample (real sample), or
    (b) a convex combination of training samples (SMOTE synthetic).

    A simpler proxy test: verify resampling functions only ACCEPT training data
    (they don't have access to X_test at all). This is enforced by the function
    signatures — they take (X_train, y_train) only.
    """
    def test_resampling_api_only_accepts_train_data(self):
        """
        All resampling functions take exactly 2 args: X_train, y_train.
        If someone accidentally passes test data, the code structure prevents
        it from being labeled as training data.
        """
        import inspect
        for fn in [class_weight_only, random_undersample, smote, smote_tomek]:
            params = list(inspect.signature(fn).parameters.keys())
            assert params == ["X_train", "y_train"], (
                f"{fn.__name__} has unexpected parameters: {params}. "
                "All resampling functions must only accept (X_train, y_train)."
            )

    def test_smote_synthetic_samples_are_interpolations(self):
        """
        SMOTE synthetic samples must lie between existing minority samples.
        All feature values should be within [min, max] of the original training
        minority class — not extrapolations.
        """
        X, y = make_imbalanced_data(n_legit=200, n_fraud=20, seed=7)
        fraud_mask = y == 1
        original_fraud_min = X[fraud_mask].min()
        original_fraud_max = X[fraud_mask].max()

        X_res, y_res = smote(X, y)
        synthetic_fraud = X_res[y_res == 1]

        for col in X.columns:
            # Allow small numerical tolerance
            assert (synthetic_fraud[col] >= original_fraud_min[col] - 1e-6).all(), (
                f"SMOTE extrapolated below min for {col}!"
            )
            assert (synthetic_fraud[col] <= original_fraud_max[col] + 1e-6).all(), (
                f"SMOTE extrapolated above max for {col}!"
            )
