"""
tests/test_evaluate.py — Verify metric functions against known examples.

We use toy classifiers with known ground-truth outcomes to verify that our
evaluation functions produce correct numbers before we run them on the real data.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluate import evaluate_model, evaluate_all_thresholds


class PerfectClassifier:
    """Trivial classifier that always predicts the correct label."""
    def fit(self, X, y):
        self.classes_ = [0, 1]
        return self

    def predict_proba(self, X):
        # Return probability 1.0 for the true class — we set y_test externally
        # For testing purposes, use the first column as the "fraud probability"
        n = len(X)
        proba = np.column_stack([np.zeros(n), np.ones(n)])
        return proba

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class AllLegitClassifier:
    """Always predicts legit (0) — simulates the accuracy trap."""
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.ones(n), np.zeros(n)])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestEvaluateModel:
    def setup_method(self):
        """Create a small test set: 90 legit + 10 fraud."""
        rng = np.random.default_rng(0)
        self.X_test = pd.DataFrame(rng.standard_normal((100, 5)))
        self.y_test = pd.Series([1] * 10 + [0] * 90)

    def test_all_legit_classifier_zero_recall(self):
        """A model that never predicts fraud must have recall=0 for fraud class."""
        clf = AllLegitClassifier()
        metrics = evaluate_model(clf, self.X_test, self.y_test)
        assert metrics["recall"] == 0.0, (
            f"All-legit classifier should have recall=0, got {metrics['recall']}"
        )

    def test_all_legit_classifier_has_high_accuracy_but_zero_f1(self):
        """This is the accuracy trap: 90% accuracy, 0 F1."""
        clf = AllLegitClassifier()
        metrics = evaluate_model(clf, self.X_test, self.y_test)
        assert metrics["f1"] == 0.0
        # True accuracy would be 0.90 — demonstrate why it's misleading
        y_pred_legit = np.zeros(100)
        naive_accuracy = (y_pred_legit == self.y_test.values).mean()
        assert naive_accuracy == 0.90

    def test_threshold_parameter_affects_predictions(self):
        """Lower threshold should increase recall at cost of precision."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((200, 5)))
        y = pd.Series([1] * 20 + [0] * 180)

        # A classifier with some signal
        class SignalClassifier:
            def predict_proba(self, X):
                n = len(X)
                # Higher probability for first 20 samples (fraud)
                proba = np.random.default_rng(1).uniform(0, 0.4, n)
                proba[:20] = np.random.default_rng(2).uniform(0.5, 1.0, 20)
                return np.column_stack([1 - proba, proba])

        clf = SignalClassifier()
        m_high = evaluate_model(clf, X, y, threshold=0.8)
        m_low = evaluate_model(clf, X, y, threshold=0.2)

        # Lower threshold → more fraud flagged → higher recall, lower precision
        assert m_low["recall"] >= m_high["recall"]
        assert m_low["precision"] <= m_high["precision"]

    def test_metrics_in_valid_range(self):
        """All metrics must be in [0, 1]."""
        clf = AllLegitClassifier()
        metrics = evaluate_model(clf, self.X_test, self.y_test)
        for key in ["precision", "recall", "f1", "roc_auc", "auprc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} = {metrics[key]} out of [0,1]"

    def test_confusion_matrix_sums_to_total(self):
        clf = AllLegitClassifier()
        metrics = evaluate_model(clf, self.X_test, self.y_test)
        total = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
        assert total == len(self.y_test)


class TestEvaluateAllThresholds:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 5)))
        y = pd.Series([1] * 10 + [0] * 90)
        clf = AllLegitClassifier()
        df = evaluate_all_thresholds(clf, X, y)
        assert "threshold" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns
        assert len(df) == 99  # THRESHOLD_SWEEP has 99 points

    def test_threshold_monotone_recall(self):
        """As threshold increases, recall should generally decrease (monotone non-increasing)."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((500, 5)))
        y = pd.Series([1] * 50 + [0] * 450)

        class GoodClassifier:
            def predict_proba(self, X):
                n = len(X)
                p = np.zeros(n)
                p[:50] = 0.9  # correct fraud signal
                p[50:] = 0.1
                return np.column_stack([1 - p, p])

        df = evaluate_all_thresholds(GoodClassifier(), X, y)
        # Recall at threshold=0.1 should be >= recall at threshold=0.9
        r_low = df[df["threshold"] < 0.15]["recall"].max()
        r_high = df[df["threshold"] > 0.85]["recall"].min()
        assert r_low >= r_high
