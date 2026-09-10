"""
tests/test_pipeline_correctness.py — Pipeline correctness & methodology tests.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.preprocessing import stratified_train_val_test_split, scale_train_val_test_features
from src.cost_analysis import run_cost_analysis
from src.train import tune_best_model
from src.autoencoder import run_autoencoder_pipeline


def make_dummy_dataset(n=500, fraud_rate=0.04, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    
    feature_names = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                     "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                     "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
    
    X_legit = rng.standard_normal((n_legit, 30))
    X_fraud = rng.standard_normal((n_fraud, 30)) + 1.5
    
    df_legit = pd.DataFrame(X_legit, columns=feature_names)
    df_legit["Class"] = 0
    df_legit["Amount"] = np.abs(df_legit["Amount"]) * 50 + 10

    df_fraud = pd.DataFrame(X_fraud, columns=feature_names)
    df_fraud["Class"] = 1
    df_fraud["Amount"] = np.abs(df_fraud["Amount"]) * 100 + 50

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return df, X, y


class DummyModel:
    def predict_proba(self, X):
        # Deterministic dummy scores based on feature V1
        scores = 1 / (1 + np.exp(-X["V1"]))
        return np.column_stack([1 - scores, scores])


def test_validation_only_threshold_optimization():
    """Verify that threshold optimization uses validation predictions and training FN cost only."""
    df, X, y = make_dummy_dataset()
    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_train_val_test_split(X, y)
    
    model = DummyModel()
    res = run_cost_analysis(model, X_val, y_val, df_raw=df, train_indices=X_tr.index)
    
    assert "optimal_threshold" in res
    assert 0.01 <= res["optimal_threshold"] <= 0.99
    assert res["split"] == "validation"


def test_autoencoder_validation_isolation():
    """Verify that Autoencoder uses legitimate training rows for fitting and validation for thresholding."""
    df, X, y = make_dummy_dataset(n=300)
    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_train_val_test_split(X, y)
    X_tr_sc, X_val_sc, X_te_sc, scaler = scale_train_val_test_features(X_tr, X_val, X_te)

    metrics, ae, threshold = run_autoencoder_pipeline(
        X_tr_sc, X_te_sc, y_tr, y_te, X_val=X_val_sc, y_val=y_val
    )

    assert "auprc" in metrics
    assert metrics["split"] == "test"
    assert threshold > 0.0


def test_pipeline_cv_tuning_smoke():
    """Smoke test for leak-free pipeline tuning with tiny grid."""
    df, X, y = make_dummy_dataset(n=200)
    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_train_val_test_split(X, y)
    
    tiny_grid = {"n_estimators": [5, 10], "max_depth": [2]}
    best_pipe = tune_best_model(
        X_tr, y_tr, model_type="RF", strategy_name="class_weight", param_grid=tiny_grid
    )
    
    probs = best_pipe.predict_proba(X_val)
    assert probs.shape == (len(X_val), 2)
