"""
src/autoencoder.py — Unsupervised fraud detection via reconstruction error.

Why an autoencoder for fraud detection?
The supervised approach requires labeled fraud examples.
In many real-world scenarios:
  - You have abundant legitimate transaction data but few or no labeled frauds
  - The fraud patterns change over time (concept drift), making old labels stale
  - A newly launched product has zero fraud history

An autoencoder trained ONLY on legitimate transactions learns the "normal"
transaction manifold. When a fraud transaction passes through it, the
reconstruction is poor (high MSE) because the autoencoder has never seen
that pattern. We threshold this reconstruction error to flag anomalies.

Approach: We use sklearn's MLPRegressor as an autoencoder. This avoids adding
TensorFlow/PyTorch as a heavy dependency while still demonstrating the concept.
The architecture is an encoder-decoder: 29 → 32 → 16 → 8 → 16 → 32 → 29.
The bottleneck layer (8 units) forces the model to learn a compressed
representation — the "essence" of a normal transaction.

Limitations:
- MLPRegressor is not as flexible as a true DL autoencoder (no GPU, no dropout)
- Performance will be lower than the supervised model (that's expected and is
  part of the comparison story)
- The reconstruction threshold is sensitive; we tune it on a validation set
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
)
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_autoencoder() -> MLPRegressor:
    """
    Build an MLPRegressor configured as an autoencoder.

    The hidden_layer_sizes form an encoder-decoder bottleneck:
    input(29) → 32 → 16 → 8 → 16 → 32 → output(29)

    We use tanh activation (bounded output, good for PCA-scaled features) and
    adam optimizer. max_iter=20 (from config) is sufficient for convergence.
    """
    return MLPRegressor(
        hidden_layer_sizes=config.AUTOENCODER_HIDDEN_LAYER_SIZES,
        activation="tanh",
        solver="adam",
        max_iter=config.AUTOENCODER_MAX_ITER,
        random_state=config.RANDOM_STATE,
        verbose=False,
    )


def train_autoencoder(X_train: pd.DataFrame, y_train: pd.Series) -> MLPRegressor:
    """
    Train the autoencoder ONLY on non-fraud (legitimate) training samples.

    This is the key design decision: the model never sees fraud during training,
    so it learns what "normal" looks like. High reconstruction error = anomaly.
    """
    # Filter to legitimate transactions only
    legit_mask = y_train == config.LEGIT_LABEL
    X_legit = X_train[legit_mask]
    print(f"Training autoencoder on {len(X_legit):,} legitimate transactions …")

    ae = build_autoencoder()
    ae.fit(X_legit, X_legit)
    print(f"Autoencoder trained. Loss: {ae.loss_:.6f}")
    return ae


def reconstruction_error(ae: MLPRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    Compute per-sample Mean Squared Reconstruction Error.

    MSE = mean((x - ae(x))^2) over features for each sample.
    High MSE → the autoencoder struggled to reconstruct this sample → likely anomalous.
    """
    X_reconstructed = ae.predict(X)
    mse = np.mean((X.values - X_reconstructed) ** 2, axis=1)
    return mse


def find_threshold_from_validation(
    ae: MLPRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    percentile: float = 95,
) -> float:
    """
    Set reconstruction error threshold as the Nth percentile of legit-only errors.

    Logic: if we set the threshold at the 95th percentile of legitimate transaction
    errors, then only the top 5% most "unusual" legitimate transactions will be
    flagged.
    """
    legit_mask = y_val == config.LEGIT_LABEL
    legit_errors = reconstruction_error(ae, X_val[legit_mask])
    threshold = float(np.percentile(legit_errors, percentile))
    print(f"Autoencoder threshold (p{percentile} of val legit errors): {threshold:.6f}")
    return threshold


def evaluate_autoencoder(
    ae: MLPRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """Evaluate the autoencoder as a binary classifier using reconstruction error."""
    errors = reconstruction_error(ae, X_test)
    y_pred = (errors >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auprc = average_precision_score(y_test, errors)
    roc_auc = roc_auc_score(y_test, errors)

    metrics = {
        "model_name": "Autoencoder (unsupervised)",
        "split": "test",
        "purpose": "final_evaluation",
        "threshold": round(threshold, 6),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auprc": round(auprc, 4),
        "roc_auc": round(roc_auc, 4),
    }
    print(
        f"[Autoencoder | test] P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
        f"AUPRC={auprc:.4f} ROC-AUC={roc_auc:.4f}"
    )
    return metrics


def run_autoencoder_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
) -> tuple[dict, MLPRegressor, float]:
    """
    Full autoencoder pipeline: train on X_train -> threshold on X_val -> evaluate on X_test.
    """
    ae = train_autoencoder(X_train, y_train)

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(ae, config.MODELS_DIR / "autoencoder.joblib")

    if X_val is None or y_val is None:
        # Fallback to internal split if X_val not passed
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=config.RANDOM_STATE,
        )

    threshold = find_threshold_from_validation(ae, X_val, y_val, percentile=95)
    metrics = evaluate_autoencoder(ae, X_test, y_test, threshold)
    metrics["threshold_percentile"] = 95

    return metrics, ae, threshold
