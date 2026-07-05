"""
src/data_loader.py — Load and validate the raw creditcard CSV.

Design decisions:
- Validation is strict: row count and fraud count must match the ULB dataset
  exactly. This ensures we always know what data we're working with and
  prevents silent failures if someone drops the wrong file into data/raw/.
- We do NOT do any feature engineering or scaling here — that belongs in
  preprocessing.py. Keeping I/O separate from transformation makes each step
  independently testable.
"""

import pandas as pd
import sys
from pathlib import Path

# Allow running this file directly from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_raw_data(csv_path: Path = config.RAW_CSV) -> pd.DataFrame:
    """
    Load the raw creditcard CSV and validate it against known dataset specs.

    Parameters
    ----------
    csv_path : Path
        Path to creditcard.csv. Defaults to config.RAW_CSV.

    Returns
    -------
    pd.DataFrame
        The raw DataFrame, exactly as downloaded — no transformations applied.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at csv_path.
    ValueError
        If row count or fraud count don't match the expected ULB dataset values.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n\n  [ERROR] Dataset not found at: {csv_path}\n\n"
            "  To download it:\n"
            "    1. Create a Kaggle account and go to Account -> Create New API Token\n"
            "    2. Place kaggle.json at: C:\\Users\\<you>\\.kaggle\\kaggle.json\n"
            "    3. pip install kaggle\n"
            "    4. kaggle datasets download -d mlg-ulb/creditcardfraud "
            "-p data/raw --unzip\n"
        )

    print(f"Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path)

    # ── Validate row count ────────────────────────────────────────────────────
    n_rows = len(df)
    if n_rows != config.EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Row count mismatch: expected {config.EXPECTED_ROW_COUNT:,}, "
            f"got {n_rows:,}.\n"
            "This does not look like the standard ULB creditcard dataset. "
            "Stopping to avoid silently using wrong data."
        )

    # ── Validate fraud count ──────────────────────────────────────────────────
    n_fraud = int(df[config.TARGET_COL].sum())
    if n_fraud != config.EXPECTED_FRAUD_COUNT:
        raise ValueError(
            f"Fraud count mismatch: expected {config.EXPECTED_FRAUD_COUNT}, "
            f"got {n_fraud}.\n"
            "This does not look like the standard ULB creditcard dataset. "
            "Stopping to avoid silently using wrong data."
        )

    fraud_pct = n_fraud / n_rows * 100
    print(
        f"[OK] Loaded {n_rows:,} rows | "
        f"{n_fraud} fraud ({fraud_pct:.4f}%) | "
        f"{n_rows - n_fraud:,} legitimate"
    )
    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Split the DataFrame into feature matrix X and target vector y.

    Returns
    -------
    X : pd.DataFrame  — all columns except Class
    y : pd.Series     — the Class column (0 = legit, 1 = fraud)
    """
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]
    return X, y


if __name__ == "__main__":
    df = load_raw_data()
    X, y = get_features_and_target(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(df.describe())
