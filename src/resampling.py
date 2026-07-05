"""
src/resampling.py — Imbalance-handling strategies (training fold only).

⚠️  LEAKAGE GUARD: Every function here accepts ONLY (X_train, y_train).
    Resampling is applied after the train/test split and never touches
    X_test or y_test. This is enforced by the API design: these functions
    don't even receive test data.

    Why does this matter?
    If you SMOTE-oversample the full dataset before splitting, synthetic
    minority samples that were used to generate test-set points will also
    appear in training. The model effectively "sees" the test set, leading
    to inflated recall figures. The fix is trivial — resample after splitting
    — but it's a classic mistake worth calling out explicitly.

Strategies implemented:
1.  class_weight_only   — no resampling; pass class_weight to the model instead
2.  random_undersample  — randomly drop majority class rows to 1:1 ratio
3.  smote               — Synthetic Minority Over-sampling Technique
4.  smote_tomek         — SMOTE + Tomek link cleaning (removes borderline pairs)
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def class_weight_only(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    No resampling — return the training fold unchanged.

    The imbalance is handled by passing class_weight='balanced' to the model.
    This is the simplest strategy and a good baseline for comparison.

    Mathematically, class_weight='balanced' multiplies each sample's loss by
    n_samples / (n_classes * n_samples_in_class_i), so the minority class
    contributes equal total weight to the loss as the majority class.
    """
    # Return unchanged — imbalance handled via model's class_weight parameter
    return X_train, y_train


def random_undersample(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Randomly remove majority-class (legit) samples until classes are balanced.

    Pro: fast, reduces training time dramatically.
    Con: discards potentially useful information from the majority class.
    Use when: the majority class has abundant redundant samples.
    """
    rus = RandomUnderSampler(random_state=config.RANDOM_STATE)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    _log_resample("RandomUnderSampler", y_train, y_res)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Synthetic Minority Over-sampling Technique (Chawla et al., 2002).

    Mechanism: for each minority sample, pick k=5 nearest minority neighbours,
    then interpolate a new synthetic point somewhere along the line between
    the sample and one of its neighbours. This creates plausible (but not real)
    fraud transactions in the feature space.

    Pro: retains all real data; richer signal than pure duplication.
    Con: synthetic points may land in ambiguous regions; can overfit if
         minority clusters are tight. Always oversample only on training data.
    """
    sm = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    _log_resample("SMOTE", y_train, y_res)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


from imblearn.under_sampling import TomekLinks

def smote_tomek(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    SMOTE + Tomek link cleaning.

    After SMOTE oversampling, Tomek links identify pairs of samples from
    different classes that are each other's nearest neighbours — i.e., they
    sit right on the decision boundary and are likely noisy or ambiguous.
    Those pairs are removed from the majority class (and optionally minority).

    This two-phase approach both adds minority samples AND cleans the boundary,
    giving the classifier a cleaner margin to learn from.
    """
    smt = SMOTETomek(tomek=TomekLinks(n_jobs=-1), random_state=config.RANDOM_STATE)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    
    # Calculate exactly how many Tomek links were removed
    majority_count = int((y_train == config.LEGIT_LABEL).sum())
    expected_smote_size = 2 * majority_count
    tomek_removed = expected_smote_size - len(y_res)
    print(f"  [Tomek Cleaning] Removed {tomek_removed:,} borderline Tomek links from the oversampled set.")
    
    _log_resample("SMOTETomek", y_train, y_res)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def _log_resample(name: str, y_before: pd.Series, y_after) -> None:
    """Print before/after class distribution."""
    before_fraud = int(y_before.sum())
    after_fraud = int(sum(y_after == config.FRAUD_LABEL))
    after_legit = int(sum(y_after == config.LEGIT_LABEL))
    print(
        f"[{name}] Before: {before_fraud} fraud / {len(y_before)-before_fraud} legit "
        f"-> After: {after_fraud} fraud / {after_legit} legit "
        f"(total {len(y_after):,})"
    )


# Registry mapping strategy name → function (used by run_pipeline.py)
RESAMPLING_STRATEGIES = {
    "class_weight": class_weight_only,
    "undersample": random_undersample,
    "smote": smote,
    "smote_tomek": smote_tomek,
}
