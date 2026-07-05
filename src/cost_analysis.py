"""
src/cost_analysis.py — Cost-sensitive threshold selection.

Why this matters for interviews:
Defaulting to threshold=0.5 is arbitrary and rarely optimal. In fraud detection,
the business cares about total dollar cost, not abstract F1 scores. By assigning
costs to errors, we can choose the threshold that minimises expected financial loss.

Cost model:
- False Negative (FN): we miss a real fraud → cost = mean fraud transaction amount
  (computed from the actual data at runtime — see IMPORTANT note below)
- False Positive (FP): we flag a legitimate transaction → cost = $5 placeholder
  (this is ILLUSTRATIVE — in reality it captures customer friction, call-centre
  costs, card reissuance costs, etc. A real number would come from the business.)

The optimal threshold is the one that minimises:
    total_cost(t) = FN(t) × fn_cost + FP(t) × fp_cost

IMPORTANT: fn_cost is computed from the real data (mean fraud amount), not hardcoded.
           This ensures the cost plot is calibrated to the actual dataset statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def compute_mean_fraud_amount(df: pd.DataFrame) -> float:
    """
    Compute the mean transaction Amount for fraud cases.

    This is used as the False Negative cost — if we miss a fraud, we assume
    the expected loss is the mean historical fraud amount.

    Note: in reality, amounts are not uniform across fraud types. A more
    sophisticated model might use per-transaction amounts instead of the mean.
    But the mean is defensible and avoids baking in an arbitrary assumption.
    """
    fraud_amounts = df.loc[df[config.TARGET_COL] == config.FRAUD_LABEL, "Amount"]
    mean_amount = float(fraud_amounts.mean())
    print(f"Mean fraud transaction amount (FN cost): ${mean_amount:.2f}")
    return mean_amount


def compute_cost_curve(
    y_test: pd.Series,
    y_proba: np.ndarray,
    fn_cost: float,
    fp_cost: float = config.FP_COST_DOLLARS,
) -> pd.DataFrame:
    """
    Sweep classification thresholds and compute total expected cost at each.

    Parameters
    ----------
    y_test   : true labels (test set)
    y_proba  : predicted fraud probabilities
    fn_cost  : cost per False Negative (missed fraud) — use mean fraud amount
    fp_cost  : cost per False Positive (falsely flagged) — illustrative ($5)

    Returns
    -------
    DataFrame with columns: threshold, fn_count, fp_count, total_cost
    """
    rows = []
    for t in config.THRESHOLD_SWEEP:
        y_pred = (y_proba >= t).astype(int)
        # Count errors at this threshold
        fn = int(((y_pred == 0) & (y_test == 1)).sum())   # missed frauds
        fp = int(((y_pred == 1) & (y_test == 0)).sum())   # false alarms
        total_cost = fn * fn_cost + fp * fp_cost
        rows.append({
            "threshold": round(float(t), 4),
            "fn_count": fn,
            "fp_count": fp,
            "fn_cost_total": round(fn * fn_cost, 2),
            "fp_cost_total": round(fp * fp_cost, 2),
            "total_cost": round(total_cost, 2),
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(cost_df: pd.DataFrame) -> float:
    """Return the threshold that minimises total expected cost."""
    idx = cost_df["total_cost"].idxmin()
    optimal = float(cost_df.loc[idx, "threshold"])
    min_cost = float(cost_df.loc[idx, "total_cost"])
    fn = int(cost_df.loc[idx, "fn_count"])
    fp = int(cost_df.loc[idx, "fp_count"])
    print(
        f"Optimal threshold (min cost): {optimal:.4f}  |  "
        f"Total cost: ${min_cost:,.2f}  |  FN={fn} FP={fp}"
    )
    return optimal


def plot_cost_curve(cost_df: pd.DataFrame, optimal_threshold: float, save=True) -> Path:
    """Save a cost curve plot showing total cost vs. threshold."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_total = "#7c3aed"
    ax1.plot(
        cost_df["threshold"], cost_df["total_cost"],
        color=color_total, lw=2.5, label="Total cost"
    )
    ax1.fill_between(
        cost_df["threshold"], cost_df["total_cost"],
        alpha=0.08, color=color_total
    )
    ax1.axvline(
        optimal_threshold, ls="--", color="orange", lw=2,
        label=f"Optimal threshold = {optimal_threshold:.2f}"
    )
    ax1.set_xlabel("Classification Threshold", fontsize=11)
    ax1.set_ylabel("Total Expected Cost ($)", fontsize=11, color=color_total)
    ax1.tick_params(axis="y", labelcolor=color_total)

    # Overlay FN and FP cost components
    ax2 = ax1.twinx()
    ax2.plot(cost_df["threshold"], cost_df["fn_cost_total"], color="#dc2626",
             lw=1.5, ls=":", alpha=0.8, label="FN cost (missed fraud)")
    ax2.plot(cost_df["threshold"], cost_df["fp_cost_total"], color="#2563eb",
             lw=1.5, ls=":", alpha=0.8, label="FP cost (false alarms)")
    ax2.set_ylabel("Component Cost ($)", fontsize=10, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    ax1.set_title(
        f"Cost-Sensitive Threshold Selection\n"
        f"FP cost = ${config.FP_COST_DOLLARS:.0f} (illustrative) | "
        f"FN cost = mean fraud amount (from data)",
        fontsize=12,
    )
    ax1.grid(alpha=0.3)
    plt.tight_layout()

    path = config.FIGURES_DIR / "cost_curve.png"
    if save:
        config.FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"  Saved -> {path}")
    plt.close(fig)
    return path


def run_cost_analysis(model, X_test, y_test, df_raw: pd.DataFrame) -> dict:
    """
    Full cost analysis: compute curve, find optimal threshold, save plot.

    Returns dict with optimal_threshold and min_total_cost.
    """
    fn_cost = compute_mean_fraud_amount(df_raw)
    y_proba = model.predict_proba(X_test)[:, 1]

    cost_df = compute_cost_curve(y_test, y_proba, fn_cost=fn_cost)
    optimal_threshold = find_optimal_threshold(cost_df)
    plot_cost_curve(cost_df, optimal_threshold)

    return {
        "fn_cost_per_transaction": round(fn_cost, 2),
        "fp_cost_per_transaction": config.FP_COST_DOLLARS,
        "optimal_threshold": round(optimal_threshold, 4),
        "min_total_cost": float(cost_df["total_cost"].min()),
    }
