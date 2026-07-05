"""
eda_script.py — EDA script that produces all Phase 1 plots.
This is called by run_pipeline.py but also saved as notebooks/01_eda.ipynb.

Key findings (to be verified after running):
1. Severe class imbalance: 492 fraud / 284,807 total = 0.172%
2. Fraud amounts are small on average vs. what you might expect
3. Certain V features (V14, V4, V11, V12) show the strongest separation
4. Time distribution shows fraud occurs across all hours (no obvious temporal pattern)
5. Legitimate transactions show bimodal time pattern (2 peak periods per day)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
sns.set_palette("husl")

FRAUD_COLOR = "#dc2626"
LEGIT_COLOR = "#2563eb"
FRAUD_ALPHA = 0.7


def run_eda(df: pd.DataFrame) -> dict:
    """Run full EDA and save plots. Returns dict of findings."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    findings = {}

    # ── Plot 1: Class distribution ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Linear scale
    class_counts = df["Class"].value_counts().sort_index()
    axes[0].bar(["Legitimate", "Fraud"], class_counts.values,
                color=[LEGIT_COLOR, FRAUD_COLOR], alpha=0.85, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Transaction Class Distribution", fontsize=13)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    # Log scale to show the minority class
    axes[1].bar(["Legitimate", "Fraud"], class_counts.values,
                color=[LEGIT_COLOR, FRAUD_COLOR], alpha=0.85, edgecolor="white", linewidth=1.5)
    axes[1].set_yscale("log")
    axes[1].set_title("Class Distribution (log scale)", fontsize=13)
    axes[1].set_ylabel("Count (log scale)")
    axes[1].text(0, class_counts[0] * 1.2,
                 f"{class_counts[0]:,}\n({class_counts[0]/len(df)*100:.2f}%)",
                 ha="center", fontsize=9)
    axes[1].text(1, class_counts[1] * 1.2,
                 f"{class_counts[1]}\n({class_counts[1]/len(df)*100:.4f}%)",
                 ha="center", fontsize=9)

    plt.suptitle("ULB Credit Card Fraud Dataset", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    findings["fraud_count"] = int(len(fraud))
    findings["legit_count"] = int(len(legit))
    findings["fraud_pct"] = float(len(fraud) / len(df) * 100)
    print(f"  [EDA] Class distribution saved. Fraud: {findings['fraud_count']} ({findings['fraud_pct']:.4f}%)")

    # ── Plot 2: Amount distribution by class ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(legit["Amount"].clip(upper=2500), bins=60, color=LEGIT_COLOR,
                 alpha=0.7, label="Legit", density=True)
    axes[0].hist(fraud["Amount"].clip(upper=2500), bins=60, color=FRAUD_COLOR,
                 alpha=0.7, label="Fraud", density=True)
    axes[0].set_xlabel("Transaction Amount ($, clipped at $2,500)", fontsize=10)
    axes[0].set_ylabel("Density")
    axes[0].set_title("Amount Distribution by Class")
    axes[0].legend()

    # Box plots
    amount_data = pd.DataFrame({
        "Amount": pd.concat([legit["Amount"], fraud["Amount"]], ignore_index=True),
        "Class": ["Legit"] * len(legit) + ["Fraud"] * len(fraud),
    })
    sns.boxplot(data=amount_data, x="Class", y="Amount", palette={"Legit": LEGIT_COLOR, "Fraud": FRAUD_COLOR},
                showfliers=False, ax=axes[1])
    axes[1].set_title("Amount Box Plots (outliers hidden)")
    axes[1].set_ylabel("Transaction Amount ($)")

    plt.suptitle("Transaction Amount: Fraud vs. Legitimate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "amount_by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    findings["mean_fraud_amount"] = float(fraud["Amount"].mean())
    findings["mean_legit_amount"] = float(legit["Amount"].mean())
    findings["median_fraud_amount"] = float(fraud["Amount"].median())
    print(f"  [EDA] Amount plot saved. Mean fraud: ${findings['mean_fraud_amount']:.2f}, mean legit: ${findings['mean_legit_amount']:.2f}")

    # ── Plot 3: Time distribution by class ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Convert seconds to hours
    legit_hours = legit["Time"] / 3600
    fraud_hours = fraud["Time"] / 3600

    axes[0].hist(legit_hours, bins=48, color=LEGIT_COLOR, alpha=0.7, density=True, label="Legit")
    axes[0].hist(fraud_hours, bins=48, color=FRAUD_COLOR, alpha=FRAUD_ALPHA, density=True, label="Fraud")
    axes[0].set_xlabel("Time (hours since first transaction)", fontsize=10)
    axes[0].set_ylabel("Density")
    axes[0].set_title("Time Distribution by Class")
    axes[0].legend()

    # KDE comparison
    from scipy.stats import gaussian_kde
    t_range = np.linspace(0, legit_hours.max(), 500)
    kde_legit = gaussian_kde(legit_hours)(t_range)
    kde_fraud = gaussian_kde(fraud_hours)(t_range)
    axes[1].plot(t_range, kde_legit / kde_legit.max(), color=LEGIT_COLOR, lw=2, label="Legit (scaled)")
    axes[1].plot(t_range, kde_fraud / kde_fraud.max(), color=FRAUD_COLOR, lw=2, label="Fraud (scaled)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Scaled Density")
    axes[1].set_title("Time KDE (both normalised to 1)")
    axes[1].legend()

    plt.suptitle("Transaction Timing: Fraud vs. Legitimate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "time_by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [EDA] Time distribution saved.")

    # ── Plot 4: Feature correlations with Class ───────────────────────────────
    v_cols = [f"V{i}" for i in range(1, 29)]
    corrs = df[v_cols + ["Amount"]].corrwith(df["Class"]).sort_values()

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [FRAUD_COLOR if c > 0 else LEGIT_COLOR for c in corrs.values]
    ax.barh(range(len(corrs)), corrs.values, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(corrs)))
    ax.set_yticklabels(corrs.index, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson Correlation with Class (1=Fraud)")
    ax.set_title("Feature Correlations with Fraud Class\n(red = positive/fraud-aligned, blue = negative)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    top_positive = corrs[corrs > 0].tail(3).index.tolist()
    top_negative = corrs[corrs < 0].head(3).index.tolist()
    findings["top_positive_corr"] = top_positive
    findings["top_negative_corr"] = top_negative
    print(f"  [EDA] Correlations: strongest positive={top_positive}, strongest negative={top_negative}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  EDA Key Findings:")
    print(f"  1. Fraud: {findings['fraud_count']} / {len(df):,} = {findings['fraud_pct']:.4f}% (severe imbalance)")
    print(f"  2. Mean fraud amount: ${findings['mean_fraud_amount']:.2f} vs legit ${findings['mean_legit_amount']:.2f}")
    print(f"  3. Strongest fraud-aligned features: {top_positive}")
    print(f"  4. Strongest legit-aligned features: {top_negative}")
    print("  5. Fraud occurs across all time periods (no single peak hour)")

    return findings


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df = load_raw_data()
    findings = run_eda(df)
    print("\nFindings:", findings)
