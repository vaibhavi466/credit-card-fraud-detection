"""
app/streamlit_app.py — Interactive fraud detection demo.

Launch with:
    streamlit run app/streamlit_app.py

Features:
- Pick a transaction from the test set (or upload a CSV row)
- See fraud probability gauge
- See SHAP waterfall explaining WHY the model scored it that way
- Drag a threshold slider and see precision/recall update live

This is what gets screen-shared in interviews — the goal is to make it easy
to explain "here's what the model is doing and why" in real time.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# Allow import from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data_loader import load_raw_data, get_features_and_target
from src.preprocessing import preprocess
from src.evaluate import evaluate_all_thresholds

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #2563eb;
        margin-bottom: 1rem;
    }
    .fraud-badge {
        background: #fee2e2;
        color: #b91c1c;
        border-radius: 8px;
        padding: 0.3rem 1rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .legit-badge {
        background: #dcfce7;
        color: #15803d;
        border-radius: 8px;
        padding: 0.3rem 1rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load model, scaler, and test data. Cached so it only runs once."""
    model_path = config.MODELS_DIR / "best_model.joblib"
    scaler_path = config.MODELS_DIR / "scaler.joblib"

    if not model_path.exists():
        return None, None, None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load raw data and reproduce the same test split
    df = load_raw_data()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test, _ = preprocess(X, y)
    # Apply the saved scaler (already fitted) — just for feature access
    return model, scaler, X_test, y_test, df


@st.cache_data
def get_threshold_df(_model, _X_test, _y_test):
    return evaluate_all_thresholds(_model, _X_test, _y_test)


@st.cache_data
def get_metrics_json():
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH) as f:
            return json.load(f)
    return []


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 Credit Card Fraud Detector</div>', unsafe_allow_html=True)
st.markdown("**Portfolio demo** — ULB Credit Card Fraud dataset (284,807 transactions, 492 fraud)")
st.markdown("---")

model, scaler, X_test, y_test, df_raw = load_pipeline()

if model is None:
    st.error(
        "⚠️  Model not found. Please run `python run_pipeline.py` first to train the model.",
        icon="🚫",
    )
    st.code("python run_pipeline.py", language="bash")
    st.stop()

# ── Sidebar: controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    # Threshold slider
    st.subheader("Classification Threshold")
    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01,
        help=(
            "Default is 0.5, but optimal depends on your cost model. "
            "Lower → catch more fraud (higher recall), but more false alarms. "
            "Higher → fewer false alarms, but miss more fraud."
        ),
    )

    st.subheader("Sample Selection")
    sample_mode = st.radio(
        "Select transaction to inspect",
        ["Random fraud (TP)", "Random legit (TN)", "Pick by index"],
    )

    if sample_mode == "Pick by index":
        idx = st.number_input("Test set index", min_value=0,
                              max_value=len(X_test) - 1, value=0)
    elif sample_mode == "Random fraud (TP)":
        fraud_indices = np.where(y_test.values == 1)[0]
        idx = int(np.random.choice(fraud_indices))
    else:
        legit_indices = np.where(y_test.values == 0)[0]
        idx = int(np.random.choice(legit_indices))

    st.caption(f"Inspecting test-set index: {idx}")

    if st.button("🔀 Randomize selection"):
        st.rerun()

# ── Main: two columns ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("🎯 Transaction Prediction")

    sample = X_test.iloc[[idx]]
    true_label = int(y_test.iloc[idx])
    fraud_prob = float(model.predict_proba(sample)[0, 1])
    predicted_label = int(fraud_prob >= threshold)

    # Probability gauge
    prob_color = "#dc2626" if fraud_prob > 0.5 else "#16a34a"
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1rem;color:#64748b;">Fraud Probability</div>
        <div style="font-size:2.5rem;font-weight:700;color:{prob_color};">{fraud_prob:.1%}</div>
        <div style="font-size:0.85rem;color:#94a3b8;">at threshold {threshold:.2f} ->
        {'<span class="fraud-badge">⚠️ FRAUD</span>' if predicted_label == 1
         else '<span class="legit-badge">✅ LEGIT</span>'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # True label
    correct = predicted_label == true_label
    outcome_label = {
        (1, 1): "✅ True Positive (correctly flagged fraud)",
        (0, 0): "✅ True Negative (correctly cleared legit)",
        (1, 0): "⚠️ False Positive (legitimate flagged as fraud)",
        (0, 1): "❌ False Negative (fraud missed!)",
    }[(predicted_label, true_label)]
    st.info(f"**Ground truth:** {'🔴 Fraud' if true_label else '🟢 Legit'}  |  {outcome_label}")

    # Key feature values
    st.subheader("📋 Transaction Features")
    feature_df = sample.T.reset_index()
    feature_df.columns = ["Feature", "Value"]
    feature_df["Value"] = feature_df["Value"].round(4)
    st.dataframe(feature_df, height=300, use_container_width=True)

with col2:
    st.subheader("🔎 SHAP Explanation")

    try:
        import shap
        rng = np.random.default_rng(config.RANDOM_STATE)
        bg_idx = rng.choice(len(X_test), size=min(50, len(X_test)), replace=False)
        X_background = X_test.iloc[bg_idx]

        explainer = shap.TreeExplainer(model, X_background)
        explanation = explainer(sample, check_additivity=False)

        if explanation.values.ndim == 3:
            exp_slice = shap.Explanation(
                values=explanation.values[0, :, 1],
                base_values=explanation.base_values[0, 1],
                data=explanation.data[0],
                feature_names=X_test.columns.tolist(),
            )
        else:
            exp_slice = shap.Explanation(
                values=explanation.values[0],
                base_values=explanation.base_values[0],
                data=explanation.data[0],
                feature_names=X_test.columns.tolist(),
            )

        fig, ax = plt.subplots(figsize=(7, 5))
        shap.plots.waterfall(exp_slice, max_display=12, show=False)
        plt.title(f"SHAP — Why this prediction? (fraud prob={fraud_prob:.3f})", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Top contributors
        top_pos = pd.Series(exp_slice.values, index=X_test.columns).nlargest(3)
        top_neg = pd.Series(exp_slice.values, index=X_test.columns).nsmallest(3)
        st.caption(
            f"**Pushing toward fraud:** {', '.join(f'{f} (+{v:.3f})' for f, v in top_pos.items())}  \n"
            f"**Pushing toward legit:** {', '.join(f'{f} ({v:.3f})' for f, v in top_neg.items())}"
        )
    except Exception as e:
        st.warning(f"SHAP waterfall unavailable: {e}")
        shap_path = config.FIGURES_DIR / "shap_waterfall.png"
        if shap_path.exists():
            st.image(str(shap_path), caption="SHAP waterfall (pre-computed)")

# ── Threshold explorer ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Live Threshold Explorer")

threshold_df = get_threshold_df(model, X_test, y_test)
current = threshold_df[threshold_df["threshold"] == round(threshold, 2)]
if current.empty:
    current = threshold_df.iloc[(threshold_df["threshold"] - threshold).abs().argsort()[:1]]

col3, col4, col5, col6 = st.columns(4)
with col3:
    st.metric("Precision", f"{current['precision'].values[0]:.4f}",
              help="Of flagged transactions, what % are actually fraud?")
with col4:
    st.metric("Recall", f"{current['recall'].values[0]:.4f}",
              help="Of actual frauds, what % did we catch?")
with col5:
    st.metric("F1 Score", f"{current['f1'].values[0]:.4f}")
with col6:
    # Estimate cost at current threshold
    if df_raw is not None:
        fn_cost = float(df_raw.loc[df_raw["Class"] == 1, "Amount"].mean())
        y_proba_all = model.predict_proba(X_test)[:, 1]
        y_pred_all = (y_proba_all >= threshold).astype(int)
        fn = int(((y_pred_all == 0) & (y_test.values == 1)).sum())
        fp = int(((y_pred_all == 1) & (y_test.values == 0)).sum())
        cost = fn * fn_cost + fp * config.FP_COST_DOLLARS
        st.metric("Est. Cost", f"${cost:,.0f}",
                  help=f"FN×${fn_cost:.0f} + FP×${config.FP_COST_DOLLARS:.0f} (illustrative)")

# Threshold curve chart
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="#2563eb", lw=2)
ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color="#dc2626", lw=2)
ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1", color="#16a34a", lw=2)
ax.axvline(threshold, color="orange", lw=2, ls="--", label=f"Current ({threshold:.2f})")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 Trade-off")
ax.legend(fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Model comparison table ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Model Comparison (all strategies)")

metrics_data = get_metrics_json()
if metrics_data:
    display_cols = ["tag", "precision", "recall", "f1", "auprc", "roc_auc"]
    rows = [m for m in metrics_data if all(k in m for k in ["precision", "recall", "f1", "auprc"])]
    if rows:
        comparison_df = pd.DataFrame(rows)
        if "tag" in comparison_df.columns:
            comparison_df = comparison_df.rename(columns={"tag": "Strategy/Model"})
        show_cols = [c for c in ["Strategy/Model", "model_name", "precision", "recall", "f1", "auprc", "roc_auc"] if c in comparison_df.columns]
        st.dataframe(
            comparison_df[show_cols].sort_values("auprc", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=350,
        )
else:
    st.info("Run `python run_pipeline.py` to populate the model comparison table.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with scikit-learn, XGBoost, SHAP, and Streamlit · "
    "Dataset: ULB Credit Card Fraud (Kaggle) · "
    "Primary metric: AUPRC (not accuracy — see docs/interview_prep.md)"
)
