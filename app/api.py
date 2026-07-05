"""
app/api.py — FastAPI endpoint for real-time fraud scoring.

Launch with:
    uvicorn app.api:app --reload --port 8000
"""

import sys
from pathlib import Path
from typing import Optional
import json
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Allow import from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Global variables for model resources
model = None
scaler = None
optimal_threshold = 0.32  # Default fallback if metrics.json is missing or fails


def load_resources():
    """Load model, scaler, and dynamic threshold."""
    global model, scaler, optimal_threshold
    model_path = config.MODELS_DIR / "best_model.joblib"
    scaler_path = config.MODELS_DIR / "scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        raise RuntimeError(
            "Trained model files not found. Please run the training pipeline first: python run_pipeline.py"
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("[OK] Model and scaler loaded successfully.")

    # Dynamically load optimal threshold from reports/metrics.json if available
    metrics_path = config.REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
            for entry in metrics_data:
                if entry.get("tag") == "cost_analysis" and "optimal_threshold" in entry:
                    optimal_threshold = float(entry["optimal_threshold"])
                    print(f"[OK] Dynamically loaded optimal threshold from metrics: {optimal_threshold}")
                    break
        except Exception as e:
            print(f"[Warning] Failed to dynamically load optimal threshold: {e}. Using default: {optimal_threshold}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources on startup and yield."""
    load_resources()
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time model inference endpoint for detecting fraudulent credit card transactions.",
    version="1.0",
    lifespan=lifespan,
)


class TransactionInput(BaseModel):
    """Pydantic schema representing the 30 numerical input features for a transaction."""
    Time: float = Field(..., description="Seconds elapsed since the first transaction", examples=[0.0])
    Amount: float = Field(..., description="Transaction amount in dollars", examples=[99.99])
    V1: float = Field(..., description="PCA anonymized feature V1", examples=[-1.3598])
    V2: float = Field(..., description="PCA anonymized feature V2", examples=[-0.0727])
    V3: float = Field(..., description="PCA anonymized feature V3", examples=[2.5363])
    V4: float = Field(..., description="PCA anonymized feature V4", examples=[1.3781])
    V5: float = Field(..., description="PCA anonymized feature V5", examples=[-0.3383])
    V6: float = Field(..., description="PCA anonymized feature V6", examples=[0.4623])
    V7: float = Field(..., description="PCA anonymized feature V7", examples=[0.2395])
    V8: float = Field(..., description="PCA anonymized feature V8", examples=[0.09869])
    V9: float = Field(..., description="PCA anonymized feature V9", examples=[0.3637])
    V10: float = Field(..., description="PCA anonymized feature V10", examples=[0.09079])
    V11: float = Field(..., description="PCA anonymized feature V11", examples=[-0.5516])
    V12: float = Field(..., description="PCA anonymized feature V12", examples=[-0.6178])
    V13: float = Field(..., description="PCA anonymized feature V13", examples=[-0.9913])
    V14: float = Field(..., description="PCA anonymized feature V14", examples=[-0.3111])
    V15: float = Field(..., description="PCA anonymized feature V15", examples=[1.4681])
    V16: float = Field(..., description="PCA anonymized feature V16", examples=[-0.4704])
    V17: float = Field(..., description="PCA anonymized feature V17", examples=[0.2079])
    V18: float = Field(..., description="PCA anonymized feature V18", examples=[0.0257])
    V19: float = Field(..., description="PCA anonymized feature V19", examples=[0.4039])
    V20: float = Field(..., description="PCA anonymized feature V20", examples=[0.2514])
    V21: float = Field(..., description="PCA anonymized feature V21", examples=[-0.0183])
    V22: float = Field(..., description="PCA anonymized feature V22", examples=[0.2778])
    V23: float = Field(..., description="PCA anonymized feature V23", examples=[-0.1104])
    V24: float = Field(..., description="PCA anonymized feature V24", examples=[0.0669])
    V25: float = Field(..., description="PCA anonymized feature V25", examples=[0.1285])
    V26: float = Field(..., description="PCA anonymized feature V26", examples=[-0.189])
    V27: float = Field(..., description="PCA anonymized feature V27", examples=[0.1335])
    V28: float = Field(..., description="PCA anonymized feature V28", examples=[-0.0210])


class PredictionResponse(BaseModel):
    """Output prediction response format."""
    fraud_probability: float = Field(..., description="Probability score between 0.0 and 1.0")
    decision: str = Field(..., description="Classification outcome: 'flag' (suspected fraud) or 'approve'")
    threshold: float = Field(..., description="Probability threshold used for the decision")


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(
    transaction: TransactionInput,
    custom_threshold: Optional[float] = Query(
        None,
        ge=0.01,
        le=0.99,
        description="Override the default cost-optimal threshold (0.32)",
    ),
):
    """
    Score a single credit card transaction and decide whether to approve or flag it.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please ensure the pipeline has run and the server initialized properly.",
        )

    # 1. Parse input to DataFrame
    input_dict = transaction.model_dump()
    df = pd.DataFrame([input_dict])

    # 2. Scale Time and Amount using the loaded StandardScaler (leakage guard compliant)
    cols_to_scale = ["Amount", "Time"]
    try:
        scaled_features = scaler.transform(df[cols_to_scale])
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = scaled_features
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Preprocessing error: failed to scale features. Details: {e}",
        )

    # 3. Model Inference
    try:
        # Re-order columns to match original training order
        feature_order = [
            "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]
        df_scaled = df_scaled[feature_order]
        fraud_prob = float(model.predict_proba(df_scaled)[0, 1])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model scoring failed. Details: {e}",
        )

    # 4. Make decision based on threshold
    thresh = custom_threshold if custom_threshold is not None else optimal_threshold
    decision = "flag" if fraud_prob >= thresh else "approve"

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 5),
        decision=decision,
        threshold=thresh,
    )
