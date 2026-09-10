"""
tests/test_api.py — Unit tests for the FastAPI endpoint.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Allow import from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.api import app, load_resources
import app.api as api_module
import config

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_api(tmp_path_factory):
    """Create a dummy fitted model and scaler, save to a temp dir, override config.MODELS_DIR, and load them."""
    feature_order = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]
    rng = np.random.default_rng(42)
    df_dummy = pd.DataFrame(rng.standard_normal((10, 30)), columns=feature_order)
    y_dummy = np.array([0, 1] * 5)  # Guaranteed two-class split
    
    model = LogisticRegression()
    model.fit(df_dummy, y_dummy)

    scaler = StandardScaler()
    scaler.fit(df_dummy[["Amount", "Time"]])

    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "best_model.joblib"
    scaler_path = tmp_dir / "scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    old_models_dir = config.MODELS_DIR
    config.MODELS_DIR = tmp_dir

    load_resources()

    yield

    config.MODELS_DIR = old_models_dir


def test_health_check():
    """Verify that the health check endpoint returns 200 and indicates model load status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True


def test_predict_legit():
    """Test /predict with a mock legitimate transaction."""
    payload = {
        "Time": 0.0,
        "Amount": 10.00,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert data["decision"] in ["approve", "flag"]
    assert data["threshold"] == api_module.optimal_threshold


def test_predict_custom_threshold():
    """Test /predict with a custom threshold override query parameter."""
    payload = {
        "Time": 0.0,
        "Amount": 150.00,
        **{f"V{i}": -1.0 for i in range(1, 29)}
    }
    response = client.post("/predict?custom_threshold=0.75", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["threshold"] == 0.75


def test_predict_invalid_schema():
    """Verify that /predict returns 422 Unprocessable Entity for missing features."""
    payload = {
        "Time": 0.0,
        "Amount": 10.00,
        **{f"V{i}": 0.0 for i in range(1, 28)}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
