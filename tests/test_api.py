"""
tests/test_api.py — Unit tests for the FastAPI endpoint.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

# Allow import from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.api import app, startup_event

# Create TestClient
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_api():
    """Trigger startup event to load model and scaler."""
    startup_event()


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
    assert data["threshold"] == 0.32


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
    # Missing V28
    payload = {
        "Time": 0.0,
        "Amount": 10.00,
        **{f"V{i}": 0.0 for i in range(1, 28)}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
