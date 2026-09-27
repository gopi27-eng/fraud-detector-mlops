import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_predict_endpoint_valid_payload():
    payload = {
        "step": 100,
        "type": "PAYMENT",
        "amount": 250.75,
        "oldbalanceOrg": 5000.00,
        "newbalanceOrig": 4749.25,
        "oldbalanceDest": 1000.00,
        "newbalanceDest": 1250.75,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert isinstance(data["is_fraud"], bool)
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_invalid_schema():
    # Missing required fields to verify Pydantic rejection
    bad_payload = {"step": 100, "amount": 250.75}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422