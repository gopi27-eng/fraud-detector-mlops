import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_legitimate_transaction(client: TestClient):
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
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_schema_validation_error(client: TestClient):
    # Invalid: negative amount violates Field(gt=0.0)
    bad_payload = {
        "step": 100,
        "type": "PAYMENT",
        "amount": -50.00,
        "oldbalanceOrg": 100.00,
        "newbalanceOrig": 50.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 0.00,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422