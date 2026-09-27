import pytest
from fastapi.testclient import TestClient
# Update this import if your app instance is located elsewhere (e.g. from src.api.app import app)
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    # 'with TestClient(app)' explicitly triggers FastAPI's startup / lifespan events
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") in ["healthy", "ok", "ready"] or "status" in data


def test_predict_endpoint_valid_payload(client):
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
    assert "fraud_probability" in data or "is_fraud" in data or "prediction" in data


def test_predict_invalid_schema(client):
    payload = {
        "step": -1,
        "type": "INVALID_TYPE",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]