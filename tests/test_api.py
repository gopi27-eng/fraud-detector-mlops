import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    # 'with' context manager executes FastAPI's lifespan startup/shutdown hooks
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200


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


def test_predict_invalid_schema(client):
    payload = {
        "step": -1,
        "type": "INVALID_TYPE",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]