import pytest
from fastapi.testclient import TestClient
from WattPredictor.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "WattPredictor REST API"
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "rmse" in data or "mape" in data or "mae" in data


def test_predict_endpoint():
    response = client.post("/predict")
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert len(data["predictions"]) == 11
