import pytest
from datetime import datetime
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


def test_version_endpoint(monkeypatch):
    sha = "a" * 40
    monkeypatch.setenv("BUILD_COMMIT_SHA", sha)
    monkeypatch.setenv("BUILD_TIME", "2026-09-30T00:00:00+00:00")
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {
        "commit_sha": sha,
        "build_time": "2026-09-30T00:00:00+00:00",
    }
    assert datetime.fromisoformat(response.json()["build_time"]).utcoffset().total_seconds() == 0


def test_version_rejects_missing_sha(monkeypatch):
    monkeypatch.delenv("BUILD_COMMIT_SHA", raising=False)
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.setenv("BUILD_TIME", "2026-09-30T00:00:00+00:00")
    assert client.get("/version").status_code == 503


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "rmse" in data or "mape" in data or "mae" in data


def test_predict_endpoint():
    response = client.post("/predict")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert len(data["predictions"]) == 11
