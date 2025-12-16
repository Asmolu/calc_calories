import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app.main import create_app


client = TestClient(create_app())


def test_health_endpoint_returns_status():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert payload["environment"] == "development"


def test_predict_endpoint_returns_placeholder_items():
    response = client.post("/api/ai/predict", files={"file": ("test.png", b"data", "image/png")})
    assert response.status_code == 200
    payload = response.json()

    assert "items" in payload and len(payload["items"]) == 2
    assert payload["items"][0]["name"] == "Salmon"
    assert payload["items"][1]["name"] == "Roasted vegetables"

    totals = payload["totals"]
    assert totals == {"calories": 320.0, "proteins": 23.0, "fats": 17.0, "carbs": 18.0}