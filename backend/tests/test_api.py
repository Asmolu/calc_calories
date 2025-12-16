import io
from typing import Any, Dict, List

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app.api import routes
from app.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_health_endpoint_returns_status(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert payload["environment"] == "development"


def test_predict_requires_file(client: TestClient) -> None:
    response = client.post("/api/ai/predict")
    assert response.status_code == 400
    assert response.json()["detail"] == "Image file is required"


def test_predict_uses_ai_inference(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    recorded_payload: Dict[str, Any] = {}

    def fake_predict(image_bytes: bytes, conf_threshold: float = 0.4) -> List[Dict[str, Any]]:
        recorded_payload["image_bytes"] = image_bytes
        recorded_payload["conf_threshold"] = conf_threshold
        return [
            {
                "class_name": "salmon",
                "confidence": 0.91,
                "bounding_box": [1.0, 2.0, 3.0, 4.0],
            }
        ]

    monkeypatch.setattr(routes.inference, "predict", fake_predict)

    response = client.post(
        "/api/ai/predict",
        files={"file": ("test.png", io.BytesIO(b"imagedata"), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()

    assert recorded_payload["image_bytes"] == b"imagedata"
    assert payload["items"] == [
        {
            "name": "salmon",
            "calories": 0.0,
            "proteins": 0.0,
            "fats": 0.0,
            "carbs": 0.0,
            "confidence": 0.91,
            "bounding_box": [1.0, 2.0, 3.0, 4.0],
        }
    ]
    assert payload["totals"] == {
        "calories": 0.0,
        "proteins": 0.0,
        "fats": 0.0,
        "carbs": 0.0,
    }