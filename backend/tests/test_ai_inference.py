import importlib
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")
pytest.importorskip("PIL.Image")
from PIL import Image


def _fake_image_bytes(width: int = 800, height: int = 600) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_scales_and_filters_detections(monkeypatch):
    class FakeModelMeta:
        custom_metadata_map = {"names": "{0: 'apple', 1: 'banana'}"}

    class FakeSession:
        def get_modelmeta(self):
            return FakeModelMeta()

        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 768, 768])]

        def run(self, *_args, **_kwargs):
            return [
                np.array(
                    [
                        [
                            [320, 320, 320, 320, 0.9, 0.8, 0.1],
                            [0, 0, 0, 0, 0.1, 0.2, 0.9],
                        ]
                    ],
                    dtype=np.float32,
                )
            ]

    monkeypatch.setattr("app.services.ai.model_loader.get_model_session", lambda: FakeSession())
    module_name = "app.services.ai.inference"
    sys.modules.pop(module_name, None)
    inference = importlib.import_module(module_name)

    predictions = inference.predict(_fake_image_bytes(), conf_threshold=0.5)

    assert len(predictions) == 1
    detection = predictions[0]
    assert detection["class_name"] == "apple"
    assert detection["confidence"] == pytest.approx(0.72, rel=1e-6)
    assert detection["bounding_box"] == pytest.approx([166.6667, 125.0, 500.0, 375.0], rel=1e-4)
    assert detection["calories"] == 52.0
    assert detection["carbs"] == 14.0