"""ONNX Runtime model loader for YOLOv8 food segmentation model."""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import onnxruntime as ort

_MODEL_PATH: Final = "backend/app/services/ai/models/yolov8_foodseg103.onnx"


@lru_cache(maxsize=1)
def get_model_session() -> ort.InferenceSession:
    """Return a cached ONNX Runtime session for the YOLOv8 model.

    The session is initialized once and reused across inference calls to avoid
    repetitive model loading overhead. CPU execution is enforced.
    """

    return ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])