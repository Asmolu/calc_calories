"""Inference pipeline for YOLOv8 food detection on CPU."""

from __future__ import annotations

import ast
from typing import Dict, List, Sequence

import numpy as np

from .model_loader import get_model_session
from .preprocessing import load_image


def _xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    """Convert ``(cx, cy, w, h)`` box format to ``[x1, y1, x2, y2]``."""

    cx, cy, w, h = box
    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = cx + (w / 2)
    y2 = cy + (h / 2)
    return [float(x1), float(y1), float(x2), float(y2)]


def _scale_bbox(
    bbox: Sequence[float], scale_x: float, scale_y: float, max_w: float, max_h: float
) -> List[float]:
    """Scale bounding box coordinates back to original image size and clamp."""

    x1, y1, x2, y2 = bbox
    scaled = [
        max(0.0, x1 * scale_x),
        max(0.0, y1 * scale_y),
        x2 * scale_x,
        y2 * scale_y,
    ]
    scaled[2] = min(scaled[2], max_w)
    scaled[3] = min(scaled[3], max_h)
    return scaled


def _load_class_names() -> List[str]:
    """Load class names from model metadata.

    YOLOv8 ONNX exports store class names under the ``names`` metadata field as a
    stringified dictionary mapping class indices to names. If the metadata is
    absent, a clear error is raised so the mapping can be provided explicitly.
    """

    session = get_model_session()
    metadata = session.get_modelmeta().custom_metadata_map or {}
    raw_names = metadata.get("names")
    if not raw_names:
        raise ValueError("Model metadata does not include class names under 'names'.")

    try:
        parsed = ast.literal_eval(raw_names)
    except (SyntaxError, ValueError) as exc:  # pragma: no cover - defensive parsing
        raise ValueError("Failed to parse class names from model metadata.") from exc

    if isinstance(parsed, dict):
        # Sort by key to ensure index alignment.
        return [name for _, name in sorted(parsed.items(), key=lambda item: int(item[0]))]

    if isinstance(parsed, list):
        return [str(name) for name in parsed]

    raise ValueError("Unsupported class names format in model metadata.")


_CLASS_NAMES: List[str] = _load_class_names()
_MODEL_SIZE = 640.0


def predict(image_bytes: bytes, conf_threshold: float = 0.4) -> List[Dict[str, object]]:
    """Run YOLOv8 detection on input image bytes.

    The function preprocesses the image, performs ONNXRuntime inference, filters
    predictions by confidence, rescales bounding boxes to the original image
    dimensions, and returns structured detection results.

    Args:
        image_bytes: Raw image bytes.
        conf_threshold: Minimum confidence threshold for predictions.

    Returns:
        A list of dictionaries with keys ``class_name``, ``confidence``, and
        ``bounding_box`` ([x1, y1, x2, y2] in original pixel coordinates).
    """

    session = get_model_session()
    input_tensor, original_size = load_image(image_bytes)
    orig_w, orig_h = original_size
    scale_x = float(orig_w) / _MODEL_SIZE
    scale_y = float(orig_h) / _MODEL_SIZE

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # YOLOv8 ONNX output is assumed to be (batch, num_predictions, attributes).
    # The typical attribute layout is [cx, cy, w, h, obj_conf, class_scores..., masks...].
    predictions = outputs[0]
    if predictions.ndim != 3:
        raise ValueError(f"Unexpected prediction tensor shape: {predictions.shape}")

    preds = predictions[0]
    num_classes = len(_CLASS_NAMES)
    results: List[Dict[str, object]] = []

    for row in preds:
        objectness = float(row[4])
        class_scores = row[5 : 5 + num_classes]
        if class_scores.size == 0:
            continue

        class_id = int(np.argmax(class_scores))
        class_conf = float(class_scores[class_id])
        confidence = objectness * class_conf
        if confidence < conf_threshold:
            continue

        bbox_xyxy = _xywh_to_xyxy(row[:4])
        scaled_bbox = _scale_bbox(
            bbox_xyxy, scale_x=scale_x, scale_y=scale_y, max_w=float(orig_w), max_h=float(orig_h)
        )
        results.append(
            {
                "class_name": _CLASS_NAMES[class_id],
                "confidence": confidence,
                "bounding_box": scaled_bbox,
            }
        )

    return results