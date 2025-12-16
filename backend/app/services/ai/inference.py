"""Inference pipeline for YOLOv8 food detection on CPU."""

from __future__ import annotations

import ast
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

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


@lru_cache(maxsize=1)
def get_class_names() -> List[str]:
    """Return class names loaded from the model metadata, cached after first read."""

    return _load_class_names()


def _sigmoid(x):
    """Numerically stable sigmoid for logits."""

    return 1.0 / (1.0 + np.exp(-x))


def _get_model_input_size(session) -> Tuple[int, int]:
    """Extract the static (width, height) from the model input shape."""

    shape = session.get_inputs()[0].shape
    if len(shape) < 4:
        raise ValueError(f"Unexpected model input shape: {shape}")

    height, width = shape[2], shape[3]
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("Model input dimensions must be static integers.")

    return int(width), int(height)


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection-over-Union for two ``[x1, y1, x2, y2]`` boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _nms(detections: List[Dict[str, object]], iou_threshold: float = 0.45) -> List[Dict[str, object]]:
    """Apply Non-Maximum Suppression to a list of detection dicts."""

    sorted_dets = sorted(detections, key=lambda d: float(d["confidence"]), reverse=True)
    kept: List[Dict[str, object]] = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)
        sorted_dets = [
            det
            for det in sorted_dets
            if _iou(best["bounding_box"], det["bounding_box"]) < iou_threshold
        ]

    return kept


_NUTRITION_MAP: Dict[str, Dict[str, float]] = {
    "apple": {"calories": 52.0, "proteins": 0.3, "fats": 0.2, "carbs": 14.0},
    "banana": {"calories": 96.0, "proteins": 1.3, "fats": 0.3, "carbs": 27.0},
    "chicken": {"calories": 165.0, "proteins": 31.0, "fats": 3.6, "carbs": 0.0},
    "salmon": {"calories": 208.0, "proteins": 20.0, "fats": 13.0, "carbs": 0.0},
    "rice": {"calories": 130.0, "proteins": 2.7, "fats": 0.3, "carbs": 28.0},
    "bread": {"calories": 265.0, "proteins": 9.0, "fats": 3.2, "carbs": 49.0},
}


def predict(image_bytes: bytes, conf_threshold: float = 0.6) -> List[Dict[str, object]]:
    """Run YOLOv8 detection on input image bytes.

    The function preprocesses the image, performs ONNXRuntime inference, filters
    predictions by confidence, applies Non-Maximum Suppression, rescales
    bounding boxes to the original image dimensions, and returns structured
    detection results with mock nutrition mapping.

    Args:
        image_bytes: Raw image bytes.
        conf_threshold: Minimum confidence threshold for predictions.

    Returns:
        A list of dictionaries with keys ``class_name``, ``confidence``, and
        ``bounding_box`` ([x1, y1, x2, y2] in original pixel coordinates).
    """

    session = get_model_session()
    model_w, model_h = _get_model_input_size(session)
    input_tensor, original_size = load_image(image_bytes, target_size=(model_w, model_h))
    orig_w, orig_h = original_size
    scale_x = float(orig_w) / float(model_w)
    scale_y = float(orig_h) / float(model_h)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # YOLOv8 ONNX output is assumed to be (batch, num_predictions, attributes).
    # The typical attribute layout is [cx, cy, w, h, obj_conf, class_scores..., masks...].
    predictions = outputs[0]
    if predictions.ndim != 3:
        raise ValueError(f"Unexpected prediction tensor shape: {predictions.shape}")

    preds = predictions[0]
    class_names = get_class_names()
    num_classes = len(class_names)
    results: List[Dict[str, object]] = []

    for row in preds:
        objectness = float(_sigmoid(float(row[4])))
        class_scores = _sigmoid(np.asarray(row[5 : 5 + num_classes], dtype=np.float32))
        if class_scores.size == 0:
            continue

        class_id = int(np.argmax(class_scores))
        class_conf = float(class_scores[class_id])
        confidence = float(objectness * class_conf)
        if confidence < conf_threshold:
            continue

        class_name = class_names[class_id]
        if class_name == "background":
            continue

        bbox_xyxy = _xywh_to_xyxy(row[:4])
        scaled_bbox = _scale_bbox(
            bbox_xyxy, scale_x=scale_x, scale_y=scale_y, max_w=float(orig_w), max_h=float(orig_h)
        )
        if not np.all(np.isfinite(scaled_bbox)):
            continue

        x1, y1, x2, y2 = scaled_bbox
        if x2 <= x1 or y2 <= y1:
            continue

        width, height = x2 - x1, y2 - y1
        if width < 10.0 or height < 10.0:
            continue

        nutrition = _NUTRITION_MAP.get(
            class_name,
            {"calories": 0.0, "proteins": 0.0, "fats": 0.0, "carbs": 0.0},
        )
        results.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "bounding_box": [float(x1), float(y1), float(x2), float(y2)],
                "calories": nutrition["calories"],
                "proteins": nutrition["proteins"],
                "fats": nutrition["fats"],
                "carbs": nutrition["carbs"],
            }
        )

    return _nms(results)