"""Image preprocessing utilities for YOLOv8 inference."""

from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image

_IMAGE_SIZE: Tuple[int, int] = (640, 640)


def load_image(image_bytes: bytes) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load and preprocess an image from raw bytes for YOLOv8 input.

    The image is converted to RGB, resized to 640x640, normalized to ``[0, 1]``,
    and converted to NCHW float32 layout.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        A tuple of (input_tensor, original_size) where input_tensor has shape
        ``(1, 3, 640, 640)`` and original_size is ``(width, height)`` of the
        source image.
    """

    with Image.open(BytesIO(image_bytes)) as img:
        rgb_image = img.convert("RGB")
        original_size = rgb_image.size  # (width, height)
        resized_image = rgb_image.resize(_IMAGE_SIZE)

    np_image = np.asarray(resized_image, dtype=np.float32) / 255.0
    # Convert to NCHW
    nchw_image = np.transpose(np_image, (2, 0, 1))[None, ...]
    return nchw_image, original_size