"""The vision slots' image contract: decode, then resize to the slot's grid.

Moved from okfgraph (``images.prepare_vision_rgb``) so a host and an
in-process caller resize the same image to the same pixels. Pillow does
the decoding and the bicubic resize, as it always has for okfgraph's
image vectors; install it with ``embroider[image]``.
"""
from __future__ import annotations

import io
from typing import Tuple

from embroider import _native


def target_size_fn(model_id: str):
    """The slot's ``(h, w) -> (h, w)`` resize rule for *model_id*."""
    if model_id.startswith("google/"):
        return _native.gemma_vision_target_size
    return _native.vision_target_size


def prepare_image(data: bytes, model_id: str) -> Tuple[bytes, int, int]:
    """Decode image bytes and resize them for *model_id*'s vision slot.

    Returns ``(rgb, h, w)``: row-major ``[h, w, 3]`` uint8, ready for
    ``encode_image(rgb, h, w)``. The image is resized (Pillow bicubic)
    only when the slot's target differs from its size. Raises
    ``ValueError`` when the bytes are not an image, and ``ImportError``
    when Pillow is missing.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "prepare_image needs Pillow: pip install 'embroider[image]'") from None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - every decoder failure
        raise ValueError(f"image bytes do not decode: {exc}") from exc
    rh, rw = target_size_fn(model_id)(img.height, img.width)
    if (rh, rw) != (img.height, img.width):
        img = img.resize((rw, rh), Image.Resampling.BICUBIC)
    return img.tobytes(), img.height, img.width
