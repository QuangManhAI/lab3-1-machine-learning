"""Image loading, preprocessing, and color-space utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image from disk and return an RGB uint8 array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_image(image: np.ndarray, max_size: int = 350) -> np.ndarray:
    """Resize an image while preserving aspect ratio."""
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_size:
        return image.copy()

    scale = max_size / longest_side
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = Image.fromarray(image).resize(new_size, Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def flatten_pixels(image: np.ndarray) -> np.ndarray:
    """Convert an image from H x W x C to N x C float features."""
    return image.reshape(-1, image.shape[-1]).astype(float)


def convert_color_space(image: np.ndarray, color_space: str = "RGB") -> np.ndarray:
    """Convert an RGB image to RGB, HSV, or LAB feature arrays."""
    color_space = color_space.upper()
    if color_space == "RGB":
        return image.copy()

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for HSV and LAB conversion") from exc

    if color_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if color_space == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    raise ValueError(f"Unsupported color_space: {color_space}")


def convert_to_rgb(image: np.ndarray, color_space: str = "RGB") -> np.ndarray:
    """Convert an image in RGB, HSV, or LAB back to RGB uint8."""
    color_space = color_space.upper()
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    if color_space == "RGB":
        return image_uint8

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for HSV and LAB conversion") from exc

    if color_space == "HSV":
        return cv2.cvtColor(image_uint8, cv2.COLOR_HSV2RGB)
    if color_space == "LAB":
        return cv2.cvtColor(image_uint8, cv2.COLOR_LAB2RGB)
    raise ValueError(f"Unsupported color_space: {color_space}")


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Save an image array to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(output_path)
    return output_path


def image_summary(image: np.ndarray) -> dict[str, object]:
    """Return basic image metadata for notebook inspection."""
    return {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min": int(image.min()),
        "max": int(image.max()),
        "unique_colors": int(np.unique(image.reshape(-1, image.shape[-1]), axis=0).shape[0]),
    }
