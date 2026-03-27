"""
Part 1 of the vision pipeline: OpenCV layout preprocessing.

Layout (raw image) → OpenCV processing → Processed image / features.

Used to prepare layout images (floor plans, room plans, documents) so that
downstream models (e.g. CNN, ControlNet-style models) get clean, structured input.

Operations:
- Resize to a consistent size
- Convert to grayscale (optional)
- Detect edges / enhance shapes
- Clean noise (denoise)
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


# Default target size for training (width, height). Common for diffusion: 512 or 768.
DEFAULT_TARGET_SIZE: Tuple[int, int] = (512, 512)


def load_layout(image_path: Union[str, Path]) -> np.ndarray:
    """Load a layout image from disk. Supports PNG, JPG, etc."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Layout image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def resize_image(
    img: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    keep_aspect: bool = False,
) -> np.ndarray:
    """Resize image to target (width, height). If keep_aspect, fit inside and pad."""
    w, h = target_size
    if keep_aspect:
        h_cur, w_cur = img.shape[:2]
        scale = min(w / w_cur, h / h_cur)
        new_w, new_h = int(w_cur * scale), int(h_cur * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Pad to exact target size
        top = (h - img.shape[0]) // 2
        left = (w - img.shape[1]) // 2
        img = cv2.copyMakeBorder(
            img, top, h - img.shape[0] - top, left, w - img.shape[1] - left,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    else:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale (single channel)."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img: np.ndarray) -> np.ndarray:
    """Reduce noise while keeping edges. Works on grayscale or BGR."""
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h=6, hForColorComponents=6, templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)


def detect_edges(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge detection. Input should be grayscale."""
    if len(img.shape) == 3:
        img = to_grayscale(img)
    return cv2.Canny(img, low, high)


def preprocess_layout(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    to_gray: bool = True,
    denoise_first: bool = True,
    edge_detection: bool = True,
    keep_aspect: bool = False,
) -> np.ndarray:
    """
    Full OpenCV preprocessing pipeline for a layout image.

    Order: load → resize → [denoise] → [grayscale] → [edge detection].
    Returns the processed image (single channel if to_gray or edge_detection, else BGR).
    """
    img = load_layout(image_path)
    img = resize_image(img, target_size=target_size, keep_aspect=keep_aspect)

    if denoise_first:
        img = denoise(img)

    if to_gray:
        img = to_grayscale(img)

    if edge_detection:
        img = detect_edges(img)

    return img


def save_processed(out_path: Union[str, Path], img: np.ndarray) -> None:
    """Save processed image (grayscale or BGR) to disk."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def find_layout_image(folder: Path) -> Optional[Path]:
    """Find a layout/floor plan image in a variation folder. Prefer image.png, then any non-output PNG."""
    candidate = folder / "image.png"
    if candidate.exists():
        return candidate
    for f in folder.glob("*.png"):
        if not f.name.startswith(("2D_plan", "output_", "semi3D", "3D_", "layout_processed")):
            return f
    for f in folder.glob("*.jpg"):
        return f
    return None
