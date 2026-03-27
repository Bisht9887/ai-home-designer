"""
Evaluation metrics for generated interior images.

- LPIPS: Perceptual similarity (lower = more similar)
- SSIM: Structural similarity (0–1, higher = more similar)
- MSE: Mean Squared Error, pixel-wise (lower = more similar)
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _load_image(path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load image as float32 RGB in [0, 1], optionally resized."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    return img


def compute_ssim(img_a_path: Path, img_b_path: Path) -> float:
    """
    Structural Similarity Index. Returns value in [0, 1].
    Higher = more structurally similar.
    """
    from skimage.metrics import structural_similarity as ssim

    a = _load_image(img_a_path)
    b = _load_image(img_b_path)

    # Resize b to match a if sizes differ
    if a.shape[:2] != b.shape[:2]:
        b = _resize_to_match(b, a)

    # SSIM returns a scalar (mean over image)
    score = ssim(a, b, channel_axis=2, data_range=1.0)
    return float(score)


def _resize_to_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    import cv2

    h, w = ref.shape[:2]
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)


def compute_lpips(img_a_path: Path, img_b_path: Path) -> float:
    """
    LPIPS (Learned Perceptual Image Patch Similarity).
    Lower = more perceptually similar.
    """
    import torch
    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    a = _load_image(img_a_path)
    b = _load_image(img_b_path)

    if a.shape[:2] != b.shape[:2]:
        b = _resize_to_match(b, a)

    # LPIPS expects NCHW, values in [-1, 1]
    def to_tensor(x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        x = (x - 0.5) / 0.5  # [0,1] -> [-1,1]
        return x.to(device)

    with torch.no_grad():
        dist = lpips_fn(to_tensor(a), to_tensor(b))
    return float(dist.squeeze().cpu().numpy())


def compute_mse(img_a_path: Path, img_b_path: Path) -> float:
    """
    Mean Squared Error (pixel-wise).
    Lower = more similar. For images in [0, 1], MSE is in [0, 1].
    """
    a = _load_image(img_a_path)
    b = _load_image(img_b_path)

    if a.shape[:2] != b.shape[:2]:
        b = _resize_to_match(b, a)

    mse = float(np.mean((a - b) ** 2))
    return mse


def compute_metrics(img_a_path: Path, img_b_path: Path) -> dict:
    """Compute LPIPS, SSIM, and MSE for a pair of images."""
    return {
        "lpips": compute_lpips(img_a_path, img_b_path),
        "ssim": compute_ssim(img_a_path, img_b_path),
        "mse": compute_mse(img_a_path, img_b_path),
    }
