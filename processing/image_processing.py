"""Stage 3 - post-detection image processing on the subject.

* Region of interest (ROI) from landmarks
* Edge-based sharpness measure (variance of the Laplacian)
* Binary person mask from the segmentation output + morphology
* Optional background suppression using that mask
"""
from __future__ import annotations

import cv2
import numpy as np

from config.thresholds import PREPROCESSING


def landmarks_roi(points: np.ndarray, visibility: np.ndarray, shape: tuple[int, ...],
                  min_visibility: float = 0.5) -> tuple[int, int, int, int] | None:
    """Padded bounding box (x1, y1, x2, y2) around the visible landmarks."""
    visible = points[visibility >= min_visibility]
    if len(visible) < 4:
        return None
    h, w = shape[:2]
    x1, y1 = visible.min(axis=0)
    x2, y2 = visible.max(axis=0)
    pad_x = (x2 - x1) * PREPROCESSING["roi_padding"]
    pad_y = (y2 - y1) * PREPROCESSING["roi_padding"]
    box = (int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y)),
           int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y)))
    return box if box[2] > box[0] and box[3] > box[1] else None


def laplacian_edges(grey: np.ndarray) -> np.ndarray:
    """Second-derivative edge response (float64)."""
    return cv2.Laplacian(grey, cv2.CV_64F, ksize=3)


def sharpness(bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> float:
    """Variance of the Laplacian: low values mean few edges -> blur/motion blur."""
    if roi is not None:
        x1, y1, x2, y2 = roi
        bgr = bgr[y1:y2, x1:x2]
    if bgr.size == 0:
        return 0.0
    return float(laplacian_edges(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)).var())


def binary_person_mask(probability: np.ndarray) -> np.ndarray:
    """Threshold a [0,1] segmentation map and clean it with morphology.

    Opening removes isolated false-positive specks; closing fills small holes
    inside the silhouette. Returns a uint8 mask with values {0, 255}.
    """
    mask = (probability > PREPROCESSING["mask_threshold"]).astype(np.uint8) * 255
    k = PREPROCESSING["mask_morph_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def suppress_background(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blur and darken everything outside the person mask (foreground focus)."""
    background = cv2.GaussianBlur(bgr, PREPROCESSING["background_blur_kernel"], 0) // 2
    alpha = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32)[..., None] / 255.0
    return (bgr * alpha + background * (1 - alpha)).astype(np.uint8)


def edge_map_for_display(bgr: np.ndarray) -> np.ndarray:
    """Min-max normalised |Laplacian| as a viewable uint8 image."""
    edges = np.abs(laplacian_edges(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
    return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
