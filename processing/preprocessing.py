"""Stage 1 - image preprocessing.

Every operation here is *adaptive*: we measure the frame first and only
apply an enhancement when the measurement says it is needed, so good
footage is not degraded by unnecessary filtering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from config.thresholds import PREPROCESSING, PROCESSING

# Immerkaer (1996) noise-estimation mask: difference of two Laplacians,
# which cancels image structure and leaves mostly noise.
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


@dataclass
class PreprocessResult:
    """Output of :func:`preprocess_frame`."""

    bgr: np.ndarray                  # processed frame (for drawing / display)
    rgb: np.ndarray                  # same frame in RGB (pose model input)
    scale: float                     # resize factor applied to the original
    brightness: float                # mean grey level of the ROI (0-255)
    contrast: float                  # grey-level std-dev of the ROI
    noise_sigma: float               # estimated Gaussian noise sigma
    applied: list[str] = field(default_factory=list)  # operations performed

    @property
    def poor_lighting(self) -> bool:
        return (self.brightness < PREPROCESSING["dark_mean_threshold"]
                or self.contrast < PREPROCESSING["low_contrast_std_threshold"])


def resize_frame(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Downscale (never upscale) so width <= max_width, preserving aspect ratio.

    INTER_AREA is used because it averages source pixels, which acts as an
    anti-aliasing low-pass filter when shrinking.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / w
    return cv2.resize(frame, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA), scale


def grey_statistics(grey: np.ndarray) -> tuple[float, float]:
    """Mean (brightness) and standard deviation (global contrast) of a grey image."""
    mean, std = cv2.meanStdDev(grey)
    return float(mean[0, 0]), float(std[0, 0])


def estimate_noise_sigma(grey: np.ndarray) -> float:
    """Fast noise standard-deviation estimate (Immerkaer, 1996).

    sigma = sqrt(pi/2) * 1/(6 (W-2)(H-2)) * sum |I * N|
    """
    h, w = grey.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    response = cv2.filter2D(grey.astype(np.float32), -1, _NOISE_KERNEL)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) * np.abs(response).sum() / (6 * (w - 2) * (h - 2)))


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation on the L channel.

    Working in LAB space equalises luminance only, so skin/clothing colours
    are preserved while shadows on the body become visible to the model.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=PREPROCESSING["clahe_clip_limit"],
                            tileGridSize=PREPROCESSING["clahe_tile_grid"])
    return cv2.cvtColor(cv2.merge((clahe.apply(l_channel), a, b)), cv2.COLOR_LAB2BGR)


def preprocess_frame(frame_bgr: np.ndarray,
                     roi: tuple[int, int, int, int] | None = None,
                     max_width: int | None = None,
                     enhance: bool = True) -> PreprocessResult:
    """Run the preprocessing chain on one BGR frame.

    Args:
        frame_bgr: raw camera/video frame.
        roi: (x1, y1, x2, y2) of the subject from the previous frame, in
            *processed* coordinates. Lighting is judged on the person, not on
            a bright window behind them.
        max_width: processing width (defaults to config).
        enhance: allow adaptive CLAHE / denoising.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty frame")
    if frame_bgr.ndim == 2:                       # greyscale source
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

    bgr, scale = resize_frame(frame_bgr, max_width or PROCESSING["max_width"])
    applied = ["resize"] if scale != 1.0 else []

    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    region = grey
    if roi is not None:
        x1, y1, x2, y2 = roi
        if x2 - x1 > 8 and y2 - y1 > 8:
            region = grey[y1:y2, x1:x2]
    brightness, contrast = grey_statistics(region)
    noise_sigma = estimate_noise_sigma(grey)

    if enhance:
        needs_clahe = (brightness < PREPROCESSING["dark_mean_threshold"]
                       or contrast < PREPROCESSING["low_contrast_std_threshold"])
        noise_limit = PREPROCESSING["noise_sigma_threshold_dark" if needs_clahe
                                    else "noise_sigma_threshold"]
        if noise_sigma > noise_limit:          # denoise first: CLAHE would amplify it
            bgr = cv2.GaussianBlur(bgr, PREPROCESSING["gaussian_kernel"], 0)
            applied.append("gaussian_denoise")
        if needs_clahe:
            bgr = apply_clahe(bgr)
            applied.append("clahe")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # MediaPipe expects RGB
    return PreprocessResult(bgr, rgb, scale, brightness, contrast, noise_sigma, applied)
