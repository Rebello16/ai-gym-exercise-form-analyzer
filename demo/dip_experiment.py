"""Experiment: does the DIP preprocessing help pose detection in bad lighting?

Darkens (and optionally adds Gaussian noise to) the frames of a demo clip,
then measures the pose-detection rate for three inputs:

* raw        - degraded frame straight into the model
* CLAHE only - contrast enhancement without denoising
* adaptive   - the project's preprocessing chain (measure -> denoise -> CLAHE)

Run:  python -m demo.dip_experiment
"""
from __future__ import annotations

import cv2
import numpy as np

from core.pose_detector import PoseDetector
from demo.synthetic_athlete import ensure_samples
from processing.preprocessing import apply_clahe, preprocess_frame
from processing.video_processor import iterate_video

CONDITIONS = [(1.0, 0), (0.2, 6), (0.1, 0), (0.1, 4), (0.1, 6), (0.06, 0)]


def degrade(frame: np.ndarray, brightness: float, noise_sd: float, rng) -> np.ndarray:
    out = cv2.convertScaleAbs(frame, alpha=brightness)
    if noise_sd:
        out = np.clip(out + rng.normal(0, noise_sd, out.shape), 0, 255).astype(np.uint8)
    return out


def detection_rate(frames: list[np.ndarray], to_rgb) -> float:
    detector = PoseDetector()
    try:
        return float(np.mean([detector.detect(to_rgb(f), i * 100).detected
                              for i, f in enumerate(frames)]))
    finally:
        detector.close()


def main(clip: str = "squat_good", step: int = 3) -> None:
    path = ensure_samples([clip])[clip]          # renders only if missing
    frames = [f for f, _, _ in iterate_video(path, step)]
    methods = {
        "raw": lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
        "CLAHE only": lambda x: cv2.cvtColor(apply_clahe(x), cv2.COLOR_BGR2RGB),
        "adaptive pipeline": lambda x: preprocess_frame(x).rgb,
    }
    print(f"Pose-detection rate on {len(frames)} frames of '{clip}'\n")
    print("| brightness | noise sd | " + " | ".join(methods) + " | operations chosen |")
    print("|---|---|" + "---|" * (len(methods) + 1))
    for brightness, noise in CONDITIONS:
        rng = np.random.default_rng(0)
        degraded = [degrade(f, brightness, noise, rng) for f in frames]
        rates = [detection_rate(degraded, fn) for fn in methods.values()]
        ops = "+".join(preprocess_frame(degraded[0]).applied) or "none"
        print(f"| {brightness:.0%} | {noise} | " + " | ".join(f"{r:.2f}" for r in rates) + f" | {ops} |")


if __name__ == "__main__":
    main()
