"""Stage 4 - feature extraction: landmarks -> geometric features.

Features are joint angles (degrees) and scale-invariant ratios (distances
divided by a body-segment length, so the result does not depend on how far
the person stands from the camera).

A feature is ``None`` whenever one of its landmarks is not visible enough;
downstream stages treat ``None`` as "unknown" instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config.thresholds import FEATURES, POSE
from core.angle_calculator import (angle_from_vertical, calculate_angle, distance,
                                   signed_distance_to_line)
from core.pose_detector import LM, PoseDetection

# Joint vertex used when drawing an angle feature on the skeleton
FEATURE_VERTEX = {
    "knee_angle": "knee", "hip_angle": "hip", "ankle_angle": "ankle",
    "elbow_angle": "elbow", "shoulder_angle": "shoulder",
}

FEATURE_UNITS = {
    "knee_angle": "deg", "hip_angle": "deg", "ankle_angle": "deg",
    "elbow_angle": "deg", "shoulder_angle": "deg", "torso_inclination": "deg",
    "upper_arm_elevation": "deg", "body_inclination": "deg", "forearm_from_vertical": "deg",
    "body_line_angle": "deg", "wrist_above_shoulder": "torso lengths",
    "hip_above_knee": "thigh lengths", "knee_over_toe": "shin lengths",
    "hip_line_offset": "body lengths", "knee_width_ratio": "ratio",
}

_SIDE_JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


@dataclass
class Features:
    """Per-frame features for the analysed body side."""

    values: dict[str, float | None] = field(default_factory=dict)
    side: str = "left"
    view: str = "side"        # "side" or "front"

    def get(self, name: str) -> float | None:
        return self.values.get(name)


class FeatureExtractor:
    """Stateful extractor: keeps the chosen body side stable and smooths values."""

    def __init__(self, ema_alpha: float | None = None):
        self.alpha = FEATURES["ema_alpha"] if ema_alpha is None else ema_alpha
        self.side = "left"
        self._side_vis = {"left": 0.0, "right": 0.0}
        self._ema: dict[str, float] = {}

    # -- helpers ----------------------------------------------------------
    def _update_side(self, vis: np.ndarray) -> None:
        """Pick the body side facing the camera, with hysteresis."""
        for side in ("left", "right"):
            v = float(np.mean([vis[LM[f"{side}_{j}"]] for j in _SIDE_JOINTS]))
            self._side_vis[side] = 0.7 * self._side_vis[side] + 0.3 * v
        other = "right" if self.side == "left" else "left"
        if self._side_vis[other] > self._side_vis[self.side] + FEATURES["side_switch_margin"]:
            self.side = other

    def _smooth(self, name: str, value: float | None) -> float | None:
        """Exponential moving average: y_t = a*x_t + (1-a)*y_(t-1)."""
        if value is None:
            self._ema.pop(name, None)
            return None
        prev = self._ema.get(name)
        smoothed = value if prev is None else self.alpha * value + (1 - self.alpha) * prev
        self._ema[name] = smoothed
        return smoothed

    def reset(self) -> None:
        self._ema.clear()

    # -- main -------------------------------------------------------------
    def extract(self, pose: PoseDetection) -> Features:
        if not pose.detected:
            self.reset()
            return Features(side=self.side)
        pts, vis = pose.points, pose.visibility
        thr = POSE["landmark_visibility_threshold"]
        self._update_side(vis)
        s = self.side

        def p(name: str, side: str | None = s) -> np.ndarray | None:
            idx = LM[f"{side}_{name}"] if side else LM[name]
            return pts[idx] if vis[idx] >= thr else None

        def safe(fn: Callable[..., float], *points) -> float | None:
            if any(q is None for q in points):
                return None
            try:
                return float(fn(*points))
            except (ValueError, ZeroDivisionError):
                return None

        sh, el, wr = p("shoulder"), p("elbow"), p("wrist")
        hip, knee, ank = p("hip"), p("knee"), p("ankle")
        heel, toe = p("heel"), p("foot_index")

        torso_len = safe(distance, hip, sh)
        thigh_len = safe(distance, hip, knee)
        shin_len = safe(distance, knee, ank)

        facing = None
        if heel is not None and toe is not None and abs(toe[0] - heel[0]) > 1:
            facing = 1.0 if toe[0] > heel[0] else -1.0

        raw: dict[str, float | None] = {
            "knee_angle": safe(calculate_angle, hip, knee, ank),
            "hip_angle": safe(calculate_angle, sh, hip, knee),
            "ankle_angle": safe(calculate_angle, knee, ank, toe),
            "elbow_angle": safe(calculate_angle, sh, el, wr),
            "shoulder_angle": safe(calculate_angle, hip, sh, el),
            "torso_inclination": safe(angle_from_vertical, hip, sh),
            "body_inclination": safe(angle_from_vertical, ank, sh),
            "upper_arm_elevation": safe(lambda a, b: angle_from_vertical(a, b, upward=False), sh, el),
            "forearm_from_vertical": safe(angle_from_vertical, el, wr),
            "body_line_angle": safe(calculate_angle, sh, hip, ank),
            "hip_line_offset": safe(signed_distance_to_line, hip, sh, ank),
            "wrist_above_shoulder": safe(lambda a, b, n: (a[1] - b[1]) / n, sh, wr, torso_len),
            "hip_above_knee": safe(lambda a, b, n: (b[1] - a[1]) / n, hip, knee, thigh_len),
            "knee_over_toe": safe(lambda k, t, n, f: (k[0] - t[0]) * f / n, knee, toe, shin_len, facing),
            "knee_width_ratio": None,
        }

        # View detection: a wide shoulder line relative to torso = facing camera
        ls, rs = p("shoulder", "left"), p("shoulder", "right")
        view = "side"
        if ls is not None and rs is not None and torso_len:
            if abs(ls[0] - rs[0]) / torso_len > FEATURES["frontal_view_ratio"]:
                view = "front"
        if view == "front":
            lk, rk = p("knee", "left"), p("knee", "right")
            la, ra = p("ankle", "left"), p("ankle", "right")
            if all(q is not None for q in (lk, rk, la, ra)) and abs(la[0] - ra[0]) > 5:
                raw["knee_width_ratio"] = abs(lk[0] - rk[0]) / abs(la[0] - ra[0])
            raw["knee_over_toe"] = None   # depth-direction offset is not observable from the front

        values = {name: self._smooth(name, v) for name, v in raw.items()}
        return Features(values, s, view)
