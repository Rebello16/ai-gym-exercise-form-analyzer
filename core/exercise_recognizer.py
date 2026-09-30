"""Stage 5 - automatic exercise recognition from pose geometry.

Rule-based classifier over a sliding time window of features. Three cues
separate the four exercises:

* body orientation  - ankle->shoulder line near horizontal => push-up
* which joint moves - knee range of motion => squat, elbow ROM => arm exercise
* hand height       - wrists above shoulders => press, below => curl

Each exercise gets a soft score in [0, 1] (products of clipped linear
ramps); the highest score wins if it clears ``min_confidence``. A majority
vote over recent frames stops the label from flickering.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from config.thresholds import RECOGNITION

UNKNOWN = "unknown"


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at ``lo``, 1 at ``hi``, linear in between (works for lo > hi too)."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class Recognition:
    label: str
    confidence: float
    scores: dict[str, float]


class ExerciseRecognizer:
    def __init__(self):
        self._window: deque[tuple[float, dict]] = deque()
        self._votes: deque[str] = deque(maxlen=RECOGNITION["vote_frames"])
        self.current = Recognition(UNKNOWN, 0.0, {})

    def reset(self) -> None:
        self._window.clear()
        self._votes.clear()
        self.current = Recognition(UNKNOWN, 0.0, {})

    @staticmethod
    def classify(samples: list[dict]) -> Recognition:
        """Score each exercise from a list of feature dicts (one window)."""
        cfg = RECOGNITION

        def col(name: str) -> np.ndarray:
            return np.array([s[name] for s in samples if s.get(name) is not None], dtype=float)

        def rom(name: str) -> float:
            v = col(name)
            return float(np.percentile(v, 95) - np.percentile(v, 5)) if len(v) >= cfg["min_samples"] else 0.0

        wrist = col("wrist_above_shoulder")
        if len(samples) < cfg["min_samples"]:
            return Recognition(UNKNOWN, 0.0, {})
        # Orientation from the whole body line; fall back to the torso when
        # the legs are out of frame, and to "upright" for upper-body framing.
        incl = col("body_inclination")
        if len(incl) < cfg["min_samples"]:
            incl = col("torso_inclination")
        orientation = float(np.median(incl)) if len(incl) >= cfg["min_samples"] else 0.0
        horizontal = _ramp(orientation, cfg["horizontal_body_deg"] - 15,
                           cfg["horizontal_body_deg"] + 15)
        upright = 1.0 - horizontal
        knee_moves = _ramp(rom("knee_angle"), cfg["min_rom_deg"], cfg["rom_full_deg"])
        elbow_moves = _ramp(rom("elbow_angle"), cfg["min_rom_deg"], cfg["rom_full_deg"])
        hands_high = _ramp(float(np.median(wrist)), -0.1, 0.2) if len(wrist) else 0.0

        scores = {
            "pushup": horizontal * max(elbow_moves, 0.6),   # a held plank still reads as push-up
            "squat": upright * knee_moves,
            "shoulder_press": upright * elbow_moves * hands_high * (1 - knee_moves),
            "bicep_curl": upright * elbow_moves * (1 - hands_high) * (1 - knee_moves),
        }
        label, conf = max(scores.items(), key=lambda kv: kv[1])
        if conf < cfg["min_confidence"]:
            label = UNKNOWN
        return Recognition(label, conf, scores)

    def update(self, t: float, features: dict) -> Recognition:
        """Add one frame and return the (vote-stabilised) recognition."""
        self._window.append((t, features))
        while self._window and t - self._window[0][0] > RECOGNITION["window_seconds"]:
            self._window.popleft()
        raw = self.classify([f for _, f in self._window])
        self._votes.append(raw.label)
        winner, n = Counter(self._votes).most_common(1)[0]
        # Switch only to a *known* exercise holding a strict majority of recent
        # votes; resting (no movement -> "unknown") keeps the last exercise.
        if winner != UNKNOWN and n > len(self._votes) // 2:
            # confidence = the window score while the label is actively winning
            self.current = Recognition(winner, raw.scores.get(winner, 0.0), raw.scores)
        return self.current
