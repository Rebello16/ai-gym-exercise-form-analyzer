"""Stage 8 - transparent, metric-driven scoring.

check score  s = clip((v - zero) / (ideal - zero), 0, 1) * 100
rep score    S = sum(w_i * s_i) / sum(w_i)   over checks that could be measured

The sign of (ideal - zero) encodes the direction, so the same formula works
for "smaller is better" (knee angle at depth) and "larger is better"
(elbow angle at lockout).
"""
from __future__ import annotations

from typing import Iterable

from config.thresholds import SCORE_BANDS, STATUS_GOOD, STATUS_WARNING


def check_score(value: float, ideal: float, zero: float) -> float:
    """Linear score in [0, 100]: 100 at/beyond ``ideal``, 0 at/beyond ``zero``."""
    if ideal == zero:
        raise ValueError("ideal and zero thresholds must differ")
    ratio = (value - zero) / (ideal - zero)
    return max(0.0, min(1.0, ratio)) * 100.0


def weighted_score(items: Iterable[tuple[float | None, float]]) -> float | None:
    """Weighted mean of (score, weight) pairs, ignoring unmeasured (None) scores."""
    pairs = [(s, w) for s, w in items if s is not None]
    total_w = sum(w for _, w in pairs)
    if not pairs or total_w == 0:
        return None
    return sum(s * w for s, w in pairs) / total_w


def grade(score: float | None) -> str:
    """Human-readable band for a score, from ``SCORE_BANDS``."""
    if score is None:
        return "Not enough data"
    for minimum, label in SCORE_BANDS:
        if round(score) >= minimum:
            return label
    return SCORE_BANDS[-1][1]


def status(score: float | None) -> str:
    """Traffic-light status used for colouring: good / warning / bad / neutral."""
    if score is None:
        return "neutral"
    if score >= STATUS_GOOD:
        return "good"
    return "warning" if score >= STATUS_WARNING else "bad"
