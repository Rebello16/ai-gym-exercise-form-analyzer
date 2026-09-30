"""Stage 7 - form analysis: turn a rep's feature history into check scores.

Each :class:`~exercises.base_exercise.FormCheck` reduces the per-frame
values of one feature over the rep (min / max / range), then scores the
result against its configured (ideal, zero) thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config.thresholds import FEATURES, STATUS_GOOD
from core.scoring import check_score, weighted_score
from exercises.base_exercise import DURATION, Exercise, FormCheck

_REDUCERS = {
    "min": min,
    "max": max,
    "range": lambda values: max(values) - min(values),
}


def median_filter(series: list[float], window: int | None = None) -> list[float]:
    """1-D running median (edges use a shrunken window).

    A median rejects impulse noise - a single mis-detected frame - without
    shaving the plateau at the bottom/top of a rep the way a mean would.
    """
    k = (window or FEATURES["median_window"]) // 2
    if k == 0 or len(series) < 3:
        return list(series)
    a = np.asarray(series, dtype=float)
    return [float(np.median(a[max(0, i - k): i + k + 1])) for i in range(len(a))]


@dataclass
class RepResult:
    """Analysis of one completed (or partial) repetition."""

    index: int
    exercise: str
    kind: str                     # "rep" or "partial"
    start_time: float
    end_time: float
    values: dict[str, float | None] = field(default_factory=dict)  # per check
    scores: dict[str, float | None] = field(default_factory=dict)  # per check
    score: float | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class LiveCheck:
    check: FormCheck
    value: float
    score: float


class FormAnalyzer:
    """Evaluates the form checks of one exercise."""

    def __init__(self, exercise: Exercise):
        self.exercise = exercise

    def _score(self, check: FormCheck, value: float) -> float:
        ideal, zero = self.exercise.check_range(check)
        return check_score(value, ideal, zero)

    def evaluate_rep(self, frames: list[dict[str, float | None]], start_time: float,
                     end_time: float, index: int, kind: str = "rep") -> RepResult:
        """Score a rep from the feature dicts of every frame inside it."""
        result = RepResult(index, self.exercise.key, kind, start_time, end_time)
        for check in self.exercise.all_checks:
            if check.feature == DURATION:
                value: float | None = end_time - start_time
            else:
                series = [f[check.feature] for f in frames if f.get(check.feature) is not None]
                value = _REDUCERS[check.reduce](median_filter(series)) if series else None
            score = None if value is None else self._score(check, value)
            result.values[check.key] = value
            result.scores[check.key] = score
            if score is not None and score < STATUS_GOOD:
                result.issues.append(check.message.format(value=value))
        result.score = weighted_score(
            (result.scores[c.key], c.weight) for c in self.exercise.all_checks)
        return result

    def live(self, features: dict[str, float | None]) -> list[LiveCheck]:
        """Instantaneous posture checks on a single frame."""
        out = []
        for check in self.exercise.checks:
            value = features.get(check.feature)
            if check.live and value is not None:
                out.append(LiveCheck(check, value, self._score(check, value)))
        return out
