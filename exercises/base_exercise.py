"""Declarative exercise definition.

Adding a new exercise = one new module that builds an :class:`Exercise`
plus its numbers in ``config/thresholds.py``. No pipeline code changes.
"""
from __future__ import annotations

from dataclasses import dataclass

from config.thresholds import EXERCISE_THRESHOLDS, TEMPO

DURATION = "_rep_duration"   # pseudo-feature: seconds taken by the rep movement


@dataclass(frozen=True)
class FormCheck:
    """One measurable form rule.

    Attributes:
        key: id of the (ideal, zero) pair in config.
        label: name shown in the score breakdown.
        feature: feature name from :mod:`core.feature_extractor` (or DURATION).
        reduce: how the per-frame values of one rep are summarised:
            "min", "max" or "range" (max - min).
        message: feedback template shown when the check scores poorly;
            ``{value}`` is replaced by the measured value.
        weight: contribution to the rep score.
        live: whether a single frame is enough to judge it (posture checks),
            used for real-time colouring before the rep finishes.
    """

    key: str
    label: str
    feature: str
    reduce: str
    message: str
    weight: float = 1.0
    live: bool = False


TEMPO_CHECK = FormCheck("tempo", "Control (tempo)", DURATION, "max",
                        "Rep was fast ({value:.1f} s): slow down and control the movement",
                        weight=0.5)


@dataclass(frozen=True)
class Exercise:
    """Everything the pipeline needs to know about one exercise.

    Attributes:
        rep_feature: joint angle that drives the repetition state machine.
        start_extended: True if a rep starts with that joint extended
            (squat, curl, push-up); False if it starts flexed (press).
        phase_names: labels for (start, moving to turn, turn, returning).
        display_features: angles drawn on the skeleton.
    """

    key: str
    name: str
    rep_feature: str
    start_extended: bool
    phase_names: tuple[str, str, str, str]
    checks: tuple[FormCheck, ...]
    display_features: tuple[str, ...]
    description: str = ""

    @property
    def all_checks(self) -> tuple[FormCheck, ...]:
        return self.checks + (TEMPO_CHECK,)

    @property
    def rep_thresholds(self) -> dict:
        return EXERCISE_THRESHOLDS[self.key]["rep"]

    def check_range(self, check: FormCheck) -> tuple[float, float]:
        """(ideal, zero) thresholds for a check."""
        if check.key == "tempo":
            return TEMPO["ideal"], TEMPO["zero"]
        return EXERCISE_THRESHOLDS[self.key]["checks"][check.key]
