"""Stage 6 - repetition counting with a hysteresis state machine.

States (for a rep that starts extended, e.g. squat)::

    WAITING --(enter start zone)--> START
    START   --(leave start zone)--> TO_TURN
    TO_TURN --(enter turn zone)---> TURN          (back to START = partial rep)
    TURN    --(leave turn zone)---> TO_START
    TO_START--(enter start zone)--> START  => +1 rep

The start zone is ``angle >= extended`` and the turn zone is
``angle <= flexed`` (swapped for exercises that start flexed). The gap
between the two thresholds is a dead band: jitter inside it can never
produce a count. Zone entry must also persist for ``confirm_frames``
consecutive frames (debouncing).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config.thresholds import POSE


class RepState(str, Enum):
    WAITING = "WAITING"
    START = "START"
    TO_TURN = "TO_TURN"
    TURN = "TURN"
    TO_START = "TO_START"


@dataclass
class RepEvent:
    """Emitted when a movement finishes."""

    kind: str            # "rep" (counted) or "partial" (turn zone never reached)
    start_time: float    # when the joint left the start zone
    end_time: float      # when it re-entered the start zone
    extreme: float       # most-flexed (or most-extended) value reached

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class RepCounter:
    """Counts repetitions of one joint-angle signal."""

    def __init__(self, extended: float, flexed: float, start_extended: bool = True,
                 min_rep_seconds: float = 0.6, confirm_frames: int = 2,
                 partial_min_deg: float = 15.0,
                 max_missing_seconds: float | None = None):
        if flexed >= extended:
            raise ValueError("`flexed` threshold must be below `extended` threshold")
        self.extended, self.flexed = extended, flexed
        self.start_extended = start_extended
        self.min_rep_seconds = min_rep_seconds
        self.confirm_frames = confirm_frames
        self.partial_min_deg = partial_min_deg
        self.max_missing = POSE["max_missing_seconds"] if max_missing_seconds is None else max_missing_seconds
        self.count = 0
        self.state = RepState.WAITING
        self._streak_zone: str | None = None
        self._streak = 0
        self._move_start = 0.0
        self._extreme = 0.0
        self._last_seen: float | None = None

    # -- helpers ----------------------------------------------------------
    def _zone(self, value: float) -> str | None:
        if value >= self.extended:
            return "start" if self.start_extended else "turn"
        if value <= self.flexed:
            return "turn" if self.start_extended else "start"
        return None

    def _further(self, a: float, b: float) -> float:
        """The value further toward the turn direction."""
        return min(a, b) if self.start_extended else max(a, b)

    @property
    def in_rep(self) -> bool:
        return self.state in (RepState.TO_TURN, RepState.TURN, RepState.TO_START)

    def reset(self) -> None:
        """Abandon any in-progress rep (count is kept)."""
        self.state = RepState.WAITING
        self._streak_zone, self._streak = None, 0

    # -- main -------------------------------------------------------------
    def update(self, value: float | None, t: float) -> RepEvent | None:
        """Feed one sample (``None`` = joint not visible). Returns a finished movement."""
        if value is None:
            if self._last_seen is not None and t - self._last_seen > self.max_missing:
                self.reset()          # person left frame: don't stitch two halves together
            return None
        self._last_seen = t

        zone = self._zone(value)
        if zone == self._streak_zone:
            self._streak += 1
        else:
            self._streak_zone, self._streak = zone, 1
        confirmed = zone if self._streak >= self.confirm_frames else None

        if self.in_rep:
            self._extreme = self._further(self._extreme, value)

        s = self.state
        if s == RepState.WAITING:
            if confirmed == "start":
                self.state = RepState.START
        elif s == RepState.START:
            if zone != "start":
                self.state = RepState.TO_TURN
                self._move_start, self._extreme = t, value
        elif s == RepState.TO_TURN:
            if confirmed == "turn":
                self.state = RepState.TURN
            elif confirmed == "start":
                self.state = RepState.START
                start_thr = self.extended if self.start_extended else self.flexed
                if abs(self._extreme - start_thr) >= self.partial_min_deg:
                    return RepEvent("partial", self._move_start, t, self._extreme)
        elif s == RepState.TURN:
            if zone != "turn":
                self.state = RepState.TO_START
        elif s == RepState.TO_START:
            if confirmed == "turn":
                self.state = RepState.TURN
            elif confirmed == "start":
                self.state = RepState.START
                if t - self._move_start >= self.min_rep_seconds:
                    self.count += 1
                    return RepEvent("rep", self._move_start, t, self._extreme)
        return None
