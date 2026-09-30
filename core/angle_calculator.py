"""Geometry primitives used by every exercise.

All functions work on 2-D image-plane points (x, y) in pixels with the
image convention that y grows downward.
"""
from __future__ import annotations

import math
from typing import Sequence

Point = Sequence[float]


def calculate_angle(point_a: Point, point_b: Point, point_c: Point) -> float:
    """Return the angle ABC in degrees (0-180), with B as the joint vertex.

    theta = arccos( (BA . BC) / (|BA| |BC|) )

    Raises ValueError when A or C coincides with B (angle undefined).
    """
    bax, bay = point_a[0] - point_b[0], point_a[1] - point_b[1]
    bcx, bcy = point_c[0] - point_b[0], point_c[1] - point_b[1]
    norm = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if norm == 0:
        raise ValueError("Angle undefined: a point coincides with the joint")
    cosine = (bax * bcx + bay * bcy) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def angle_from_vertical(start: Point, end: Point, upward: bool = True) -> float:
    """Angle (0-180 deg) between the segment start->end and the vertical.

    With ``upward=True`` 0 deg means the segment points straight up (e.g. a
    perfectly upright torso hip->shoulder); with ``upward=False`` 0 deg means
    it hangs straight down (e.g. a relaxed upper arm shoulder->elbow).
    """
    reference = (start[0], start[1] - 1) if upward else (start[0], start[1] + 1)
    return calculate_angle(reference, start, end)


def distance(p: Point, q: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def signed_distance_to_line(p: Point, a: Point, b: Point) -> float:
    """Perpendicular distance of ``p`` from line a->b, normalised by |ab|.

    Positive when ``p`` lies *below* the line in image coordinates (larger y),
    which for a push-up means sagging hips.
    """
    length = distance(a, b)
    if length == 0:
        raise ValueError("Degenerate line")
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    d = cross / (length * length)
    # cross sign depends on line direction; orient so "below" is positive
    return d if b[0] >= a[0] else -d
