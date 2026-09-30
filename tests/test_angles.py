import math

import pytest

from core.angle_calculator import (angle_from_vertical, calculate_angle, distance,
                                   signed_distance_to_line)


@pytest.mark.parametrize("a, b, c, expected", [
    ((1, 0), (0, 0), (0, 1), 90.0),        # right angle
    ((-1, 0), (0, 0), (1, 0), 180.0),      # straight limb
    ((1, 0), (0, 0), (1, 0), 0.0),         # fully folded
    ((1, 0), (0, 0), (1, 1), 45.0),
    ((2, 0), (0, 0), (-1, math.sqrt(3)), 120.0),
])
def test_calculate_angle_known_values(a, b, c, expected):
    assert calculate_angle(a, b, c) == pytest.approx(expected, abs=1e-6)


def test_angle_is_scale_and_translation_invariant():
    base = calculate_angle((3, 1), (1, 1), (1, 4))
    moved = calculate_angle((3 * 5 + 7, 1 * 5 - 2), (1 * 5 + 7, 1 * 5 - 2), (1 * 5 + 7, 4 * 5 - 2))
    assert base == pytest.approx(moved)


def test_angle_is_symmetric():
    assert calculate_angle((5, 2), (1, 1), (0, 7)) == pytest.approx(calculate_angle((0, 7), (1, 1), (5, 2)))


def test_coincident_points_raise():
    with pytest.raises(ValueError):
        calculate_angle((0, 0), (0, 0), (1, 1))


def test_angle_from_vertical_image_coordinates():
    # y grows downward: a point straight above has smaller y
    assert angle_from_vertical((0, 10), (0, 0)) == pytest.approx(0)            # upright
    assert angle_from_vertical((0, 10), (10, 10)) == pytest.approx(90)         # horizontal
    assert angle_from_vertical((0, 0), (0, 10), upward=False) == pytest.approx(0)  # hanging


def test_signed_distance_positive_below_line():
    a, b = (0, 0), (10, 0)
    assert signed_distance_to_line((5, 2), a, b) == pytest.approx(0.2)
    assert signed_distance_to_line((5, -2), a, b) == pytest.approx(-0.2)
    # same result if the line is given in the opposite direction
    assert signed_distance_to_line((5, 2), b, a) == pytest.approx(0.2)


def test_distance():
    assert distance((0, 0), (3, 4)) == 5
