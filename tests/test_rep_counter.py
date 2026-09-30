import math

import numpy as np
import pytest

from core.rep_counter import RepCounter, RepState

FPS = 30


def squat_signal(reps=3, bottom=85.0, top=175.0, rep_s=2.0, pause_s=0.5, lead_s=1.0,
                 noise=0.0, seed=0):
    """Synthetic knee-angle trace: top -> bottom -> top, `reps` times."""
    rng = np.random.default_rng(seed)
    total = lead_s + reps * (rep_s + pause_s) + 1.0
    out = []
    for i in range(int(total * FPS)):
        t = i / FPS - lead_s
        cycle = rep_s + pause_s
        p = 0.0
        if 0 <= t < reps * cycle and (t % cycle) < rep_s:
            p = (1 - math.cos(2 * math.pi * (t % cycle) / rep_s)) / 2
        out.append((i / FPS, top - (top - bottom) * p + rng.normal(0, noise)))
    return out


def run(counter, signal):
    events = [counter.update(v, t) for t, v in signal]
    return [e for e in events if e is not None]


def test_counts_clean_reps():
    c = RepCounter(extended=160, flexed=110)
    events = run(c, squat_signal(reps=4))
    assert c.count == 4
    assert [e.kind for e in events] == ["rep"] * 4


def test_noise_does_not_create_false_counts():
    c = RepCounter(extended=160, flexed=110)
    run(c, squat_signal(reps=3, noise=4.0))
    assert c.count == 3


def test_jitter_around_one_threshold_never_counts():
    """Oscillating across a single threshold is not a rep (hysteresis)."""
    c = RepCounter(extended=160, flexed=110)
    signal = [(i / FPS, 175.0) for i in range(30)]
    signal += [(1 + i / FPS, 158.0 if i % 2 else 162.0) for i in range(200)]
    run(c, signal)
    assert c.count == 0


def test_shallow_movement_is_partial_not_counted():
    c = RepCounter(extended=160, flexed=110)
    events = run(c, squat_signal(reps=2, bottom=125))
    assert c.count == 0
    assert [e.kind for e in events] == ["partial", "partial"]


def test_starting_mid_rep_is_not_counted():
    c = RepCounter(extended=160, flexed=110)
    signal = [(i / FPS, 90.0) for i in range(20)] + [(1 + i / FPS, 175.0) for i in range(20)]
    run(c, signal)
    assert c.count == 0 and c.state == RepState.START


def test_missing_frames_hold_state_briefly():
    c = RepCounter(extended=160, flexed=110, max_missing_seconds=1.0)
    signal = squat_signal(reps=1)
    # drop 10 frames (0.33 s) in the middle of the rep
    mid = len(signal) // 2
    signal = [(t, None if mid <= i < mid + 10 else v) for i, (t, v) in enumerate(signal)]
    run(c, signal)
    assert c.count == 1


def test_long_absence_discards_in_progress_rep():
    c = RepCounter(extended=160, flexed=110, max_missing_seconds=1.0)
    signal = [(i / FPS, 175.0) for i in range(20)]             # standing
    signal += [(1 + i / FPS, 90.0) for i in range(20)]          # at the bottom
    signal += [(2 + i / FPS, None) for i in range(60)]          # left frame for 2 s
    signal += [(4 + i / FPS, 175.0) for i in range(20)]         # back, standing
    run(c, signal)
    assert c.count == 0


def test_press_starts_flexed():
    c = RepCounter(extended=150, flexed=110, start_extended=False)
    signal = [(t, 260 - v) for t, v in squat_signal(reps=3, bottom=85, top=175)]  # 85 -> 175 -> 85
    run(c, signal)
    assert c.count == 3


def test_too_fast_reps_are_rejected():
    c = RepCounter(extended=160, flexed=110, min_rep_seconds=2.0)
    run(c, squat_signal(reps=2, rep_s=1.0))
    assert c.count == 0


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        RepCounter(extended=100, flexed=120)
