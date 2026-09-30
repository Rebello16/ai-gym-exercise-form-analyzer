"""Exercise definitions, feature extraction, recognition and state transitions."""
import numpy as np
import pytest

from config.thresholds import EXERCISE_THRESHOLDS
from core.exercise_recognizer import UNKNOWN, ExerciseRecognizer
from core.feature_extractor import FEATURE_UNITS, FeatureExtractor
from core.pipeline import ExerciseTracker
from core.pose_detector import LM, PoseDetection
from core.rep_counter import RepState
from exercises import EXERCISES
from exercises.base_exercise import DURATION


def test_registry_has_required_exercises():
    assert set(EXERCISES) >= {"squat", "bicep_curl", "pushup", "shoulder_press"}


@pytest.mark.parametrize("key", list(EXERCISES))
def test_exercise_definition_is_consistent(key):
    ex = EXERCISES[key]
    rep = EXERCISE_THRESHOLDS[key]["rep"]
    assert rep["flexed"] < rep["extended"]
    assert len(ex.phase_names) == 4
    assert ex.rep_feature in FEATURE_UNITS
    for check in ex.all_checks:
        ideal, zero = ex.check_range(check)
        assert ideal != zero
        assert check.feature == DURATION or check.feature in FEATURE_UNITS
        assert check.reduce in ("min", "max", "range")
        check.message.format(value=1.0)          # template must format


# ---------------------------------------------------------------------------
def make_pose(joints: dict[str, tuple[float, float]], hidden: tuple[str, ...] = ()) -> PoseDetection:
    """Synthetic landmarks: both body sides at the given positions."""
    pts = np.zeros((33, 2), np.float32)
    vis = np.zeros(33, np.float32)
    for name, xy in joints.items():
        for side in ("left", "right"):
            idx = LM[f"{side}_{name}"]
            pts[idx] = xy
            vis[idx] = 0.0 if name in hidden else 0.99
    return PoseDetection(pts, vis, 1)


STANDING = {"shoulder": (100, 100), "elbow": (100, 160), "wrist": (100, 220), "hip": (100, 250),
            "knee": (100, 350), "ankle": (100, 450), "heel": (90, 460), "foot_index": (140, 462)}


def test_feature_extractor_standing():
    f = FeatureExtractor(ema_alpha=1.0).extract(make_pose(STANDING))
    assert f.get("knee_angle") == pytest.approx(180)
    assert f.get("elbow_angle") == pytest.approx(180)
    assert f.get("torso_inclination") == pytest.approx(0)
    assert f.get("upper_arm_elevation") == pytest.approx(0)
    assert f.get("wrist_above_shoulder") < 0
    assert f.view == "side"


def test_feature_extractor_right_angle_knee():
    joints = dict(STANDING, knee=(150, 350), hip=(50, 350))
    f = FeatureExtractor(ema_alpha=1.0).extract(make_pose(joints))
    # knee->hip = (-100, 0), knee->ankle = (-50, 100)
    assert f.get("knee_angle") == pytest.approx(
        np.degrees(np.arccos(np.dot((-100, 0), (-50, 100)) / (100 * np.hypot(50, 100)))))


def test_missing_landmarks_give_none_not_guesses():
    f = FeatureExtractor(ema_alpha=1.0).extract(make_pose(STANDING, hidden=("knee",)))
    assert f.get("knee_angle") is None and f.get("hip_angle") is None
    assert f.get("elbow_angle") == pytest.approx(180)       # unaffected features still work


def test_no_person_gives_empty_features():
    f = FeatureExtractor().extract(PoseDetection())
    assert f.values == {}


def test_ema_smoothing():
    fx = FeatureExtractor(ema_alpha=0.5)
    fx.extract(make_pose(STANDING))
    bent = dict(STANDING, wrist=(160, 160))                     # elbow 90 deg
    assert fx.extract(make_pose(bent)).get("elbow_angle") == pytest.approx(135)


# ---------------------------------------------------------------------------
def _window(**series):
    n = len(next(iter(series.values())))
    return [{k: v[i] for k, v in series.items()} for i in range(n)]


WAVE = (1 - np.cos(np.linspace(0, 2 * np.pi, 40))) / 2


@pytest.mark.parametrize("expected, window", [
    ("squat", _window(knee_angle=175 - 85 * WAVE, elbow_angle=[178] * 40,
                      torso_inclination=30 * WAVE, body_inclination=10 * WAVE,
                      wrist_above_shoulder=[0.0] * 40)),
    ("bicep_curl", _window(knee_angle=[178] * 40, elbow_angle=165 - 120 * WAVE,
                           torso_inclination=[2] * 40, body_inclination=[3] * 40,
                           wrist_above_shoulder=-1 + 0.8 * WAVE)),
    ("shoulder_press", _window(knee_angle=[178] * 40, elbow_angle=85 + 90 * WAVE,
                               torso_inclination=[2] * 40, body_inclination=[3] * 40,
                               wrist_above_shoulder=0.4 + 0.6 * WAVE)),
    ("pushup", _window(knee_angle=[176] * 40, elbow_angle=165 - 75 * WAVE,
                       torso_inclination=[70] * 40, body_inclination=[72] * 40,
                       wrist_above_shoulder=[-0.9] * 40)),
])
def test_recognizer_classifies_geometry(expected, window):
    assert ExerciseRecognizer.classify(window).label == expected


def test_recognizer_idle_is_unknown_and_sticky():
    idle = _window(knee_angle=[178] * 40, elbow_angle=[175] * 40, torso_inclination=[2] * 40,
                   body_inclination=[2] * 40, wrist_above_shoulder=[-1.0] * 40)
    assert ExerciseRecognizer.classify(idle).label == UNKNOWN
    rec = ExerciseRecognizer()
    squat = _window(knee_angle=175 - 85 * WAVE, torso_inclination=30 * WAVE,
                    body_inclination=10 * WAVE, elbow_angle=[178] * 40, wrist_above_shoulder=[0] * 40)
    for i, f in enumerate(squat):
        rec.update(i / 20, f)
    assert rec.current.label == "squat"
    for i, f in enumerate(idle * 3):                  # resting keeps the last exercise
        rec.update(2 + i / 20, f)
    assert rec.current.label == "squat"


def test_recognizer_needs_enough_samples():
    assert ExerciseRecognizer.classify([{"torso_inclination": 2}]).label == UNKNOWN


# ---------------------------------------------------------------------------
def test_tracker_state_transitions_and_rep_result():
    tracker = ExerciseTracker(EXERCISES["squat"])
    phases, results = [], []
    knee = list(np.full(15, 175.0)) + list(175 - 85 * WAVE) + list(np.full(15, 175.0))
    for i, k in enumerate(knee):
        r = tracker.update(i / 20, {"knee_angle": k, "torso_inclination": 25.0, "knee_over_toe": 0.1})
        phases.append(tracker.phase)
        if r:
            results.append(r)
    names = EXERCISES["squat"].phase_names
    # visits every phase in order
    order = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert order == ["GET IN POSITION", *names, names[0]]
    assert len(results) == 1 and results[0].kind == "rep"
    assert results[0].score > 90
    assert tracker.counter.state == RepState.START
