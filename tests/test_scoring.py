import pytest

from config.thresholds import SCORE_BANDS
from core.form_analyzer import FormAnalyzer, median_filter
from core.scoring import check_score, grade, status, weighted_score
from exercises import EXERCISES


@pytest.mark.parametrize("value, ideal, zero, expected", [
    (90, 90, 140, 100),     # at ideal
    (70, 90, 140, 100),     # better than ideal -> clipped
    (115, 90, 140, 50),     # halfway
    (150, 90, 140, 0),      # worse than zero -> clipped
    (160, 165, 135, 83.33),  # "larger is better" direction
])
def test_check_score(value, ideal, zero, expected):
    assert check_score(value, ideal, zero) == pytest.approx(expected, abs=0.01)


def test_check_score_rejects_equal_thresholds():
    with pytest.raises(ValueError):
        check_score(1, 5, 5)


def test_weighted_score_ignores_unmeasured_checks():
    assert weighted_score([(100, 1), (50, 1), (None, 5)]) == 75
    assert weighted_score([(100, 3), (0, 1)]) == 75
    assert weighted_score([(None, 1)]) is None


def test_grade_bands():
    assert grade(100) == SCORE_BANDS[0][1]
    assert grade(95) == "Very good"
    assert grade(80) == "Acceptable"
    assert grade(65) == "Needs improvement"
    assert grade(10) == "Poor detected form"
    assert grade(None) == "Not enough data"


def test_status_colours():
    assert status(90) == "good" and status(60) == "warning" and status(20) == "bad"
    assert status(None) == "neutral"


def _squat_frames(min_knee, max_lean):
    frames = []
    for i in range(40):
        p = 1 - abs(i - 20) / 20
        frames.append({"knee_angle": 175 - (175 - min_knee) * p,
                       "torso_inclination": max_lean * p, "knee_over_toe": 0.1,
                       "knee_width_ratio": None})
    return frames


def test_form_analyzer_good_squat_scores_high():
    result = FormAnalyzer(EXERCISES["squat"]).evaluate_rep(_squat_frames(88, 30), 0.0, 1.5, 1)
    assert result.score == pytest.approx(100)
    assert result.issues == []
    assert result.scores["knee_alignment"] is None          # not measurable from the side


def test_form_analyzer_detects_shallow_and_leaning_squat():
    result = FormAnalyzer(EXERCISES["squat"]).evaluate_rep(_squat_frames(125, 65), 0.0, 1.5, 1)
    assert result.scores["depth"] < 50 and result.scores["back_angle"] < 50
    assert any("deeper" in m for m in result.issues)
    assert any("upright" in m for m in result.issues)
    assert result.score < 75


def test_single_frame_spike_is_filtered():
    frames = _squat_frames(88, 30)
    frames[5]["torso_inclination"] = 89.0                      # one bad landmark frame
    result = FormAnalyzer(EXERCISES["squat"]).evaluate_rep(frames, 0.0, 1.5, 1)
    assert result.scores["back_angle"] == pytest.approx(100)


def test_median_filter():
    assert median_filter([1, 1, 9, 1, 1]) == [1, 1, 1, 1, 1]
    assert median_filter([1, 2]) == [1, 2]


def test_live_checks_only_use_posture_rules():
    live = FormAnalyzer(EXERCISES["squat"]).live({"knee_angle": 90, "torso_inclination": 65})
    assert [lc.check.key for lc in live] == ["back_angle"]
    assert live[0].score < 50
