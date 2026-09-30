"""End-to-end: rendered video -> real MediaPipe pipeline -> counts, scores, exports.

Uses short synthetic clips with known ground truth (see demo/synthetic_athlete.py).
"""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from core.pipeline import AnalysisPipeline
from core.pose_detector import PoseDetector
from demo.synthetic_athlete import SCENARIO_BY_NAME, render_scenario
from processing.video_processor import VideoSourceError, analyze_video
from utils.export import report_csv, report_pdf


@pytest.fixture(scope="module")
def detector_factory():
    made = []

    def make():
        d = PoseDetector()
        made.append(d)
        return d
    yield make
    for d in made:
        d.close()


def _analyse(tmp_path, name, detector, reps=3, **kw):
    sc = replace(SCENARIO_BY_NAME[name], reps=reps)
    path = render_scenario(sc, tmp_path / f"{name}.webm")
    pipe = AnalysisPipeline(detector, source=name, **kw)
    analyze_video(path, pipe, save_output=False)
    return pipe.session


@pytest.mark.parametrize("name, exercise", [
    ("squat_good", "squat"),
    ("bicep_curl_good", "bicep_curl"),
    ("pushup_good", "pushup"),
    ("shoulder_press_good", "shoulder_press"),
])
def test_auto_detection_counts_good_reps(tmp_path, detector_factory, name, exercise):
    session = _analyse(tmp_path, name, detector_factory())
    assert session.exercises_done() == [exercise]
    assert len(session.reps) == 3
    assert session.summary()["average_score"] >= 85


def test_bad_form_is_detected(tmp_path, detector_factory):
    session = _analyse(tmp_path, "squat_bad", detector_factory(), manual_exercise="squat")
    assert len(session.reps) == 3
    assert session.summary()["average_score"] < 75
    issues = " ".join(m for r in session.reps for m in r.issues)
    assert "deeper" in issues and "upright" in issues


def test_multiple_people_reported_and_primary_analysed(tmp_path, detector_factory):
    session = _analyse(tmp_path, "squat_two_people", detector_factory())
    assert max(f["num_people"] for f in session.frames) == 2
    assert len(session.reps) == 3 and session.exercises_done() == ["squat"]


def test_no_person_video_does_not_crash(tmp_path, detector_factory):
    path = tmp_path / "empty.webm"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), 30, (320, 240))
    for _ in range(30):
        writer.write(np.full((240, 320, 3), 128, np.uint8))
    writer.release()
    pipe = AnalysisPipeline(detector_factory(), source="empty")
    analyze_video(path, pipe, save_output=False)
    assert pipe.session.reps == [] and pipe.session.summary()["pose_detection_rate"] == 0
    _, status = pipe.process(np.full((240, 320, 3), 128, np.uint8), 5.0)
    assert any("No person" in w for w in status.warnings)
    assert "summary" in report_csv(pipe.session)          # exports still work when empty
    assert report_pdf(pipe.session).startswith(b"%PDF")


def test_invalid_video_raises_friendly_error(tmp_path, detector_factory):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(VideoSourceError):
        analyze_video(bad, AnalysisPipeline(detector_factory()), save_output=False)


def test_exports_contain_session_data(tmp_path, detector_factory):
    session = _analyse(tmp_path, "squat_good", detector_factory(), reps=2)
    csv = report_csv(session)
    for text in ("total_reps", "average_score", "angle_statistics", "knee_angle", "date_time"):
        assert text in csv
    assert report_pdf(session).startswith(b"%PDF")


def test_unknown_manual_exercise_rejected(detector_factory):
    with pytest.raises(ValueError):
        AnalysisPipeline(detector_factory(), manual_exercise="deadlift")
