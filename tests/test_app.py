"""Headless UI test: drives app.py with Streamlit's AppTest harness."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui.pages import PAGES

APP = Path(__file__).resolve().parent.parent / "app.py"


def _app():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    return at


def _goto(at, page):
    at.session_state["page"] = page
    at.run()
    assert not at.exception, at.exception


def _click(at, label):
    [b for b in at.button if b.label == label][0].click().run()
    assert not at.exception, at.exception


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _app()
    _goto(at, page)
    assert not at.error or page == "Live Analysis"
    assert not at.sidebar.radio            # navigation lives in the top bar now


def test_top_nav_navigates():
    at = _app()
    for label, page in [("About", "About"), ("Analyze", "Live Analysis"), ("Exercises", "Exercise Analysis"),
                        ("Results", "Results"), ("Home", "Home")]:
        at.button(key=f"nav_{label.lower()}").click().run()
        assert not at.exception, at.exception
        assert at.session_state["page"] == page


def test_home_ctas_and_source_switcher():
    at = _app()
    at.button(key="cta_demo").click().run()
    assert at.session_state["page"] == "Demo Mode"
    at.segmented_control(key="source_choice").set_value("Video file").run()
    assert at.session_state["page"] == "Video Upload"


def test_query_param_deep_link():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.query_params["page"] = "About"
    at.run()
    assert at.session_state["page"] == "About"


def test_demo_analysis_then_results():
    at = _app()
    _goto(at, "Demo Mode")
    at.selectbox[0].set_value("squat_bad").run()
    _click(at, "Analyse this clip")
    session = at.session_state["last_session"]
    assert session.source == "demo: squat_bad" and len(session.reps) == 5
    assert session.summary()["average_score"] < 75
    _goto(at, "Results")
    assert "Total reps" in _text(at) and "Form breakdown" in _text(at)
    assert any("Download report (CSV)" == b.label for b in at.get("download_button"))


def test_manual_mode_is_applied_and_kept_across_pages():
    at = _app()
    _goto(at, "Demo Mode")
    at.segmented_control(key="exercise_choice").set_value("pushup").run()
    _goto(at, "Home")                        # settings widgets are not rendered here
    _goto(at, "Demo Mode")
    assert at.segmented_control(key="exercise_choice").value == "pushup"
    at.selectbox[0].set_value("pushup_good").run()
    _click(at, "Analyse this clip")
    session = at.session_state["last_session"]
    assert session.mode == "manual" and session.exercises_done() == ["pushup"]


def test_results_without_session_shows_hint():
    at = _app()
    _goto(at, "Results")
    assert "No analysis yet" in _text(at)


def test_missing_camera_shows_error_not_crash():
    at = _app()
    _goto(at, "Live Analysis")
    at.number_input(key="cam").set_value(5).run()    # no camera at index 5
    _click(at, "Start")
    assert any("unavailable" in e.value for e in at.error)
    assert at.session_state["webcam_running"] is False
