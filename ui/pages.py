"""The Streamlit pages."""
from __future__ import annotations

import html
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from config.thresholds import EXERCISE_THRESHOLDS, OUTPUT_DIR, SCORE_BANDS, SUPPORTED_VIDEO_TYPES, TEMPO
from core.pose_detector import LANDMARK_NAMES, PoseDetector
from core.pipeline import AnalysisPipeline
from core.scoring import status as score_status
from demo.synthetic_athlete import SCENARIO_BY_NAME, SCENARIOS, ensure_samples
from exercises import EXERCISES
from processing.video_processor import VideoSourceError, analyze_video, open_webcam, video_info
from ui import charts, figures
from ui.components import (Settings, css, dip_stage_viewer, idle_panel, kpi_strip, make_pipeline, md,
                           read_frame_at, run_video_analysis, session_overview, settings_bar,
                           status_panel)
from ui.styles import BRAND_MARK, HERO_BG, NAV_ACTIVE
from utils.export import DISCLAIMER, report_csv, report_pdf
from utils.helpers import format_seconds, is_supported_video

PAGES = ["Home", "Live Analysis", "Video Upload", "Demo Mode", "Exercise Analysis",
         "Results", "About"]
ANALYZE = {"Live Analysis": "Camera", "Video Upload": "Video file", "Demo Mode": "Demo clips"}
NAV = [("Home", "Home"), ("Analyze", "Live Analysis"), ("Exercises", "Exercise Analysis"),
       ("Results", "Results"), ("About", "About")]
e = html.escape


def goto(page: str) -> None:
    st.session_state["page"] = page
    st.query_params["page"] = page


def _set(key: str, value) -> None:
    st.session_state[key] = value


# ---------------------------------------------------------------------------
# Frame: navigation and footer
# ---------------------------------------------------------------------------
def top_nav() -> None:
    page = st.session_state["page"]
    active = next(label for label, target in NAV
                  if page == target or (label == "Analyze" and page in ANALYZE))
    css(NAV_ACTIVE.format(slug=active.lower()))
    with st.container(horizontal=True, vertical_alignment="center", key="topnav"):
        md(f"<div class='brand'>{BRAND_MARK}<span class='brand-name'>FORM<i>/</i>AI</span>"
           "<span class='brand-sub'>GYM FORM INTELLIGENCE</span></div>")
        with st.container(horizontal=True, horizontal_alignment="center", key="navlinks"):
            for label, target in NAV:
                st.button(label, key=f"nav_{label.lower()}", type="tertiary", on_click=goto, args=(target,))
        st.button("Start analysis", key="nav_cta", type="primary", on_click=goto, args=("Live Analysis",))


def footer() -> None:
    md("<div class='site-foot'><span class='label'>FORM/AI</span>"
       f"<span class='fine' style='max-width:70ch'>{e(DISCLAIMER)}</span></div>")


def _title(text: str, sub: str = "") -> None:
    md(f"<div class='spacer-m'></div><div class='page-title' role='heading' aria-level='1'>{e(text)}</div>"
       + (f"<p class='lede' style='margin-top:12px'>{e(sub)}</p>" if sub else ""))


# ---------------------------------------------------------------------------
def home(settings: Settings) -> None:
    css(HERO_BG.format(uri=figures.studies_background()))
    with st.container(key="hero"):
        left, right = st.columns([5, 6], vertical_alignment="center", gap="large")
        with left:
            md("<div class='hero-copy'><div class='hero-kicker label'>Pose-based form analysis</div>"
               "<div class='hero-title' role='heading' aria-level='1'>Every rep,<br><em>measured.</em></div>"
               "<p class='lede'>Point a camera at your set. FORM/AI tracks 33 body landmarks, counts "
               "each rep and scores your technique joint by joint.</p></div>")
            with st.container(horizontal=True, key="herocta"):
                st.button("Start analysis", key="cta_start", type="primary", on_click=goto,
                          args=("Live Analysis",))
                st.button("Try a demo clip", key="cta_demo", on_click=goto, args=("Demo Mode",))
        with right:
            md(f"<div class='hero-figure'>{figures.hero_svg()}</div>")

    checks = sum(len(ex.all_checks) for ex in EXERCISES.values())
    md("<div class='stats rise'>"
       f"<div><div class='stat-n'>{len(LANDMARK_NAMES)}</div><div class='stat-t'>body landmarks located in every frame</div></div>"
       f"<div><div class='stat-n'>{len(EXERCISES)}</div><div class='stat-t'>lifts recognised automatically from pose geometry</div></div>"
       f"<div><div class='stat-n'>{checks}</div><div class='stat-t'>form checks, each scored 0 to 100 on every rep</div></div>"
       "<div><div class='stat-n'>1<small>camera</small></div><div class='stat-t'>No wearables, no GPU. A webcam or a video file is enough.</div></div>"
       "</div>")

    cards = []
    for key, ex in EXERCISES.items():
        _, angles, _ = figures.rep_cycle(key)
        cards.append(
            f"<div class='ex-card'><div class='fig'>{figures.exercise_svg(key)}</div>"
            f"<div class='ex-name'>{e(ex.name)}</div><div class='ex-meta'>"
            f"Rep signal <b>{e(ex.rep_feature.replace('_', ' '))}</b><br>"
            f"Range <b>{max(angles):.0f}&#176; to {min(angles):.0f}&#176;</b><br>"
            f"Checks <b>{len(ex.all_checks)}</b></div></div>")
    md("<div class='section-title'>Four lifts, read joint by joint.</div>"
       "<p class='lede'>Each lift has its own rep signal, phase state machine and weighted form "
       f"checks. The figures below replay the scripted good-form rep the demo clips are built from.</p>"
       f"<div class='ex-grid'>{''.join(cards)}</div>")
    st.button("How each lift is scored  →", key="home_guide", type="tertiary", on_click=goto,
              args=("Exercise Analysis",))


# ---------------------------------------------------------------------------
# Analyze: camera / video file / demo clips
# ---------------------------------------------------------------------------
def _source_changed() -> None:
    choice = st.session_state.get("source_choice")
    if choice:
        goto(next(p for p, label in ANALYZE.items() if label == choice))


def _analyze_header(page: str) -> None:
    st.session_state["source_choice"] = ANALYZE[page]
    md("<div class='spacer-m'></div>")
    with st.container(horizontal=True, vertical_alignment="bottom", horizontal_alignment="distribute",
                      key="analyzehead"):
        md("<div class='page-title' role='heading' aria-level='1'>Analyze</div>")
        st.segmented_control("Source", list(ANALYZE.values()), key="source_choice",
                             on_change=_source_changed, label_visibility="collapsed")
    settings_bar(camera=page == "Live Analysis")
    md("<div class='spacer-s'></div>")


def live(settings: Settings) -> None:
    _analyze_header("Live Analysis")
    running = st.session_state.get("webcam_running", False)
    stage_col, side = st.columns([3, 2], gap="large")
    with side:
        with st.container(horizontal=True):
            st.button("Start", type="primary", on_click=_set, args=("webcam_running", True),
                      disabled=running)
            st.button("Stop", on_click=_set, args=("webcam_running", False), disabled=not running)
        panel = st.empty()
    mode = EXERCISES[settings.manual_exercise].name if settings.manual_exercise else "Auto detection"
    if not running:
        stage_col.markdown(
            f"<div class='stage'>{figures.camera_placeholder_svg()}<div class='stage-note'>"
            f"<span>CAMERA {settings.webcam_index} &middot; STANDBY</span><span>{e(mode.upper())}</span></div></div>",
            unsafe_allow_html=True)
        panel.markdown(idle_panel("Ready", [
            "Stand 2-3 m from the camera with your whole body in frame.",
            "Side-on for squats, push-ups and presses.",
            "Press Start. No camera? Use the demo clips."]), unsafe_allow_html=True)
        session = st.session_state.get("last_session")
        if session is not None and session.source.startswith("webcam"):
            md("<div class='spacer-m'></div><div class='section-title'>Last camera session</div>")
            session_overview(session)
            st.button("Open full report", on_click=goto, args=("Results",), key="live_report")
        return

    try:
        cap = open_webcam(settings.webcam_index)
    except VideoSourceError as exc:
        st.session_state["webcam_running"] = False
        st.error(str(exc))
        return
    pipeline = make_pipeline(settings, f"webcam {settings.webcam_index}")
    st.session_state["last_session"] = pipeline.session     # live reference, survives Stop
    st.session_state["last_output"] = None
    frame_slot = stage_col.empty()
    t0 = last = time.perf_counter()
    display_fps, failures = 0.0, 0
    try:
        while time.perf_counter() - t0 < settings.webcam_seconds:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 30:
                    st.error("The camera stopped delivering frames (disconnected or in use).")
                    break
                continue
            failures = 0
            if settings.mirror:
                frame = cv2.flip(frame, 1)
            annotated, stt = pipeline.process(frame, time.perf_counter() - t0)
            now = time.perf_counter()
            display_fps = 0.9 * display_fps + 0.1 / max(now - last, 1e-6) if display_fps else 1 / max(now - last, 1e-6)
            last = now
            frame_slot.image(annotated, channels="BGR", width="stretch")
            panel.markdown(status_panel(stt, display_fps), unsafe_allow_html=True)
    finally:
        cap.release()
        pipeline.detector.close()
    st.session_state["webcam_running"] = False
    st.rerun()


def _analysis_result_block(key: str) -> None:
    session = st.session_state.get("last_session")
    if session is None or session.source != key:
        return
    md("<div class='spacer-m'></div><div class='section-title'>Result</div>")
    session_overview(session)
    out = st.session_state.get("last_output")
    if out and Path(out).exists():
        with st.expander("Annotated output video", expanded=True):
            st.video(out, format="video/webm" if out.endswith(".webm") else "video/mp4")
    st.button("Open full report", on_click=goto, args=("Results",), type="primary", key=f"report_{key}")


def upload(settings: Settings) -> None:
    _analyze_header("Video Upload")
    file = st.file_uploader("Exercise video", type=list(SUPPORTED_VIDEO_TYPES),
                            help="MP4, AVI or MOV. One person, whole body visible works best.")
    if file is None:
        md("<p class='fine'>The video is analysed frame by frame on this machine: pose, joint angles, "
           "reps and form score, with an annotated copy saved to outputs/.</p>")
        return
    if not is_supported_video(file.name):
        st.error(f"Unsupported file type. Please upload one of: {', '.join(SUPPORTED_VIDEO_TYPES)}.")
        return
    path = OUTPUT_DIR / "uploads" / Path(file.name).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file.getbuffer())
    try:
        fps, frames = video_info(path)
    except VideoSourceError as exc:
        st.error(str(exc))
        return
    key = f"upload: {file.name}"
    with st.container(horizontal=True, vertical_alignment="center"):
        md(f"<div class='meta'><b>{e(file.name)}</b> &nbsp; {frames} frames &middot; {fps:.1f} fps "
           f"&middot; {frames / fps:.1f} s</div>")
        start = st.button("Analyse video", type="primary")
    if start:
        run_video_analysis(path, settings, key)
    _analysis_result_block(key)
    md("<div class='spacer-m'></div>")
    with st.expander("DIP pipeline viewer: inspect every processing stage on one frame"):
        t = st.slider("Frame time (s)", 0.0, max(frames / fps - 0.1, 0.0), 0.0, 0.1, key="dip_t_upload")
        frame = read_frame_at(path, t)
        if frame is not None:
            dip_stage_viewer(frame, settings.max_width)


def demo(settings: Settings) -> None:
    _analyze_header("Demo Mode")
    stage_col, side = st.columns([3, 2], gap="large")
    with side:
        name = st.selectbox("Demo clip", [s.name for s in SCENARIOS],
                            format_func=lambda n: f"{n}  -  {SCENARIO_BY_NAME[n].description}")
    sc = SCENARIO_BY_NAME[name]
    with st.spinner("Preparing demo clip (first run renders it)..."):
        path = ensure_samples([name])[name]
    with stage_col:
        with st.container(key="stagecol"):
            st.video(str(path), format="video/webm")
    with side:
        md(f"<div class='panel' style='margin-top:10px'><div class='panel-head'><span class='panel-ex'>"
           f"{e(EXERCISES[sc.exercise].name)}</span><span class='label'>Scripted ground truth</span></div>"
           f"<div class='readouts' style='grid-template-columns:1fr 2fr'><div><span class='label'>Reps</span>"
           f"<div class='ro-v'>{sc.reps}</div></div><div><span class='label'>Scenario</span>"
           f"<div class='ro-v phase' style='color:var(--ink);font-family:var(--sans);letter-spacing:0'>"
           f"{e(sc.description)}</div></div></div>"
           "<p class='fine' style='margin-top:12px'>The clip is synthetic: an articulated figure rendered "
           "from scripted joint angles, then analysed by the same real pipeline. Every count and score "
           "below is measured, not scripted.</p></div>")
        start = st.button("Analyse this clip", type="primary")
    key = f"demo: {name}"
    if start:
        run_video_analysis(path, settings, key)
    _analysis_result_block(key)

    md("<div class='spacer-l'></div><div class='section-title'>Validate on all demo clips</div>"
       "<p class='lede'>Runs the full pipeline with auto detection on every clip and compares the "
       "result with the script.</p><div class='spacer-s'></div>")
    if st.button("Run batch validation"):
        rows, bar = [], st.progress(0.0)
        paths = ensure_samples()
        for i, s in enumerate(SCENARIOS):
            pipe = AnalysisPipeline(PoseDetector(), source=s.name,
                                    enhance=settings.enhance, max_width=settings.max_width)
            try:
                analyze_video(paths[s.name], pipe, save_output=False)
            finally:
                pipe.detector.close()
            summ = pipe.session.summary()
            detected = pipe.session.exercises_done()
            rows.append({"clip": s.name, "expected exercise": EXERCISES[s.exercise].name,
                         "detected exercise": summ["exercises"], "expected reps": s.reps,
                         "counted reps": summ["total_reps"], "average score": summ["average_score"],
                         "correct": detected == [s.exercise] and summ["total_reps"] == s.reps})
            bar.progress((i + 1) / len(SCENARIOS), text=f"{s.name} done")
        st.session_state["batch"] = pd.DataFrame(rows)
    if "batch" in st.session_state:
        df = st.session_state["batch"]
        st.dataframe(df, hide_index=True)
        st.success(f"{int(df['correct'].sum())} / {len(df)} clips: exercise and rep count match the script.")


# ---------------------------------------------------------------------------
def _checks_table(ex) -> str:
    rows = []
    for chk in ex.all_checks:
        ideal, zero = ex.check_range(chk)
        rows.append(f"<tr><td>{e(chk.label)}</td><td class='n'>{e(chk.feature.strip('_').replace('_', ' '))}</td>"
                    f"<td class='n'>{e(chk.reduce)}</td><td class='n'>{ideal}</td><td class='n'>{zero}</td>"
                    f"<td class='n'>{chk.weight:g}</td><td class='n'>{'yes' if chk.live else '-'}</td></tr>")
    return ("<table class='spec'><thead><tr><th>Check</th><th>Feature</th><th>Over rep</th>"
            "<th>Ideal (100)</th><th>Zero (0)</th><th>Weight</th><th>Live</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def exercise_guide(settings: Settings) -> None:
    _title("Exercises", "Each lift is a declarative rule set. Every check reduces one geometric feature "
                        "over a rep (min, max or range) and scores it linearly: 100 at the ideal value, "
                        "0 at the zero value.")
    md("<div class='spacer-s'></div>")
    key = st.segmented_control("Exercise", list(EXERCISES), default="squat", key="guide_ex",
                               format_func=lambda k: EXERCISES[k].name) or "squat"
    ex = EXERCISES[key]
    rep = EXERCISE_THRESHOLDS[ex.key]["rep"]
    start_zone = f">= {rep['extended']}&#176;" if ex.start_extended else f"<= {rep['flexed']}&#176;"
    turn_zone = f"<= {rep['flexed']}&#176;" if ex.start_extended else f">= {rep['extended']}&#176;"
    signals = sorted({c.feature.strip("_").replace("_", " ") for c in ex.all_checks} | {ex.rep_feature.replace("_", " ")})
    phases = "".join(f"<span class='{'turn' if i == 2 else ''}'>{e(p)}</span>" for i, p in enumerate(ex.phase_names))
    left, right = st.columns([6, 5], gap="large")
    with left:
        md(f"<div class='ex-figure rise'>{figures.exercise_svg(key)}</div>"
           f"<div class='ex-figure' style='margin-top:2px;padding-top:18px'>{figures.pose_pair_svg(key)}</div>")
    with right:
        md(f"<div class='section-title' style='font-size:2.6rem;text-transform:uppercase'>{e(ex.name)}</div>"
           f"<p class='lede'>{e(ex.description)}</p>"
           "<div class='facts'>"
           f"<div><span class='label'>Rep signal</span><div class='v'>{e(ex.rep_feature.replace('_', ' '))}</div></div>"
           f"<div><span class='label'>Primary joint</span><div class='v'>{e(figures.rep_vertex(key))}</div></div>"
           f"<div><span class='label'>Start zone</span><div class='v mono'>{start_zone}</div></div>"
           f"<div><span class='label'>Turn zone</span><div class='v mono'>{turn_zone}</div></div>"
           "</div>"
           f"<div class='block'><span class='label'>Phases</span><div class='phases'>{phases}</div></div>"
           f"<div class='block'><span class='label'>Signals measured</span><div class='v' style='color:var(--ink-2);font-size:.92rem'>"
           f"{e(', '.join(signals))}</div></div>")
    md(f"<div class='spacer-m'></div><div class='label'>Form checks</div>{_checks_table(ex)}")

    bands = sorted(SCORE_BANDS)
    cells = []
    colours = {"good": "var(--good)", "warning": "var(--warning)", "bad": "var(--bad)"}
    for i, (minimum, label) in enumerate(bands):
        upper = bands[i + 1][0] if i + 1 < len(bands) else 101
        cells.append(f"<div style='flex:{max(upper - minimum, 4)};background:{colours[score_status(minimum)]}'>"
                     f"{e(label) if upper - minimum >= 10 else ''}</div>")
    md("<div class='spacer-l'></div><div class='section-title'>Score bands</div>"
       "<p class='lede'>A rep's score is the weighted mean of its check scores.</p>"
       f"<div class='scale'>{''.join(cells)}</div><div class='scale-axis'><span>0</span>"
       + "".join(f"<span>{m}</span>" for m, _ in bands[1:]) + "<span>100</span></div>"
       + f"<p class='fine' style='margin-top:16px'>{' &middot; '.join(f'{m}+ {e(l)}' for m, l in reversed(bands))}. "
       f"Tempo check (all exercises): 100 at >= {TEMPO['ideal']} s of movement, 0 at {TEMPO['zero']} s.</p>")


# ---------------------------------------------------------------------------
def _breakdown(session, key: str) -> str:
    df = session.check_breakdown(key)
    rows = []
    for _, r in df.iterrows():
        s = r["average_score"]
        has = s is not None and not pd.isna(s)
        band = score_status(s) if has else "neutral"
        rows.append(f"<li><div class='bar-row'><span>{e(r['check'])}</span><b>{f'{s:.0f}' if has else 'n/a'}</b></div>"
                    f"<div class='bar {band}' style='width:{max(s, 2) if has else 2:.0f}%'></div></li>")
    return f"<ul class='bars'>{''.join(rows)}</ul>"


def results(settings: Settings) -> None:
    session = st.session_state.get("last_session")
    if session is None or not session.frames:
        _title("Results")
        md(f"<div class='stage' style='aspect-ratio:auto;padding:56px 24px;margin-top:24px'>"
           f"<div style='width:180px'>{figures.exercise_svg('squat', label_angle=False)}</div>"
           "<p class='lede' style='text-align:center;margin-top:18px'>No analysis yet. Run a camera "
           "session, upload a video or analyse a demo clip, and the report appears here.</p></div>")
        md("<div class='spacer-s'></div>")
        with st.container(horizontal=True, horizontal_alignment="center"):
            st.button("Demo clips", on_click=goto, args=("Demo Mode",), type="primary")
            st.button("Video file", on_click=goto, args=("Video Upload",))
            st.button("Camera", on_click=goto, args=("Live Analysis",))
        return
    s = session.summary()
    avg = s["average_score"]
    _title("Session report")
    md(f"<div class='meta' style='margin:10px 0 22px'><b>{e(s['source'])}</b> &nbsp; {e(s['mode'])} mode "
       f"&middot; {e(s['date_time'])} &middot; {s['frames_analysed']} frames &middot; person detected in "
       f"{s['pose_detection_rate'] * 100:.0f}%</div>")
    md(kpi_strip([
        ("Total reps", str(s["total_reps"]), "", f"{s['partial_reps']} partial, not counted"),
        ("Average form score", "-" if avg is None else f"{avg:.0f}", "" if avg is None else "/100",
         s["grade"], score_status(avg)),
        ("Best rep", s["best_rep"].split(" ")[0] if s["best_rep"] else "-", "",
         f"score {s['best_rep'].split('(')[1].rstrip(')')}" if s["best_rep"] else "no scored reps"),
        ("Session time", format_seconds(s["duration_s"]), "",
         f"worst rep {s['worst_rep']}" if s["worst_rep"] else s["exercises"]),
    ]))
    if s["pose_detection_rate"] < 0.6 and s["frames_analysed"]:
        st.warning(f"A person was detected in only {s['pose_detection_rate'] * 100:.0f}% of "
                   "frames - results may be incomplete. Check lighting and framing.")

    keys = session.exercises_done() or (
        [session.frames[-1]["exercise"]] if session.frames[-1].get("exercise") else [])
    md("<div class='spacer-m'></div>")
    left, right = st.columns([2, 3], gap="large")
    with left:
        md("<div class='section-title'>Form breakdown</div>")
        for key in session.exercises_done():
            md(f"<span class='label'>{e(EXERCISES[key].name)} &middot; average per check</span>{_breakdown(session, key)}")
        if not session.reps:
            md("<p class='fine'>No repetitions were counted, so there is no per-check breakdown.</p>")
    with right:
        md("<div class='section-title'>Score per rep</div>")
        if session.reps:
            st.altair_chart(charts.ui_score_per_rep(session), theme=None, width="stretch")
        else:
            st.info("No repetitions were counted, so there are no per-rep charts.")
    for key in keys:
        md(f"<div class='spacer-m'></div><div class='section-title'>{e(EXERCISES[key].name)}: "
           f"{e(EXERCISES[key].rep_feature.replace('_', ' '))} over time</div>"
           "<p class='fine'>Shaded bands are counted reps; dashed lines are the rep state machine thresholds.</p>")
        st.altair_chart(charts.ui_angle_vs_time(session, key), theme=None, width="stretch")

    md("<div class='spacer-m'></div>")
    tabs = st.tabs(["Repetitions", "Rep timing", "Form errors", "Joint-angle statistics", "Export"])
    with tabs[0]:
        df = session.reps_df()
        st.dataframe(df, hide_index=True) if not df.empty else st.info("No repetitions counted.")
    with tabs[1]:
        if session.reps:
            st.altair_chart(charts.ui_rep_durations(session), theme=None, width="stretch")
        else:
            st.info("No repetitions counted.")
    with tabs[2]:
        issues = session.issue_counts()
        if issues.empty:
            st.success("No form deviations were detected in the counted repetitions.")
        else:
            st.dataframe(issues, hide_index=True)
        for r in session.reps:
            if r.issues:
                md(f"<div class='block'><span class='label'>{e(EXERCISES[r.exercise].name)} #{r.index} "
                   f"&middot; score {r.score:.0f}</span><ul class='feedback'>"
                   + "".join(f"<li>{e(m)}</li>" for m in r.issues) + "</ul></div>")
    with tabs[3]:
        st.dataframe(session.angle_statistics(), hide_index=True)
    with tabs[4]:
        stamp = session.started_at.strftime("%Y%m%d_%H%M%S")
        with st.container(horizontal=True):
            st.download_button("Download report (CSV)", report_csv(session),
                               f"form_report_{stamp}.csv", "text/csv", type="primary")
            st.download_button("Download repetitions (CSV)", session.reps_df().to_csv(index=False),
                               f"form_reps_{stamp}.csv", "text/csv")
            st.download_button("Download per-frame angles (CSV)", session.frames_df().to_csv(index=False),
                               f"form_frames_{stamp}.csv", "text/csv")
        if st.button("Build PDF report"):
            st.session_state["pdf"] = report_pdf(session)
        if "pdf" in st.session_state:
            st.download_button("Download report (PDF)", st.session_state["pdf"],
                               f"form_report_{stamp}.pdf", "application/pdf")


# ---------------------------------------------------------------------------
DIP_CONCEPTS = [
    ("Geometric transformation", "Resizing with area interpolation (anti-aliased downsampling)."),
    ("Colour-space conversion", "BGR to RGB (model input), BGR to grey (statistics), BGR to LAB (CLAHE)."),
    ("Histogram processing", "Brightness and contrast from grey-level statistics; CLAHE contrast "
                             "enhancement applied only in poor lighting."),
    ("Noise estimation and spatial filtering", "Immerkaer's Laplacian-difference noise estimate decides "
                                               "whether to apply a Gaussian low-pass filter."),
    ("Edge detection", "Variance of the Laplacian as a sharpness and motion-blur measure."),
    ("Segmentation and morphology", "Person probability map to binary mask (thresholding), morphological "
                                    "opening and closing, optional background suppression."),
    ("Region of interest", "Subject bounding box from landmarks; lighting and sharpness are measured on "
                           "the person, not the background."),
    ("Temporal filtering", "Exponential moving average on features and a median filter per rep."),
]
PIPELINE = ["Acquisition", "Preprocessing", "Pose detection (BlazePose, 33 landmarks)", "ROI, mask, sharpness",
            "Feature extraction", "Exercise recognition", "Rep state machine", "Form analysis", "Scoring",
            "Visualisation", "Report"]


def about(settings: Settings) -> None:
    _title("About")
    md("<div class='spacer-s'></div>")
    left, right = st.columns([6, 5], gap="large")
    with left:
        md("<p class='statement'>Form feedback from an ordinary camera.</p>")
    with right:
        md("<p class='lede'><b style='color:var(--ink)'>Problem.</b> People training alone rarely get "
           "feedback on their exercise technique. This project measures exercise form from ordinary video "
           "using Digital Image Processing and computer vision: no sensors, no GPU.</p>"
           "<p class='lede' style='margin-top:14px'><b style='color:var(--ink)'>Objectives.</b> Detect the "
           "body and its landmarks, compute joint angles, recognise the exercise, count repetitions "
           "robustly, detect specific form deviations, score them transparently and give real-time visual "
           "feedback with an exportable report.</p>")
    flow = "<i></i>".join(f"<span>{e(p)}</span>" for p in PIPELINE)
    md(f"<div class='spacer-l'></div><div class='section-title'>Pipeline</div><div class='flow'>{flow}</div>")
    concepts = "".join(f"<div><h4>{e(t)}</h4><p>{e(d)}</p></div>" for t, d in DIP_CONCEPTS)
    md(f"<div class='spacer-l'></div><div class='section-title'>Digital Image Processing concepts used</div>"
       f"<div class='dip' style='margin-top:14px'>{concepts}</div>")
    md("<div class='spacer-l'></div><div class='cols3'>"
       "<div><span class='label'>Technologies</span><p>Python, OpenCV, MediaPipe Tasks (Pose Landmarker "
       "lite), NumPy, pandas, Matplotlib, Altair, Streamlit, pytest.</p></div>"
       "<div><span class='label'>Limitations</span><p>2-D image-plane angles depend on camera viewpoint "
       "(side view recommended); one subject is analysed at a time; landmark accuracy drops with "
       "occlusion, baggy clothing, unusual poses and very poor lighting; recognition is rule-based over "
       "four exercises; thresholds are sensible defaults, not clinically validated.</p></div>"
       "<div><span class='label'>Future improvements</span><p>3-D world landmarks for viewpoint "
       "invariance, learned (sequence-model) exercise recognition, per-user calibration of thresholds, "
       "more exercises, audio cues, and mobile deployment.</p></div></div>"
       "<p class='fine' style='margin-top:32px'>See README.md and PROJECT_DOCUMENTATION.md for the full "
       "methodology, maths and references.</p>")
