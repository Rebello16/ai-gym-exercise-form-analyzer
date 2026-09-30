"""Reusable Streamlit building blocks."""
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from config.thresholds import PROCESSING
from core.pipeline import AnalysisPipeline, FrameStatus
from core.pose_detector import PoseDetector
from core.scoring import grade, status as score_status
from core.session import SessionLog
from exercises import EXERCISES
from processing import image_processing as ip
from processing import preprocessing as pp
from processing.video_processor import VideoSourceError, analyze_video
from processing.visualization import draw_skeleton
from utils.helpers import format_seconds

AUTO = "auto"
TAG_WORD = {"good": "PASS", "warning": "WATCH", "bad": "FIX", "neutral": "N/A"}
STATUS_WORD = {"good": "GOOD FORM", "warning": "CHECK FORM", "bad": "CORRECT YOUR FORM",
               "neutral": "WAITING"}

# Session-state keys of the settings widgets and their defaults. The widgets
# only render on the analysis pages; read_settings() keeps their values alive
# on every other page.
SETTING_DEFAULTS = {
    "exercise_choice": AUTO, "max_width": PROCESSING["max_width"],
    "frame_skip": PROCESSING["frame_skip"], "enhance": True, "focus": False,
    "cam": PROCESSING["webcam_index"], "cam_secs": PROCESSING["webcam_max_seconds"], "mirror": True,
}


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def css(markup: str) -> None:
    """Style-only HTML: st.html adds it to the page without taking layout space."""
    st.html(markup)


def css(markup: str) -> None:
    """Style-only HTML: st.html adds it to the page without taking layout space."""
    st.html(markup)


@dataclass
class Settings:
    manual_exercise: str | None
    max_width: int
    frame_skip: int
    enhance: bool
    background_focus: bool
    webcam_index: int
    webcam_seconds: int
    mirror: bool


def read_settings() -> Settings:
    ss = st.session_state
    for key, default in SETTING_DEFAULTS.items():
        ss[key] = ss.get(key, default)   # re-assign so Streamlit keeps it while the widget is hidden
    choice = ss["exercise_choice"] or AUTO
    return Settings(None if choice == AUTO else choice, ss["max_width"], ss["frame_skip"],
                    ss["enhance"], ss["focus"], int(ss["cam"]), ss["cam_secs"], ss["mirror"])


def _keep_a_choice() -> None:
    if st.session_state["exercise_choice"] is None:     # clicking the active segment clears it
        st.session_state["exercise_choice"] = AUTO


def settings_bar(camera: bool) -> None:
    """Exercise selection (auto detection or a forced exercise) and processing options."""
    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium", key="setup"):
        st.segmented_control(
            "Exercise", [AUTO, *EXERCISES], key="exercise_choice", on_change=_keep_a_choice,
            format_func=lambda k: "Auto detect" if k == AUTO else EXERCISES[k].name,
            help="Auto detection classifies the exercise from pose geometry. Pick an "
                 "exercise to force it (use it if auto is unsure).")
        with st.popover("Processing", icon=":material/tune:"):
            st.select_slider("Processing width (px)", [320, 480, 640, 960], key="max_width",
                             help="Frames are downscaled to this width. Smaller = faster.")
            if not camera:
                st.slider("Analyse every n-th video frame", 1, 4, key="frame_skip")
            st.checkbox("Adaptive enhancement (CLAHE + denoise)", key="enhance",
                        help="Applied only when poor lighting / noise is measured.")
            st.checkbox("Background focus (segmentation mask)", key="focus",
                        help="Blurs everything outside the detected person. Slightly slower.")
            if camera:
                st.number_input("Camera index", 0, 5, key="cam")
                st.slider("Max session length (s)", 15, 600, key="cam_secs")
                st.checkbox("Mirror view", key="mirror")


def make_pipeline(settings: Settings, source: str) -> AnalysisPipeline:
    """Fresh pipeline + detector for one session (clean tracking state)."""
    detector = PoseDetector(with_segmentation=settings.background_focus)
    return AnalysisPipeline(detector, source=source, manual_exercise=settings.manual_exercise,
                            enhance=settings.enhance, background_focus=settings.background_focus,
                            max_width=settings.max_width)


def _check_rows(items: list[tuple[str, float | None]]) -> str:
    rows = []
    for label, score in items:
        band = score_status(score)
        rows.append(f"<li><span class='tag {band}'>{TAG_WORD[band]}</span><span>{html.escape(label)}</span>"
                    f"<b>{'n/a' if score is None else round(score)}</b></li>")
    return f"<ul class='checks'>{''.join(rows)}</ul>"


def status_panel(stt: FrameStatus, display_fps: float | None = None) -> str:
    """HTML read-out of the live analysis state (rep, phase, score, checks, feedback)."""
    e = html.escape
    name = EXERCISES[stt.exercise].name if stt.exercise else "Detecting"
    if stt.mode == "manual":
        source = "MANUAL SELECTION"
    else:
        conf = stt.recognition.confidence * 100 if stt.recognition and stt.exercise else 0
        source = f"AUTO DETECTION {conf:.0f}%" if stt.exercise else "AUTO DETECTION"
    score = stt.last_rep.score if stt.last_rep else None
    score_html = "-" if score is None else f"{score:.0f}<small>/100</small>"
    readouts = (f"<div class='readouts'><div><span class='label'>Reps</span><div class='ro-v'>{stt.reps}</div></div>"
                f"<div><span class='label'>Phase</span><div class='ro-v phase'>{e(stt.phase)}</div></div>"
                f"<div><span class='label'>Last rep</span><div class='ro-v'>{score_html}</div></div></div>")
    blocks = []
    if stt.live_checks:
        blocks.append("<div class='block'><span class='label'>Posture, this frame</span>"
                      + _check_rows([(lc.check.label, lc.score) for lc in stt.live_checks]) + "</div>")
    if stt.last_rep is not None:
        ex = EXERCISES[stt.last_rep.exercise]
        blocks.append(f"<div class='block'><span class='label'>Last rep &middot; {e(grade(score))}</span>"
                      + _check_rows([(c.label, stt.last_rep.scores.get(c.key)) for c in ex.all_checks])
                      + "</div>")
    if stt.feedback:
        fb = f"<ul class='feedback'>{''.join(f'<li>{e(m)}</li>' for m in stt.feedback)}</ul>"
    else:
        quiet = "No deviations detected." if stt.last_rep else "Complete a rep for feedback."
        fb = f"<ul class='feedback quiet'><li>{quiet}</li></ul>"
    blocks.append(f"<div class='block'><span class='label'>Feedback</span>{fb}</div>")
    if stt.warnings:
        blocks.append("<div class='block'><span class='label'>Warnings</span><ul class='feedback warnings'>"
                      + "".join(f"<li>{e(m)}</li>" for m in stt.warnings) + "</ul></div>")
    tele = [("Processing", f"{stt.fps:.1f} fps")]
    if display_fps is not None:
        tele.append(("Display", f"{display_fps:.1f} fps"))
    tele.append(("People", stt.num_people))
    telemetry = "".join(f"<span>{k} <b>{e(str(v))}</b></span>" for k, v in tele)
    return (f"<div class='panel {stt.status}'><div class='panel-head'><span class='panel-ex'>{e(name)}</span>"
            f"<span class='tag {stt.status}' style='padding:4px 8px'>{STATUS_WORD[stt.status]}</span></div>"
            f"<div class='meta' style='margin-top:6px'>{e(source)}</div>{readouts}{''.join(blocks)}"
            f"<div class='telemetry'>{telemetry}</div></div>")


def idle_panel(title: str, lines: list[str]) -> str:
    """Status panel before anything runs: same frame as the live read-out."""
    items = "".join(f"<li>{html.escape(t)}</li>" for t in lines)
    return (f"<div class='panel'><div class='panel-head'><span class='panel-ex'>{html.escape(title)}</span>"
            f"</div><div class='readouts'><div><span class='label'>Reps</span><div class='ro-v'>0</div></div>"
            f"<div><span class='label'>Phase</span><div class='ro-v phase'>STANDBY</div></div>"
            f"<div><span class='label'>Last rep</span><div class='ro-v'>-</div></div></div>"
            f"<div class='block'><span class='label'>Setup</span><ul class='feedback quiet'>{items}</ul></div></div>")


def run_video_analysis(path: Path, settings: Settings, label: str) -> SessionLog | None:
    """Analyse a video file with a live preview; stores the session for the Results page."""
    col_video, col_status = st.columns([3, 2], gap="large")
    preview, panel = col_video.empty(), col_status.empty()
    progress = st.progress(0.0, text="Analysing frames...")
    counter = {"n": 0}

    def on_frame(annotated: np.ndarray, stt: FrameStatus, frac: float) -> None:
        counter["n"] += 1
        if counter["n"] % 3 == 0 or frac >= 1.0:      # throttle UI updates
            preview.image(annotated, channels="BGR", width="stretch")
            panel.markdown(status_panel(stt), unsafe_allow_html=True)
            progress.progress(min(frac, 1.0), text=f"Analysing frames... {frac * 100:.0f}%")

    try:
        pipeline = make_pipeline(settings, label)
        output = analyze_video(path, pipeline, settings.frame_skip, on_frame=on_frame)
    except (VideoSourceError, RuntimeError) as exc:
        progress.empty()
        st.error(str(exc))
        return None
    finally:
        if "pipeline" in locals():
            pipeline.detector.close()
    progress.progress(1.0, text="Analysis complete")
    st.session_state["last_session"] = pipeline.session
    st.session_state["last_output"] = str(output) if output else None
    return pipeline.session


def kpi_strip(items: list[tuple[str, str, str, str]]) -> str:
    """(label, value, unit, sub-line[, band]) cells in one ruled strip."""
    cells = []
    for label, value, unit, sub, *band in items:
        u = f"<small>{html.escape(unit)}</small>" if unit else ""
        cells.append(f"<div><span class='label'>{html.escape(label)}</span><div class='kpi-v'>"
                     f"{html.escape(value)}{u}</div><div class='kpi-s {band[0] if band else ''}'>"
                     f"{html.escape(sub)}</div></div>")
    return f"<div class='kpis rise'>{''.join(cells)}</div>"


def session_overview(session: SessionLog) -> None:
    """Headline numbers for a finished session."""
    s = session.summary()
    avg = s["average_score"]
    md(kpi_strip([
        ("Exercise", s["exercises"], "", f"{s['mode']} mode"),
        ("Repetitions", str(s["total_reps"]), "", f"{s['partial_reps']} partial, not counted"),
        ("Average form score", "-" if avg is None else f"{avg:.0f}", "" if avg is None else "/100",
         s["grade"], score_status(avg)),
        ("Duration", format_seconds(s["duration_s"]), "", f"{s['frames_analysed']} frames analysed"),
    ]))
    if s["pose_detection_rate"] < 0.6 and s["frames_analysed"]:
        st.warning(f"A person was detected in only {s['pose_detection_rate'] * 100:.0f}% of "
                   "frames - results may be incomplete. Check lighting and framing.")


def read_frame_at(path: Path, seconds: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _histogram(grey_before: np.ndarray, grey_after: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 2.2), dpi=110)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for img, label, color in ((grey_before, "input", "#85837D"), (grey_after, "after CLAHE", "#F4513C")):
        hist = cv2.calcHist([img], [0], None, [64], [0, 256]).ravel()
        ax.plot(np.linspace(0, 255, 64), hist / hist.sum(), color=color, linewidth=2, label=label)
    ax.set_xlabel("grey level", fontsize=8, color="#B4B2AC")
    ax.set_ylabel("fraction of pixels", fontsize=8, color="#B4B2AC")
    ax.tick_params(labelsize=7, colors="#85837D")
    legend = ax.legend(fontsize=7, frameon=False)
    for text in legend.get_texts():
        text.set_color("#B4B2AC")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#34343A")
    fig.tight_layout()
    return fig


def dip_stage_viewer(frame_bgr: np.ndarray, max_width: int) -> None:
    """Show every DIP stage applied (or measured) on one frame."""
    resized, _ = pp.resize_frame(frame_bgr, max_width)
    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    brightness, contrast = pp.grey_statistics(grey)
    sigma = pp.estimate_noise_sigma(grey)
    clahe = pp.apply_clahe(resized)
    blurred = cv2.GaussianBlur(resized, pp.PREPROCESSING["gaussian_kernel"], 0)
    edges = ip.edge_map_for_display(resized)

    detector = PoseDetector(with_segmentation=True)
    try:
        pose = detector.detect(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), 0)
    finally:
        detector.close()
    mask = ip.binary_person_mask(pose.mask) if pose.mask is not None else np.zeros_like(grey)
    skeleton = resized.copy()
    roi = None
    if pose.detected:
        draw_skeleton(skeleton, pose.points, pose.visibility, (80, 200, 80))
        roi = ip.landmarks_roi(pose.points, pose.visibility, resized.shape)
        if roi:
            cv2.rectangle(skeleton, roi[:2], roi[2:], (0, 200, 255), 2)
    sharp = ip.sharpness(resized, roi)

    row1 = st.columns(3)
    row1[0].image(resized, channels="BGR", caption="1. Resized input (INTER_AREA)")
    row1[1].image(grey, caption="2. Greyscale (BGR -> Y)")
    row1[2].image(clahe, channels="BGR", caption="3. CLAHE on L channel (LAB)")
    row2 = st.columns(3)
    row2[0].image(blurred, channels="BGR", caption="4. Gaussian blur 5x5 (denoising)")
    row2[1].image(edges, caption="5. |Laplacian| edge map (min-max normalised)")
    row2[2].image(mask, caption="6. Binary person mask (threshold + open/close)")
    row3 = st.columns([1, 1])
    row3[0].image(skeleton, channels="BGR", caption="7. Landmarks, skeleton and subject ROI")
    row3[1].pyplot(_histogram(grey, cv2.cvtColor(clahe, cv2.COLOR_BGR2GRAY)))
    th = pp.PREPROCESSING
    st.markdown(
        f"""
| Measurement | Value | Threshold | Decision in the pipeline |
|---|---|---|---|
| Mean brightness | {brightness:.1f} | < {th['dark_mean_threshold']} = dark | {'CLAHE applied' if brightness < th['dark_mean_threshold'] else 'OK'} |
| Contrast (grey std-dev) | {contrast:.1f} | < {th['low_contrast_std_threshold']} = flat | {'CLAHE applied' if contrast < th['low_contrast_std_threshold'] else 'OK'} |
| Noise sigma (Immerkaer) | {sigma:.2f} | > {th['noise_sigma_threshold']} = noisy | {'Gaussian denoise applied' if sigma > th['noise_sigma_threshold'] else 'OK'} |
| Sharpness (var. of Laplacian, ROI) | {sharp:.1f} | < {th['blur_laplacian_threshold']} = blurry | {'blur warning' if sharp < th['blur_laplacian_threshold'] else 'OK'} |
| People detected | {pose.num_people} | > 1 = warn | {'most prominent person analysed' if pose.num_people > 1 else 'OK'} |
""")
