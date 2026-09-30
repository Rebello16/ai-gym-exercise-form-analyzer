"""Stage 9 - drawing the analysis on the frame with OpenCV."""
from __future__ import annotations

import cv2
import numpy as np

from config.thresholds import COLORS, POSE
from core.feature_extractor import FEATURE_VERTEX
from core.pose_detector import LM, SKELETON
from core.scoring import grade, status as score_status
from exercises import EXERCISES

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, text, org, scale, color, thickness=1):
    """Text with a dark outline so it stays readable on any background.

    The outline is made of 1-px shifted copies (not a thicker stroke) because
    stroke width changes glyph width with OpenCV 5's font renderer.
    """
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, text, (x + dx, y + dy), _FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(img: np.ndarray, points: np.ndarray, visibility: np.ndarray,
                  color: tuple[int, int, int]) -> None:
    thr = POSE["landmark_visibility_threshold"]
    lw = max(2, img.shape[1] // 250)
    for a, b in SKELETON:
        ia, ib = LM[a], LM[b]
        if visibility[ia] >= thr and visibility[ib] >= thr:
            cv2.line(img, tuple(map(int, points[ia])), tuple(map(int, points[ib])), color, lw, cv2.LINE_AA)
    for name in LM:
        i = LM[name]
        if i > 10 and visibility[i] >= thr:          # skip face points
            cv2.circle(img, tuple(map(int, points[i])), lw + 2, COLORS["joint"], -1, cv2.LINE_AA)
            cv2.circle(img, tuple(map(int, points[i])), lw + 2, color, 1, cv2.LINE_AA)


def draw_angles(img, points, features, exercise_key: str, scale: float) -> None:
    for feature in EXERCISES[exercise_key].display_features:
        value = features.get(feature)
        if value is None:
            continue
        x, y = points[LM[f"{features.side}_{FEATURE_VERTEX[feature]}"]]
        _text(img, f"{value:.0f} deg", (int(x) + 10, int(y) - 10), scale * 0.9, COLORS["neutral"])


def draw_overlay(img: np.ndarray, st) -> np.ndarray:
    """Draw skeleton, angles, info panel, feedback and warnings for a FrameStatus."""
    h, w = img.shape[:2]
    scale = max(0.4, w / 1100)
    line_h = int(28 * scale / 0.6)
    color = COLORS[st.status]

    if st.pose_detected:
        draw_skeleton(img, st.pose.points, st.pose.visibility, color)
        if st.exercise:
            draw_angles(img, st.pose.points, st.features, st.exercise, scale)

    # --- info panel ----------------------------------------------------
    name = EXERCISES[st.exercise].name if st.exercise else "Detecting..."
    mode = "MANUAL" if st.mode == "manual" else "AUTO"
    if st.mode == "auto" and st.recognition and st.exercise:
        mode += f" {st.recognition.confidence * 100:.0f}%"
    lines = [(f"{name}  [{mode}]", COLORS["neutral"]),
             (f"Reps: {st.reps}   Phase: {st.phase}", COLORS["neutral"])]
    if st.last_rep is not None and st.last_rep.score is not None:
        s = st.last_rep.score
        lines.append((f"Last rep score: {s:.0f} ({grade(s)})", COLORS[score_status(s)]))
    lines.append((f"FPS: {st.fps:.1f}", COLORS["neutral"]))
    panel_w = int(w * 0.62)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, line_h * len(lines) + 12), COLORS["panel"], -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (text, c) in enumerate(lines):
        _text(img, text, (10, line_h * (i + 1)), scale, c, 1)

    # --- feedback / warnings (bottom) ----------------------------------
    messages = [(m, COLORS["warning"]) for m in st.warnings[:2]]
    if st.feedback:
        messages += [(m, color) for m in st.feedback[:3]]
    elif st.last_rep is not None:
        messages.append(("GOOD FORM", COLORS["good"]))
    small = scale * 0.85
    for i, (text, c) in enumerate(reversed(messages)):
        _text(img, text[:80], (10, h - 12 - i * int(line_h * 0.9)), small, c, 1)

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 4)   # status border
    return img
