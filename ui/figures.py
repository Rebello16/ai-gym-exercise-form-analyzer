"""Skeleton figures for the UI, built from the project's own pose data.

Nothing here is hand-drawn. Every figure is a pose from the scripted good-form
rep of an exercise (the forward-kinematics athlete in demo/synthetic_athlete.py
that also renders the demo clips), drawn with the landmark topology the live
overlay uses (core.pose_detector.SKELETON). Angles are measured with
core.angle_calculator, phases come from the real rep state machine
(core.pipeline.ExerciseTracker) and the subject box from
processing.image_processing.landmarks_roi.

Output is SVG; motion is SMIL ``<animate>`` over the sampled rep, so the
figures move exactly as the scripted athlete does.
"""
from __future__ import annotations

import base64
import itertools
from functools import lru_cache

import numpy as np

from core.angle_calculator import calculate_angle
from core.feature_extractor import FEATURE_VERTEX
from core.pipeline import ExerciseTracker
from core.pose_detector import LANDMARK_NAMES, LM, SKELETON
from demo.synthetic_athlete import FLOOR, H, POSES, SCENARIO_BY_NAME, W
from exercises import EXERCISES
from processing.image_processing import landmarks_roi

INK, ACCENT, MUTED, BG = "#EDECE8", "#F4513C", "#8A8882", "#0C0C0D"
FAR = np.array([-9.0, -5.0])          # far-side limbs sit slightly behind, as in the demo renderer
HEAD_R = 17
PARTS = {"shoulder": "shoulder", "elbow": "elbow", "wrist": "wrist", "hip": "hip",
         "knee": "knee", "ankle": "ankle", "heel": "heel", "toe": "foot_index"}
ARMS = {"knee": ("hip", "ankle"), "elbow": ("shoulder", "wrist"), "hip": ("shoulder", "knee")}
# body landmarks the scripted athlete defines (the overlay skips face points too; no hands)
BODY = [n for n in LANDMARK_NAMES if LM[n] > 10 and n.split("_", 1)[1] in PARTS.values()]


# ---------------------------------------------------------------------------
# Data: sampled reps
# ---------------------------------------------------------------------------
def _landmarks(joints: dict) -> dict:
    lm = {"nose": np.asarray(joints["head"], float)}
    for part, name in PARTS.items():
        lm[f"right_{name}"] = np.asarray(joints[part], float)
        lm[f"left_{name}"] = lm[f"right_{name}"] + FAR
    return lm


def joint_angle(lm: dict, vertex: str) -> float:
    a, c = ARMS[vertex]
    return calculate_angle(lm[f"right_{a}"], lm[f"right_{vertex}"], lm[f"right_{c}"])


def rep_vertex(key: str) -> str:
    return FEATURE_VERTEX[EXERCISES[key].rep_feature]


@lru_cache
def rep_cycle(key: str, n: int = 24) -> tuple[list[dict], list[float], list[str]]:
    """One scripted good-form rep (incl. the pause): landmarks, rep angle, phase per sample."""
    sc = SCENARIO_BY_NAME[f"{key}_good"]
    cycle = sc.rep_seconds + sc.pause_seconds
    ts = [cycle * i / n for i in range(n + 1)]
    frames = [_landmarks(POSES[key](sc.progress(sc.lead_seconds + t), sc.params)) for t in ts]
    angles = [joint_angle(f, rep_vertex(key)) for f in frames]
    tracker = ExerciseTracker(EXERCISES[key])     # run the real state machine for two cycles
    for t, a in zip(ts, angles):
        tracker.counter.update(a, t)
    phases = []
    for t, a in zip(ts, angles):
        tracker.counter.update(a, cycle + t)
        phases.append(tracker.phase)
    return frames, angles, phases


def turn_index(key: str) -> int:
    _, angles, _ = rep_cycle(key)
    return int(np.argmin(angles) if EXERCISES[key].start_extended else np.argmax(angles))


def _bbox(frames: list[dict]) -> tuple[float, float, float, float]:
    pts = np.array([p for f in frames for p in f.values()])
    (x1, y1), (x2, y2) = pts.min(axis=0), pts.max(axis=0)
    return x1 - HEAD_R, y1 - HEAD_R, x2 + HEAD_R, max(y2, FLOOR - 4)


def _roi(lm: dict) -> tuple[int, int, int, int]:
    points, vis = np.zeros((len(LANDMARK_NAMES), 2)), np.zeros(len(LANDMARK_NAMES))
    for name, p in lm.items():
        points[LM[name]], vis[LM[name]] = p, 1.0
    return landmarks_roi(points, vis, (H, W))


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------
def _p(v) -> str:
    return f"{v[0]:.0f} {v[1]:.0f}"


def _bones(lm: dict, near: bool) -> str:
    return "".join(f"M{_p(lm[a])}L{_p(lm[b])}" for a, b in SKELETON
                   if (a.startswith("right") and b.startswith("right")) == near)


def _joints(lm: dict, near: bool, only: tuple[str, ...] = ()) -> str:
    names = only or [n for n in BODY if n.startswith("right") == near]
    return "".join(f"M{_p(lm[n])}h0" for n in names)


def _head(lm: dict) -> str:
    x, y = lm["nose"]
    return f"M{x - HEAD_R:.1f} {y:.1f}a{HEAD_R} {HEAD_R} 0 1 0 {2 * HEAD_R} 0a{HEAD_R} {HEAD_R} 0 1 0 {-2 * HEAD_R} 0"


def _arc(lm: dict, vertex: str, r: float = 30) -> str:
    a, c = ARMS[vertex]
    v = lm[f"right_{vertex}"]
    ua, uc = (lm[f"right_{n}"] - v for n in (a, c))
    ua, uc = ua / np.linalg.norm(ua), uc / np.linalg.norm(uc)
    sweep = 1 if ua[0] * uc[1] - ua[1] * uc[0] > 0 else 0
    return f"M{_p(v + ua * r)}A{r} {r} 0 0 {sweep} {_p(v + uc * r)}"


def _brackets(box, k: float = 16) -> str:
    x1, y1, x2, y2 = box
    return (f"M{x1} {y1 + k}V{y1}H{x1 + k}M{x2 - k} {y1}H{x2}V{y1 + k}"
            f"M{x2} {y2 - k}V{y2}H{x2 - k}M{x1 + k} {y2}H{x1}V{y2 - k}")


def _path(ds: list[str], dur: float, **attrs) -> str:
    """<path> whose ``d`` is animated through ``ds`` (static if only one shape)."""
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    anim = (f'<animate attributeName="d" values="{";".join(ds)}" dur="{dur}s" '
            f'repeatCount="indefinite"/>' if len(set(ds)) > 1 else "")
    return f'<path d="{ds[0]}" {a} fill="none">{anim}</path>'


def _discrete(values: list[str], dur: float, x: float, y: float, **attrs) -> str:
    """Text that switches value in step with the sampled frames."""
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    runs, n = [], len(values) - 1               # last sample == first (closed loop)
    for i, v in enumerate(values[:-1]):
        if runs and runs[-1][0] == v:
            runs[-1][2] = i + 1
        else:
            runs.append([v, i, i + 1])
    if len(runs) == 1:
        return f'<text x="{x}" y="{y}" {a}>{values[0]}</text>'
    out = []
    for v, i, j in runs:
        vals, times = ("1;0", f"0;{j / n:.4f}") if i == 0 else (
            ("0;1", f"0;{i / n:.4f}") if j == n else ("0;1;0", f"0;{i / n:.4f};{j / n:.4f}"))
        out.append(f'<text x="{x}" y="{y}" {a} opacity="{1 if i == 0 else 0}">{v}'
                   f'<animate attributeName="opacity" values="{vals}" keyTimes="{times}" '
                   f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/></text>')
    return "".join(out)


_ids = itertools.count()


def skeleton(frames: list[dict], dur: float = 0, ink: str = INK, accent: str = ACCENT,
             width: float = 3.2, alpha: float = 1.0, focus: str | None = None,
             joints: bool = True, head: bool = True) -> str:
    """A pose skeleton in the overlay's topology; animated when given several frames.

    Joints are the overlay's hollow nodes: one shared (animated) geometry drawn
    twice through <use>, once as the accent ring and once as the dark centre.
    """
    def d(fn, *args):
        return [fn(f, *args) for f in frames]
    w, jid = width, f"j{next(_ids)}"
    parts = [
        _path(d(_bones, False), dur, stroke=ink, stroke_width=w * 0.8, stroke_opacity=0.3,
              stroke_linecap="round"),
        _path(d(_bones, True), dur, stroke=ink, stroke_width=w, stroke_linecap="round",
              stroke_linejoin="round"),
    ]
    if head:
        parts.append(_path(d(_head), dur, stroke=ink, stroke_width=w * 0.8, stroke_opacity=0.8))
    if joints:
        parts += [
            f'<defs>{_path(d(_joints, True), dur, id=jid)}</defs>',
            f'<use href="#{jid}" stroke="{accent}" stroke-width="{w * 3:.1f}" stroke-linecap="round"/>',
            f'<use href="#{jid}" stroke="{BG}" stroke-width="{w * 1.4:.1f}" stroke-linecap="round"/>',
        ]
    if focus:     # the joint that drives the rep counter, filled
        parts.append(_path(d(_joints, True, (f"right_{focus}",)), dur, stroke=accent,
                           stroke_width=w * 4.2, stroke_linecap="round"))
    return f'<g opacity="{alpha}">{"".join(parts)}</g>'


def _svg(view: tuple[float, float, float, float], body: str, cls: str = "", label: str = "") -> str:
    x, y, w, h = view
    return (f'<svg class="{cls}" viewBox="{x:.0f} {y:.0f} {w:.0f} {h:.0f}" role="img" '
            f'aria-label="{label}" xmlns="http://www.w3.org/2000/svg" '
            f'preserveAspectRatio="xMidYMax meet">{body}</svg>')


def _motion(animated: str, still: str) -> str:
    """Animated group plus a still one shown under prefers-reduced-motion."""
    return f'<g class="fx-motion">{animated}</g><g class="fx-still">{still}</g>'


MONO = "font-family=\"IBM Plex Mono, monospace\""


# ---------------------------------------------------------------------------
# Compositions
# ---------------------------------------------------------------------------
def hero_svg() -> str:
    """Squat rep with ghosted descent, joint trajectories, ROI and live read-outs."""
    key, dur = "squat", 5.2
    frames, angles, phases = rep_cycle(key)
    bottom = turn_index(key)
    x1, y1, x2, y2 = _bbox(frames)
    view = (x1 - 70, y1 - 46, (x2 - x1) + 330, (y2 - y1) + 110)
    hips = " ".join(_p(f["right_hip"]) for f in frames[:bottom + 1])
    knees = " ".join(_p(f["right_knee"]) for f in frames[:bottom + 1])
    ghosts = "".join(skeleton([frames[i]], alpha=0.05 + 0.035 * k, width=2.2, joints=False, head=False)
                     for k, i in enumerate(range(0, bottom, max(bottom // 4, 1))))
    rois = [_roi(f) for f in frames]
    hip_angles = [f"{joint_angle(f, 'hip'):.0f}&#176;" for f in frames]
    rx = x2 + 70
    readout = (
        f'<g {MONO} fill="{MUTED}" font-size="10" letter-spacing="1.4">'
        f'<text x="{rx}" y="{y1 + 20}">KNEE ANGLE</text>'
        f'<text x="{rx}" y="{y1 + 112}">HIP ANGLE</text>'
        f'<text x="{rx}" y="{y1 + 184}">PHASE</text></g>'
        + _discrete([f"{a:.0f}&#176;" for a in angles], dur, rx - 3, y1 + 80, fill=INK,
                    font_size="60", font_weight="700", letter_spacing="-2",
                    font_family="Archivo, sans-serif")
        + _discrete(hip_angles, dur, rx, y1 + 146, fill=INK, font_size="26",
                    font_weight="600", font_family="Archivo, sans-serif")
        + _discrete(phases, dur, rx, y1 + 210, fill=ACCENT, font_size="14", letter_spacing="1.6",
                    font_family="IBM Plex Mono, monospace", font_weight="500"))
    floor = (f'<path d="M{x1 - 60} {FLOOR - 4}H{rx + 190}" stroke="{INK}" stroke-opacity=".14"/>'
             + "".join(f'<path d="M{x:.0f} {FLOOR - 4}v6" stroke="{INK}" stroke-opacity=".25"/>'
                       f'<text x="{x:.0f}" y="{FLOOR + 16}" {MONO} font-size="8.5" fill="{MUTED}" '
                       f'text-anchor="middle">{x / W:.2f}</text>'
                       for x in (W * k / 20 for k in range(int(x1 / W * 20), int(x2 / W * 20) + 2))))
    trails = (f'<polyline points="{hips}" fill="none" stroke="{ACCENT}" stroke-opacity=".55" '
              f'stroke-width="1.2" stroke-dasharray="2 4"/>'
              f'<polyline points="{knees}" fill="none" stroke="{INK}" stroke-opacity=".3" '
              f'stroke-width="1.2" stroke-dasharray="2 4"/>')
    roi_label = (f'<text {MONO} font-size="9" fill="{MUTED}" letter-spacing="1.2">SUBJECT ROI'
                 f'<animate attributeName="x" values="{";".join(str(r[0]) for r in rois)}" '
                 f'dur="{dur}s" repeatCount="indefinite"/>'
                 f'<animate attributeName="y" values="{";".join(str(r[1] - 8) for r in rois)}" '
                 f'dur="{dur}s" repeatCount="indefinite"/></text>')
    animated = (_path([_brackets(r) for r in rois], dur, stroke=INK, stroke_opacity=".35",
                      stroke_width="1.2") + roi_label
                + _path([_arc(f, "knee", 34) for f in frames], dur, stroke=ACCENT, stroke_width="2")
                + skeleton(frames, dur, width=3.4, focus="knee") + readout)
    f0 = frames[bottom]
    still = (_path([_arc(f0, "knee", 34)], 0, stroke=ACCENT, stroke_width="2")
             + skeleton([f0], width=3.4, focus="knee")
             + f'<text x="{rx}" y="{y1 + 80}" fill="{INK}" font-size="60" font-weight="700" '
               f'font-family="Archivo, sans-serif">{angles[bottom]:.0f}&#176;</text>')
    return _svg(view, floor + ghosts + trails + _motion(animated, still), "fx-hero",
                "Animated pose skeleton performing a squat, with knee angle read-out")


def exercise_svg(key: str, dur: float | None = None, label_angle: bool = True) -> str:
    """Looping rep of one exercise with its rep-driving joint highlighted."""
    frames, angles, _ = rep_cycle(key)
    sc = SCENARIO_BY_NAME[f"{key}_good"]
    dur = dur or (sc.rep_seconds + sc.pause_seconds) * 1.5
    vertex = rep_vertex(key)
    x1, y1, x2, y2 = _bbox(frames)
    pad = 26
    view = (x1 - pad, y1 - pad, x2 - x1 + 2 * pad, y2 - y1 + pad + 10)
    turn = frames[turn_index(key)]
    label = ""
    if label_angle:
        label = _discrete([f"{a:.0f}&#176;" for a in angles], dur, x1 - pad + 6, y1 - pad + 22,
                          fill=INK, font_size="22", font_weight="600",
                          font_family="Archivo, sans-serif")
    floor = f'<path d="M{x1 - pad} {FLOOR - 4}H{x2 + pad}" stroke="{INK}" stroke-opacity=".14"/>'
    animated = (_path([_arc(f, vertex, 26) for f in frames], dur, stroke=ACCENT, stroke_width="2")
                + skeleton(frames, dur, width=4, focus=vertex) + label)
    still = skeleton([turn], width=4, focus=vertex)
    return _svg(view, floor + skeleton([turn], alpha=0.12, width=3) + _motion(animated, still),
                "fx-card", f"{EXERCISES[key].name} pose skeleton")


def pose_pair_svg(key: str) -> str:
    """Start and turn positions side by side, with the measured rep angle at each."""
    frames, angles, phases = rep_cycle(key)
    vertex = rep_vertex(key)
    ti = turn_index(key)
    x1, y1, x2, y2 = _bbox(frames)
    w, gap, pad = x2 - x1, 70, 30
    fs = (2 * w + gap + 2 * pad) / 30          # type scales with the view, not the pose
    body = f'<path d="M{x1 - pad} {FLOOR - 4}H{x2 + w + gap + pad}" stroke="{INK}" stroke-opacity=".14"/>'
    for n, i in enumerate((0, ti)):
        dx = n * (w + gap)
        f = {k: v + (dx, 0) for k, v in frames[i].items()}
        v = f[f"right_{vertex}"]
        body += (skeleton([f], width=4, focus=vertex)
                 + _path([_arc(f, vertex, 30)], 0, stroke=ACCENT, stroke_width="2")
                 + f'<text x="{v[0] + 40:.0f}" y="{v[1] + fs / 3:.0f}" fill="{INK}" font-size="{fs:.0f}" '
                   f'font-weight="600" font-family="Archivo, sans-serif">{angles[i]:.0f}&#176;</text>'
                 + f'<text x="{x1 + dx:.0f}" y="{y1 - 8:.0f}" {MONO} font-size="{fs * 0.75:.0f}" '
                   f'letter-spacing="1.4" fill="{MUTED}">{phases[i]}</text>')
    view = (x1 - pad, y1 - fs * 1.4, 2 * w + gap + 2 * pad, y2 - y1 + fs * 1.4 + 14)
    return _svg(view, body, "fx-pair", f"{EXERCISES[key].name}: start and turn positions")


def studies_background(width: int = 1240, height: int = 620) -> str:
    """Faint chronophotograph-style movement studies for the hero background (CSS data URI).

    Each study overlays evenly spaced samples of one scripted rep, start to turn.
    """
    def study(key, scale, x, floor_y, alpha, strobe=4):
        frames, _, _ = rep_cycle(key)
        bx1 = _bbox(frames)[0]
        ti = turn_index(key)
        idx = list(range(0, ti + 1, max(ti // strobe, 1)))
        last = len(idx) - 1
        body = "".join(skeleton([frames[i]], width=2.6, joints=k == last, head=k == last,
                                alpha=alpha * (0.35 + 0.65 * k / max(last, 1)))
                       for k, i in enumerate(idx))
        return (f'<g transform="translate({x - bx1 * scale:.0f} {floor_y - (FLOOR - 4) * scale:.0f}) '
                f'scale({scale})">{body}</g>')
    # foreground-left: a push-up study under the copy; far-right: a row of distant lifts
    body = (study("pushup", 0.8, 10, height - 20, 0.16, strobe=5)
            + study("bicep_curl", 0.36, width - 290, height - 185, 0.12)
            + study("squat", 0.34, width - 200, height - 185, 0.1)
            + study("shoulder_press", 0.36, width - 100, height - 185, 0.12)
            + f'<path d="M{width - 320} {height - 185}H{width - 20}" stroke="{INK}" stroke-opacity=".08"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'preserveAspectRatio="xMinYMax slice">{body}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def camera_placeholder_svg() -> str:
    """Framing guide for the idle camera stage: the start pose, framed as the ROI."""
    frames, _, _ = rep_cycle("squat")
    f = frames[0]
    x1, y1, x2, y2 = _roi(f)
    view = (x1 - 180, y1 - 30, (x2 - x1) + 360, (y2 - y1) + 60)
    return _svg(view, _path([_brackets((x1, y1, x2, y2), 22)], 0, stroke=INK, stroke_opacity=".4",
                            stroke_width="1.5") + skeleton([f], width=3.4, alpha=0.5),
                "fx-guide", "Framing guide: whole body inside the frame")


if __name__ == "__main__":        # self-check: phases come from the real state machine
    for k in EXERCISES:
        fr, ang, ph = rep_cycle(k)
        names = EXERCISES[k].phase_names
        assert ph[0] == names[0] and ph[turn_index(k)] == names[2], (k, ph)
        assert set(ph) <= set(names), (k, set(ph))
    print("ok", {k: (round(min(rep_cycle(k)[1])), round(max(rep_cycle(k)[1]))) for k in EXERCISES})
