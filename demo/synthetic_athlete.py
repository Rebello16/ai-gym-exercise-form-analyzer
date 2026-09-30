"""Synthetic demonstration videos.

Renders a side-view articulated human figure whose joints follow scripted
kinematics (forward kinematics from segment angles). The videos are then
analysed by the *real* pipeline - MediaPipe has to find the person, and the
counts/scores come out of the normal processing. Because the script is
known, the videos double as ground truth for integration tests.

Run ``python -m demo.synthetic_athlete`` to (re)generate data/sample/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config.thresholds import SAMPLE_DIR

W, H, FPS = 640, 480, 30
FLOOR = 430
FIGURE_SCALE = 0.82          # keeps raised arms inside the frame
SHIN, THIGH, TORSO, UPPER, FORE = 95, 100, 135, 75, 70
SKIN, SHIRT, PANTS, SHOE, HAIR = (140, 170, 215), (60, 60, 190), (90, 50, 30), (35, 35, 35), (40, 40, 60)


def u(phi: float) -> np.ndarray:
    """Unit vector at angle ``phi`` (deg) from vertical-up, positive toward +x."""
    a = math.radians(phi)
    return np.array([math.sin(a), -math.cos(a)])


@dataclass
class Scenario:
    """A scripted exercise clip."""

    name: str
    exercise: str
    reps: int
    params: dict = field(default_factory=dict)
    rep_seconds: float = 2.4
    pause_seconds: float = 0.5
    lead_seconds: float = 1.2
    low_light: bool = False
    second_person: bool = False
    description: str = ""

    @property
    def duration(self) -> float:
        return self.lead_seconds + self.reps * (self.rep_seconds + self.pause_seconds) + 0.8

    def progress(self, t: float) -> float:
        """Movement progress p in [0, 1] (0 = start position, 1 = turn position)."""
        t -= self.lead_seconds
        cycle = self.rep_seconds + self.pause_seconds
        if t < 0 or t >= self.reps * cycle:
            return 0.0
        tau = (t % cycle) / self.rep_seconds
        return 0.0 if tau >= 1 else (1 - math.cos(2 * math.pi * tau)) / 2


# ---------------------------------------------------------------------------
# Pose generators: return named joint positions for a progress value
# ---------------------------------------------------------------------------
def _standing(ankle, shin, thigh, torso, upper, fore):
    knee = ankle + SHIN * u(shin)
    hip = knee + THIGH * u(thigh)
    shoulder = hip + TORSO * u(torso)
    elbow = shoulder + UPPER * u(upper)
    wrist = elbow + FORE * u(fore)
    head = shoulder + 48 * u(torso + 8)
    return dict(ankle=ankle, knee=knee, hip=hip, shoulder=shoulder, elbow=elbow,
                wrist=wrist, head=head, heel=ankle + (-14, 10), toe=ankle + (42, 12))


def squat_pose(p: float, q: dict) -> dict:
    knee_angle = 175 - (175 - q["knee_min"]) * p
    shin = q["shin_max"] * p
    return _standing(np.array([300.0, FLOOR - 12]), shin, shin - (180 - knee_angle),
                     q["lean_max"] * p, 90, 90)


def curl_pose(p: float, q: dict) -> dict:
    elbow_angle = q["elbow_max"] - (q["elbow_max"] - q["elbow_min"]) * p
    upper = 180 - q["swing"] * p
    torso = -q["torso_sway"] * p
    return _standing(np.array([320.0, FLOOR - 12]), 0, 0, torso, upper,
                     upper - (180 - elbow_angle))


def press_pose(p: float, q: dict) -> dict:
    elbow_angle = q["elbow_bottom"] + (q["elbow_top"] - q["elbow_bottom"]) * p
    upper = q["upper_bottom"] + (q["upper_top"] - q["upper_bottom"]) * p
    return _standing(np.array([300.0, FLOOR - 12]), 0, 0, -q["lean_back"] * p, upper,
                     upper - (180 - elbow_angle))


def pushup_pose(p: float, q: dict) -> dict:
    elbow_angle = q["elbow_top"] - (q["elbow_top"] - q["elbow_min"]) * p
    hand = np.array([470.0, FLOOR - 6])
    reach = math.sqrt(UPPER ** 2 + FORE ** 2 - 2 * UPPER * FORE * math.cos(math.radians(elbow_angle)))
    shoulder = hand + (0, -reach)
    # elbow: circle intersection, bending back toward the feet
    a = (UPPER ** 2 - FORE ** 2 + reach ** 2) / (2 * reach)
    hgt = math.sqrt(max(UPPER ** 2 - a ** 2, 0))
    elbow = shoulder + (0, a) + (-hgt, 0)
    body = SHIN + THIGH + TORSO
    ankle = np.array([0.0, FLOOR - 14])
    ankle[0] = shoulder[0] - math.sqrt(body ** 2 - (ankle[1] - shoulder[1]) ** 2)
    direction = (shoulder - ankle) / body
    normal = np.array([-direction[1], direction[0]])        # points downward
    if normal[1] < 0:
        normal = -normal
    hip = ankle + direction * (SHIN + THIGH) + normal * q["sag"] * body
    knee = ankle + (hip - ankle) * SHIN / (SHIN + THIGH)
    head = shoulder + direction * 48 + (0, -8)
    return dict(ankle=ankle, knee=knee, hip=hip, shoulder=shoulder, elbow=elbow,
                wrist=hand, head=head, heel=ankle + (-8, -10), toe=ankle + (6, 12))


POSES = {"squat": squat_pose, "bicep_curl": curl_pose,
         "shoulder_press": press_pose, "pushup": pushup_pose}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _limb(img, a, b, width, color):
    a, b = tuple(int(v) for v in a), tuple(int(v) for v in b)
    cv2.line(img, a, b, color, width, cv2.LINE_AA)
    cv2.circle(img, a, width // 2, color, -1, cv2.LINE_AA)
    cv2.circle(img, b, width // 2, color, -1, cv2.LINE_AA)


def _shade(color, f=0.7):
    return tuple(int(c * f) for c in color)


def _arm(img, shoulder, elbow, wrist, s, shade):
    """Bare arm with a short sleeve, outlined so it stands out against the torso."""
    for width, color in ((s(26), _shade(SKIN, 0.55 * shade)), (s(22), _shade(SKIN, shade))):
        _limb(img, shoulder, elbow, width, color)
        _limb(img, elbow, wrist, width - s(3), color)
    _limb(img, shoulder, shoulder + (elbow - shoulder) * 0.35, s(28), _shade(SHIRT, shade))
    cv2.circle(img, tuple(int(v) for v in wrist), s(12), _shade(SKIN, shade), -1, cv2.LINE_AA)


def draw_figure(img: np.ndarray, j: dict, scale: float = 1.0, offset=(0, 0), dumbbell=False) -> None:
    """Draw the figure (far-side limbs darker, drawn first)."""
    j = {k: np.asarray(v, float) * scale + offset for k, v in j.items()}
    s = lambda w: max(2, int(w * scale))
    far = np.array([-7, -4]) * scale
    _limb(img, j["knee"] + far, j["ankle"] + far, s(30), _shade(PANTS))
    _limb(img, j["hip"] + far, j["knee"] + far, s(38), _shade(PANTS))
    _limb(img, j["ankle"] + far, j["toe"] + far, s(16), _shade(SHOE))
    _arm(img, j["shoulder"] + far, j["elbow"] + far, j["wrist"] + far, s, 0.75)
    _limb(img, j["ankle"], j["toe"], s(18), SHOE)
    _limb(img, j["heel"], j["ankle"], s(16), SHOE)
    _limb(img, j["knee"], j["ankle"], s(32), PANTS)
    _limb(img, j["hip"], j["knee"], s(42), PANTS)
    _limb(img, j["hip"], j["shoulder"], s(58), SHIRT)
    head = tuple(int(v) for v in j["head"])
    cv2.circle(img, head, s(29), SKIN, -1, cv2.LINE_AA)
    facing = 1 if (j["head"] - j["shoulder"])[0] >= -5 else -1
    cv2.ellipse(img, (head[0] - facing * s(4), head[1] - s(10)), (s(29), s(20)), 0, 180, 360, HAIR, -1)
    cv2.circle(img, (head[0] + facing * s(14), head[1] - s(3)), max(1, s(3)), (30, 30, 30), -1)
    if dumbbell:
        w = tuple(int(v) for v in j["wrist"])
        cv2.rectangle(img, (w[0] - s(20), w[1] - s(6)), (w[0] + s(20), w[1] + s(6)), (50, 50, 50), -1)
    _arm(img, j["shoulder"], j["elbow"], j["wrist"], s, 1.0)


def _background() -> np.ndarray:
    img = np.empty((H, W, 3), np.uint8)
    ramp = np.linspace(215, 185, FLOOR)[:, None, None]
    img[:FLOOR] = (ramp * np.array([0.97, 1.0, 1.02])).astype(np.uint8)
    img[FLOOR:] = (115, 125, 135)
    cv2.line(img, (0, FLOOR), (W, FLOOR), (90, 95, 100), 2)
    return img


def render_scenario(sc: Scenario, path: Path) -> Path:
    """Render one scenario to a WebM (VP8) file - playable in browsers and by OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {path}")
    base = _background()
    pose_fn = POSES[sc.exercise]
    offset = (1 - FIGURE_SCALE) * np.array([320.0, FLOOR])
    if sc.second_person:
        offset += (-120, 0)      # make room for the bystander on the right
    for i in range(int(sc.duration * FPS)):
        img = base.copy()
        if sc.second_person:     # smaller (further away) bystander standing still
            draw_figure(img, _standing(np.array([0.0, 0.0]), 0, 0, 0, 180, 180),
                        scale=0.7, offset=(470, FLOOR - 10))
        draw_figure(img, pose_fn(sc.progress(i / FPS), sc.params), scale=FIGURE_SCALE,
                    offset=offset,
                    dumbbell=sc.exercise in ("bicep_curl", "shoulder_press"))
        if sc.low_light:
            img = cv2.convertScaleAbs(img, alpha=0.28)
        noise = np.empty(img.shape, np.int16)
        cv2.randn(noise, 0, 7 if sc.low_light else 2)        # sensor noise
        writer.write(cv2.add(img, noise, dtype=cv2.CV_8U))
    writer.release()
    return path


GOOD_SQUAT = dict(knee_min=88, shin_max=30, lean_max=32)
SCENARIOS = [
    Scenario("squat_good", "squat", 5, GOOD_SQUAT, description="Full-depth squats, upright torso"),
    Scenario("squat_bad", "squat", 5, dict(knee_min=112, shin_max=22, lean_max=62),
             description="Shallow squats with excessive forward lean"),
    Scenario("bicep_curl_good", "bicep_curl", 5,
             dict(elbow_max=170, elbow_min=38, swing=6, torso_sway=2),
             description="Full-range curls, still upper arm"),
    Scenario("bicep_curl_bad", "bicep_curl", 5,
             dict(elbow_max=140, elbow_min=68, swing=45, torso_sway=16),
             description="Half curls with swinging arm and torso"),
    Scenario("pushup_good", "pushup", 5, dict(elbow_top=170, elbow_min=80, sag=0.0),
             description="Deep push-ups, straight body line"),
    Scenario("pushup_bad", "pushup", 5, dict(elbow_top=165, elbow_min=100, sag=0.13),
             description="Shallow push-ups with sagging hips"),
    Scenario("shoulder_press_good", "shoulder_press", 5,
             dict(elbow_bottom=85, elbow_top=175, upper_bottom=95, upper_top=5, lean_back=2),
             description="Full lockout, stacked wrists, upright torso"),
    Scenario("shoulder_press_bad", "shoulder_press", 5,
             dict(elbow_bottom=90, elbow_top=150, upper_bottom=125, upper_top=40, lean_back=24),
             description="Partial lockout, leaning back, wrists drifting forward"),
    Scenario("squat_low_light", "squat", 4, GOOD_SQUAT, low_light=True,
             description="Good squats filmed in poor, noisy lighting (tests CLAHE + denoising)"),
    Scenario("squat_two_people", "squat", 3, GOOD_SQUAT, second_person=True,
             description="A bystander in the frame (tests multi-person handling)"),
]
SCENARIO_BY_NAME = {s.name: s for s in SCENARIOS}


def ensure_samples(names: list[str] | None = None, folder: Path = SAMPLE_DIR) -> dict[str, Path]:
    """Render any missing sample videos; returns name -> path."""
    out = {}
    for sc in SCENARIOS:
        if names and sc.name not in names:
            continue
        path = folder / f"{sc.name}.webm"
        if not path.exists() or path.stat().st_size == 0:
            render_scenario(sc, path)
        out[sc.name] = path
    return out


if __name__ == "__main__":
    for sc in SCENARIOS:
        print("rendering", render_scenario(sc, SAMPLE_DIR / f"{sc.name}.webm"))
