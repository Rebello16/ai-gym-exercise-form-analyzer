"""The end-to-end analysis pipeline, one frame at a time.

    frame -> preprocess -> pose detection -> ROI / mask / sharpness
          -> features -> recognition -> rep state machine -> form analysis
          -> scoring -> visualisation -> session log
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config.thresholds import POSE, PREPROCESSING, PROCESSING
from core.exercise_recognizer import UNKNOWN, ExerciseRecognizer, Recognition
from core.feature_extractor import FeatureExtractor, Features
from core.form_analyzer import FormAnalyzer, LiveCheck, RepResult
from core.pose_detector import PoseDetection, PoseDetector
from core.rep_counter import RepCounter, RepState
from core.scoring import status as score_status
from core.session import SessionLog
from exercises import EXERCISES
from exercises.base_exercise import Exercise
from processing.image_processing import (binary_person_mask, landmarks_roi, sharpness,
                                         suppress_background)
from processing.preprocessing import PreprocessResult, preprocess_frame

_STATUS_RANK = {"neutral": 0, "good": 1, "warning": 2, "bad": 3}


class ExerciseTracker:
    """Rep counter + form analyzer for one exercise."""

    def __init__(self, exercise: Exercise):
        cfg = exercise.rep_thresholds
        self.exercise = exercise
        self.counter = RepCounter(cfg["extended"], cfg["flexed"], exercise.start_extended,
                                  cfg["min_rep_seconds"])
        self.analyzer = FormAnalyzer(exercise)
        # Frames spent resting in the start position before the rep (so e.g.
        # "full extension before the curl" is measurable) + frames of the rep.
        self._start_frames: deque[dict] = deque(maxlen=60)
        self._rep_frames: list[dict] = []

    @property
    def phase(self) -> str:
        names = self.exercise.phase_names
        return {RepState.WAITING: "GET IN POSITION", RepState.START: names[0],
                RepState.TO_TURN: names[1], RepState.TURN: names[2],
                RepState.TO_START: names[3]}[self.counter.state]

    def update(self, t: float, values: dict) -> RepResult | None:
        event = self.counter.update(values.get(self.exercise.rep_feature), t)
        if self.counter.state == RepState.WAITING:
            self._start_frames.clear()
            self._rep_frames.clear()
            return None
        if event is None:
            (self._rep_frames if self.counter.in_rep else self._start_frames).append(values)
            return None
        frames = list(self._start_frames) + self._rep_frames + [values]
        self._start_frames.clear()
        self._rep_frames.clear()
        self._start_frames.append(values)
        return self.analyzer.evaluate_rep(frames, event.start_time, event.end_time,
                                          index=0, kind=event.kind)


@dataclass
class FrameStatus:
    """Everything the UI needs to show for one frame."""

    t: float
    pose_detected: bool = False
    num_people: int = 0
    mode: str = "auto"
    exercise: str | None = None           # active exercise key
    recognition: Recognition | None = None
    reps: int = 0
    phase: str = "-"
    status: str = "neutral"               # good / warning / bad / neutral
    live_checks: list[LiveCheck] = field(default_factory=list)
    last_rep: RepResult | None = None
    feedback: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fps: float = 0.0
    features: Features | None = None
    pre: PreprocessResult | None = None
    pose: PoseDetection | None = None
    sharpness: float = 0.0


class AnalysisPipeline:
    """Stateful per-session pipeline. Create one per video / webcam session."""

    def __init__(self, detector: PoseDetector, source: str = "video",
                 manual_exercise: str | None = None, enhance: bool = True,
                 background_focus: bool = False, max_width: int | None = None):
        if manual_exercise is not None and manual_exercise not in EXERCISES:
            raise ValueError(f"Unknown exercise '{manual_exercise}'")
        self.detector = detector
        self.manual_exercise = manual_exercise
        self.enhance = enhance
        self.background_focus = background_focus
        self.max_width = max_width or PROCESSING["max_width"]
        self.extractor = FeatureExtractor()
        self.recognizer = ExerciseRecognizer()
        keys = [manual_exercise] if manual_exercise else list(EXERCISES)
        self.trackers = {k: ExerciseTracker(EXERCISES[k]) for k in keys}
        self.session = SessionLog(source=source, mode="manual" if manual_exercise else "auto")
        self._roi: tuple[int, int, int, int] | None = None
        self._fps = 0.0
        self._last_rep: RepResult | None = None
        self._last_partial: RepResult | None = None
        self._ever_detected = False

    @property
    def mode(self) -> str:
        return self.session.mode

    def rep_count(self, exercise: str | None) -> int:
        return sum(1 for r in self.session.reps if r.exercise == exercise)

    def process(self, frame_bgr: np.ndarray, t: float) -> tuple[np.ndarray, FrameStatus]:
        """Analyse one frame taken at time ``t`` seconds. Returns (annotated BGR, status)."""
        from processing.visualization import draw_overlay   # avoid import cycle

        tick = time.perf_counter()
        st = FrameStatus(t=t, mode=self.mode)

        # 1. preprocessing
        pre = preprocess_frame(frame_bgr, self._roi, self.max_width, self.enhance)
        st.pre = pre
        # 2. pose detection
        pose = self.detector.detect(pre.rgb, int(t * 1000))
        st.pose, st.pose_detected, st.num_people = pose, pose.detected, pose.num_people
        display = pre.bgr
        # 3. subject ROI / sharpness / mask
        if pose.detected:
            self._ever_detected = True
            self._roi = landmarks_roi(pose.points, pose.visibility, pre.bgr.shape,
                                      POSE["landmark_visibility_threshold"])
            st.sharpness = sharpness(pre.bgr, self._roi)
            if self.background_focus and pose.mask is not None:
                display = suppress_background(pre.bgr, binary_person_mask(pose.mask))
        else:
            self._roi = None
        # 4. features
        feats = self.extractor.extract(pose)
        st.features = feats
        # 5. recognition (always computed, shown alongside manual choice)
        if pose.detected:
            st.recognition = self.recognizer.update(t, feats.values)
        else:
            st.recognition = self.recognizer.current
        active = self.manual_exercise or (
            st.recognition.label if st.recognition.label != UNKNOWN else None)
        st.exercise = active
        # 6-8. rep state machines + form analysis + scoring
        for key, tracker in self.trackers.items():
            result = tracker.update(t, feats.values)
            if result is None or key != active:
                continue
            if result.kind == "rep":
                result.index = self.rep_count(key) + 1
                self.session.reps.append(result)
                self._last_rep, self._last_partial = result, None
            else:
                result.index = len(self.session.partials) + 1
                self.session.partials.append(result)
                self._last_partial = result
        if active:
            tracker = self.trackers[active]
            st.reps, st.phase = self.rep_count(active), tracker.phase
            st.live_checks = tracker.analyzer.live(feats.values) if pose.detected else []
            self._compose_feedback(st, tracker.exercise)
        self._collect_warnings(st, pose, pre, feats)

        # 9. visualisation
        dt = time.perf_counter() - tick
        self._fps = (1 / dt) if self._fps == 0 else 0.9 * self._fps + 0.1 / max(dt, 1e-6)
        st.fps = self._fps
        annotated = draw_overlay(display.copy(), st)
        # 10. log
        self._log(st, feats, pre)
        return annotated, st

    # -- helpers ----------------------------------------------------------
    def _compose_feedback(self, st: FrameStatus, exercise: Exercise) -> None:
        live_bad = [lc for lc in st.live_checks if score_status(lc.score) != "good"]
        st.feedback = [lc.check.message.format(value=lc.value) for lc in live_bad]
        st.last_rep = self._last_rep if (self._last_rep and self._last_rep.exercise == exercise.key) else None
        if self._last_partial is not None and self._last_partial.exercise == exercise.key:
            st.feedback.append(f"Partial rep not counted: {exercise.phase_names[2].lower()} "
                               "position was not reached")
            st.feedback += [m for m in self._last_partial.issues if m not in st.feedback]
        elif st.last_rep is not None:
            st.feedback += [m for m in st.last_rep.issues if m not in st.feedback]
        statuses = [score_status(lc.score) for lc in st.live_checks]
        if st.last_rep is not None and st.last_rep.score is not None:
            statuses.append(score_status(st.last_rep.score))
        if self._last_partial is not None and self._last_partial.exercise == exercise.key:
            statuses.append("bad")
        st.status = max(statuses, key=_STATUS_RANK.get, default="neutral")

    def _collect_warnings(self, st: FrameStatus, pose: PoseDetection,
                          pre: PreprocessResult, feats: Features) -> None:
        w = st.warnings
        if pose.error:
            w.append(pose.error)
        elif not pose.detected:
            w.append("Person left the frame - tracking paused" if self._ever_detected
                     else "No person detected - step into the camera view")
        else:
            if pose.num_people > 1:
                w.append(f"{pose.num_people} people detected - analysing the most prominent "
                         "person only (designed for one subject)")
            if st.exercise:
                rep_feature = EXERCISES[st.exercise].rep_feature
                if feats.get(rep_feature) is None:
                    w.append("Key joints not visible - make sure the whole body is in view")
            if st.sharpness and st.sharpness < PREPROCESSING["blur_laplacian_threshold"]:
                w.append("Frame is blurry - landmarks may be less reliable")
        if pre.poor_lighting:
            w.append("Poor lighting - contrast enhancement (CLAHE) applied" if "clahe" in pre.applied
                     else "Poor lighting detected")
        if self.mode == "auto" and st.exercise is None and pose.detected:
            w.append("Auto-detect: start moving to identify the exercise")

    def _log(self, st: FrameStatus, feats: Features, pre: PreprocessResult) -> None:
        rec = st.recognition
        self.session.frames.append({
            "t": round(st.t, 3), "pose_detected": st.pose_detected, "num_people": st.num_people,
            "exercise": st.exercise or "", "recognized": rec.label if rec else UNKNOWN,
            "recognition_confidence": round(rec.confidence, 3) if rec else 0.0,
            "reps": st.reps, "phase": st.phase, "status": st.status,
            "brightness": round(pre.brightness, 1), "contrast": round(pre.contrast, 1),
            "noise_sigma": round(pre.noise_sigma, 2), "sharpness": round(st.sharpness, 1),
            "preprocessing": "+".join(pre.applied), "side": feats.side, "view": feats.view,
            **{k: (None if v is None else round(v, 3)) for k, v in feats.values.items()},
        })
