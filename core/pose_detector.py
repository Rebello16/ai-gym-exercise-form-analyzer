"""Stage 2 - human pose detection with MediaPipe Pose Landmarker (BlazePose).

The detector returns 33 body landmarks per person. We ask for up to
``POSE["num_poses"]`` people so that extra people can be *detected and
reported*; only the most prominent person is analysed, so landmarks from
different people are never mixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config.thresholds import POSE, POSE_MODEL_PATH
from utils.helpers import ensure_pose_model
from utils.logger import get_logger

log = get_logger(__name__)

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]
LM = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# Body skeleton (face points omitted to keep the overlay uncluttered)
SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"), ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
]


@dataclass
class PoseDetection:
    """Landmarks of the selected (primary) person in pixel coordinates."""

    points: np.ndarray | None = None       # (33, 2) x, y in pixels
    visibility: np.ndarray | None = None   # (33,) in [0, 1]
    num_people: int = 0
    mask: np.ndarray | None = None         # (H, W) person probability map
    error: str | None = None
    all_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return self.points is not None


def prominence(points: np.ndarray, visibility: np.ndarray) -> float:
    """Bounding-box area of visible landmarks x mean visibility.

    The largest, most clearly visible person is usually the one exercising
    closest to the camera.
    """
    visible = points[visibility >= POSE["landmark_visibility_threshold"]]
    if len(visible) < 2:
        return 0.0
    span = visible.max(axis=0) - visible.min(axis=0)
    return float(span[0] * span[1] * visibility.mean())


class PoseDetector:
    """Thin, reusable wrapper around the MediaPipe Tasks PoseLandmarker.

    Created once and reused for every frame (model loading is the expensive
    part). Uses VIDEO running mode, which tracks the person between frames
    instead of re-running the full detector each time.
    """

    def __init__(self, with_segmentation: bool = False, num_poses: int | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        ensure_pose_model()
        self._mp = mp
        self.with_segmentation = with_segmentation
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses or POSE["num_poses"],
            min_pose_detection_confidence=POSE["min_detection_confidence"],
            min_pose_presence_confidence=POSE["min_presence_confidence"],
            min_tracking_confidence=POSE["min_tracking_confidence"],
            output_segmentation_masks=with_segmentation,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1
        log.info("PoseDetector ready (segmentation=%s)", with_segmentation)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> PoseDetection:
        """Detect poses in an RGB frame. Never raises; errors are reported."""
        # VIDEO mode requires strictly increasing timestamps, even across
        # sessions that reuse this detector.
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        h, w = rgb.shape[:2]
        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=np.ascontiguousarray(rgb))
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except Exception as exc:  # pragma: no cover - depends on native runtime
            log.exception("Pose detector failure")
            return PoseDetection(error=f"Pose detector failure: {exc}")

        people = []
        for i, landmarks in enumerate(result.pose_landmarks):
            pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
            vis = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
            people.append((prominence(pts, vis), i, pts, vis))
        if not people:
            return PoseDetection()

        boxes = [(int(p[2][:, 0].min()), int(p[2][:, 1].min()),
                  int(p[2][:, 0].max()), int(p[2][:, 1].max())) for p in people]
        _, idx, pts, vis = max(people, key=lambda p: p[0])
        mask = None
        if self.with_segmentation and result.segmentation_masks:
            mask = result.segmentation_masks[idx].numpy_view().astype(np.float32)
            if mask.ndim == 3:
                mask = mask[..., 0]
        return PoseDetection(pts, vis, len(people), mask, None, boxes)

    def close(self) -> None:
        self._landmarker.close()
