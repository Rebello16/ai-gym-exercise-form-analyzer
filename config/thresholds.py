"""Centralised configuration for the whole analyzer.

Every tunable number lives here so that behaviour can be adjusted (or
explained in a viva) without hunting through the code base.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root -> no machine-specific paths)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

SUPPORTED_VIDEO_TYPES = ("mp4", "avi", "mov")

# --------------------------------------------------------------------------
# Processing / performance
# --------------------------------------------------------------------------
PROCESSING = {
    "max_width": 640,          # frames wider than this are downscaled
    "frame_skip": 1,           # analyse every n-th frame of an uploaded video
    "webcam_index": 0,
    "webcam_max_seconds": 120,
}

# --------------------------------------------------------------------------
# Image preprocessing (DIP stage)
# --------------------------------------------------------------------------
PREPROCESSING = {
    # Lighting analysis on the subject ROI (grey-level statistics, 0-255)
    "dark_mean_threshold": 70,       # mean intensity below this = poor lighting
    "low_contrast_std_threshold": 30,  # std-dev below this = low contrast
    # CLAHE contrast enhancement (applied only when lighting is poor)
    "clahe_clip_limit": 2.5,
    "clahe_tile_grid": (8, 8),
    # Noise estimation (Immerkaer) + Gaussian denoising
    "noise_sigma_threshold": 4.0,     # estimated sigma above this -> denoise
    # CLAHE amplifies noise, and dark frames under-report it (values clip at 0),
    # so denoise at a lower sigma whenever CLAHE is about to be applied.
    "noise_sigma_threshold_dark": 1.5,
    "gaussian_kernel": (5, 5),
    # Sharpness (variance of Laplacian) below this -> "blurry frame" warning
    "blur_laplacian_threshold": 40.0,
    # Segmentation mask -> binary mask
    "mask_threshold": 0.5,
    "mask_morph_kernel": 5,
    "background_blur_kernel": (31, 31),
    "roi_padding": 0.15,              # fractional padding around the person box
}

# --------------------------------------------------------------------------
# Pose detection
# --------------------------------------------------------------------------
POSE = {
    "num_poses": 2,                      # >1 lets us detect & warn about extra people
    "min_detection_confidence": 0.5,
    "min_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "landmark_visibility_threshold": 0.5,  # below this a landmark is "missing"
    "max_missing_seconds": 1.5,            # longer gap -> in-progress rep discarded
}

# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
FEATURES = {
    "ema_alpha": 0.5,            # temporal smoothing factor (1 = no smoothing)
    "median_window": 5,          # per-rep median filter (removes one-off landmark spikes)
    "side_switch_margin": 0.15,  # visibility margin needed to switch body side
    "frontal_view_ratio": 0.45,  # shoulder-width / torso-length above this = facing camera
}

# --------------------------------------------------------------------------
# Exercise recognition
# --------------------------------------------------------------------------
RECOGNITION = {
    "window_seconds": 2.0,
    "min_samples": 8,
    "horizontal_body_deg": 50.0,    # body (ankle->shoulder) inclination above this = lying
    "min_rom_deg": 15.0,            # joint range of motion below this = no movement
    "rom_full_deg": 40.0,           # ROM at which the movement cue saturates
    "min_confidence": 0.5,
    "vote_frames": 15,              # label must win the majority of recent frames
}

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
# Grade bands: (minimum score, label). Checked top-down.
SCORE_BANDS = [
    (100, "Excellent detected form"),
    (90, "Very good"),
    (75, "Acceptable"),
    (60, "Needs improvement"),
    (0, "Poor detected form"),
]
STATUS_GOOD = 75      # check score >= this -> green / no feedback
STATUS_WARNING = 50   # check score >= this -> yellow, below -> red

# Rep tempo (duration of the movement, seconds) shared by all exercises:
# score 100 at `ideal` seconds or slower, 0 at `zero` seconds.
TEMPO = {"ideal": 1.0, "zero": 0.3}

# --------------------------------------------------------------------------
# Per-exercise thresholds
# --------------------------------------------------------------------------
# rep:    hysteresis state machine thresholds on the exercise's rep signal.
#         `extended` / `flexed` are joint angles in degrees; the band between
#         them is the hysteresis dead-zone that absorbs landmark jitter.
# checks: (ideal, zero) pairs -> score is 100 at `ideal` (or better) and falls
#         linearly to 0 at `zero`. Direction is inferred from the ordering.
EXERCISE_THRESHOLDS = {
    "squat": {
        "rep": {"extended": 160, "flexed": 125, "min_rep_seconds": 0.6},
        "checks": {
            "depth": (95, 140),           # minimum knee angle (deg)
            "back_angle": (40, 70),       # maximum torso inclination from vertical (deg)
            "knee_tracking": (0.25, 0.7),  # knee travel past toes / shin length
            "knee_alignment": (0.9, 0.6),  # knee width / ankle width (front view)
        },
    },
    "bicep_curl": {
        "rep": {"extended": 130, "flexed": 90, "min_rep_seconds": 0.6},
        "checks": {
            "curl_height": (55, 100),     # minimum elbow angle (deg)
            "extension": (155, 125),      # maximum elbow angle (deg)
            "elbow_stability": (20, 50),  # maximum upper-arm swing from vertical (deg)
            "torso_swing": (6, 20),       # range of torso inclination during rep (deg)
        },
    },
    "pushup": {
        "rep": {"extended": 145, "flexed": 120, "min_rep_seconds": 0.6},
        "checks": {
            "depth": (90, 130),           # minimum elbow angle (deg)
            "lockout": (160, 135),        # maximum elbow angle (deg)
            "hip_sag": (0.04, 0.15),      # hip below shoulder-ankle line / body length
            "hip_pike": (-0.04, -0.15),   # hip above shoulder-ankle line / body length
        },
    },
    "shoulder_press": {
        "rep": {"extended": 135, "flexed": 110, "min_rep_seconds": 0.6},
        "checks": {
            "lockout": (165, 135),        # maximum elbow angle (deg)
            "bottom_range": (95, 125),    # minimum elbow angle (deg)
            "back_arch": (10, 28),        # maximum torso inclination (deg)
            "wrist_stack": (15, 45),      # maximum forearm deviation from vertical (deg)
        },
    },
}

# Visual colours (BGR for OpenCV)
COLORS = {
    "good": (80, 200, 80),
    "warning": (0, 200, 255),
    "bad": (60, 60, 230),
    "neutral": (230, 230, 230),
    "joint": (255, 255, 255),
    "panel": (30, 30, 30),
}
