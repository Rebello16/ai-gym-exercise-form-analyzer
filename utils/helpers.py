"""Small shared helpers."""
from __future__ import annotations

import urllib.request
from pathlib import Path

from config.thresholds import POSE_MODEL_PATH, POSE_MODEL_URL, SUPPORTED_VIDEO_TYPES
from utils.logger import get_logger

log = get_logger(__name__)


def ensure_pose_model(path: Path = POSE_MODEL_PATH) -> Path:
    """Return the pose model path, downloading it once if it is missing."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading pose model to %s", path)
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, path)
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Pose model not found and download failed ({exc}). "
            f"Download it manually from {POSE_MODEL_URL} into {path.parent}"
        ) from exc
    return path


def is_supported_video(filename: str) -> bool:
    """True if the file extension is an accepted video type."""
    return Path(filename).suffix.lower().lstrip(".") in SUPPORTED_VIDEO_TYPES


def format_seconds(seconds: float) -> str:
    """Format a duration as m:ss."""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"
