"""Frame acquisition (video files, webcam) and annotated-video output."""
from __future__ import annotations

import math
import platform
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from config.thresholds import OUTPUT_DIR, PROCESSING
from core.pipeline import AnalysisPipeline, FrameStatus
from utils.logger import get_logger

log = get_logger(__name__)


class VideoSourceError(RuntimeError):
    """Raised with a user-friendly message when a source cannot be read."""


def video_info(path: str | Path) -> tuple[float, int]:
    """(fps, frame_count) of a video file; validates that it can be decoded."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoSourceError("Could not open the video - the file may be corrupt "
                                   "or use an unsupported codec.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoSourceError("The video contains no readable frames.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and math.isfinite(fps) and fps > 1 else 30.0
        return fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def iterate_video(path: str | Path, frame_skip: int = 1) -> Iterator[tuple[np.ndarray, float, int]]:
    """Yield (frame, timestamp_seconds, frame_index) for every ``frame_skip``-th frame.

    Timestamps come from the frame index and the file's FPS, so results do
    not depend on how fast this computer processes the video.
    """
    fps, _ = video_info(path)
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, frame_skip) == 0:
                yield frame, idx / fps, idx
            idx += 1
    finally:
        cap.release()


def create_writer(stem: str, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path] | None:
    """Browser-playable WebM (VP8) with an MP4 fallback. Returns None if neither works."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, fourcc in (("webm", "VP80"), ("mp4", "mp4v")):
        path = OUTPUT_DIR / f"{stem}.{ext}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
    log.warning("No video encoder available; annotated video will not be saved")
    return None


def analyze_video(path: str | Path, pipeline: AnalysisPipeline, frame_skip: int | None = None,
                  save_output: bool = True,
                  on_frame: Callable[[np.ndarray, FrameStatus, float], None] | None = None,
                  ) -> Path | None:
    """Run the pipeline over a whole video file.

    Args:
        on_frame: optional callback(annotated_bgr, status, progress 0-1) for live UI updates.
    Returns:
        Path of the annotated output video (or None if not saved).
    """
    frame_skip = frame_skip or PROCESSING["frame_skip"]
    fps, total = video_info(path)
    writer, out_path = None, None
    for frame, t, idx in iterate_video(path, frame_skip):
        annotated, status = pipeline.process(frame, t)
        if save_output and writer is None:
            h, w = annotated.shape[:2]
            created = create_writer(f"{Path(path).stem}_analyzed", fps / frame_skip, (w, h))
            if created:
                writer, out_path = created
            save_output = created is not None
        if writer is not None:
            writer.write(annotated)
        if on_frame is not None:
            on_frame(annotated, status, min(1.0, (idx + 1) / total) if total > 0 else 0.0)
    if writer is not None:
        writer.release()
    if not pipeline.session.frames:
        raise VideoSourceError("No frames could be analysed from this video.")
    return out_path


def open_webcam(index: int | None = None) -> cv2.VideoCapture:
    """Open the webcam or raise a VideoSourceError explaining what to check."""
    index = PROCESSING["webcam_index"] if index is None else index
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    raise VideoSourceError(
        f"Camera {index} is unavailable. Check that a webcam is connected, not used by "
        "another app, and that camera access is allowed (Windows: Settings > Privacy & "
        "security > Camera). You can use Demo Mode or upload a video instead.")
