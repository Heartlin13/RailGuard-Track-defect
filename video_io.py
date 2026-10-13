"""Video I/O module for RailGuard.

Handles robust video reading, frame skipping, timestamp calculation,
and encoding annotated video output with fallback codec support.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("rail-guard.video_io")


@dataclass
class VideoMetadata:
    """Metadata container for video files."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float


class VideoReader:
    """Reads video files frame by frame with support for frame skipping and timestamps."""

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path).resolve()
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file (corrupted or unsupported codec): {self.video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if (fps and fps > 0 and not np.isnan(fps)) else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0

        logger.info(
            f"Opened video '{self.video_path.name}': {self.width}x{self.height} @ "
            f"{self.fps:.2f} FPS ({self.total_frames} frames, {self.duration_sec:.1f}s)"
        )

    @property
    def metadata(self) -> VideoMetadata:
        """Returns metadata for the opened video."""
        return VideoMetadata(
            width=self.width,
            height=self.height,
            fps=self.fps,
            total_frames=self.total_frames,
            duration_sec=self.duration_sec,
        )

    def frames(self, frame_skip: int = 1) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Generator that yields (frame_idx, timestamp_sec, frame) with frame skipping.

        Args:
            frame_skip: Interval of frames to yield (1 = every frame, 2 = every 2nd frame, etc.)
        """
        frame_skip = max(1, frame_skip)
        frame_idx = 0

        # Rewind to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_skip == 0:
                timestamp_sec = frame_idx / self.fps
                yield frame_idx, timestamp_sec, frame

            frame_idx += 1

    def release(self) -> None:
        """Closes video capture."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """Writes processed frames to an output video with automatic codec fallback."""

    SUPPORTED_CODECS = ["mp4v", "XVID", "MJPG", "avc1"]

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
    ):
        self.output_path = Path(output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = max(1.0, fps)
        self.writer: Optional[cv2.VideoWriter] = None

        self._init_writer()

    def _init_writer(self) -> None:
        """Attempt to initialize VideoWriter with available codecs."""
        for codec in self.SUPPORTED_CODECS:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    str(self.output_path),
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                )
                if writer.isOpened():
                    self.writer = writer
                    logger.info(f"Initialized VideoWriter with codec '{codec}' -> {self.output_path.name}")
                    return
            except Exception as e:
                logger.debug(f"Codec '{codec}' failed: {e}")

        raise RuntimeError(
            f"Failed to initialize VideoWriter for '{self.output_path}'. "
            f"Tested codecs: {self.SUPPORTED_CODECS}"
        )

    def write(self, frame: np.ndarray) -> None:
        """Writes a single BGR frame to the video."""
        if self.writer is None or not self.writer.isOpened():
            raise RuntimeError("VideoWriter is not open.")

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        self.writer.write(frame)

    def release(self) -> None:
        """Releases the video writer and optimizes for browser HTML5 playback."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self._transcode_to_h264_if_needed()
            logger.info(f"Saved annotated video to {self.output_path}")

    def _transcode_to_h264_if_needed(self) -> None:
        """Transcode video to browser-native H.264 (yuv420p) with faststart moov atom."""
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return

        try:
            import imageio_ffmpeg
            import subprocess
            import shutil

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_path = self.output_path.with_name(f"temp_h264_{self.output_path.name}")

            cmd = [
                ffmpeg_exe,
                "-y",
                "-i", str(self.output_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(temp_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
                shutil.move(temp_path, self.output_path)
                logger.info(f"Transcoded '{self.output_path.name}' to HTML5 browser-compatible H.264 (avc1)")
            elif temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.debug(f"H.264 transcoding skipped: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
