"""Preprocessing module for RailGuard.

Handles image resizing, adaptive contrast/brightness normalisation,
and optional optical flow video stabilisation for drone footage.
"""

from collections import deque
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("rail-guard.preprocess")


class VideoStabilizer:
    """Stabilizes video frames from moving drone or handheld cameras using feature tracking."""

    def __init__(self, smoothing_radius: int = 15):
        self.smoothing_radius = smoothing_radius
        self.prev_gray: Optional[np.ndarray] = None
        self.transforms: deque = deque(maxlen=smoothing_radius * 2 + 1)
        # Cumulative trajectory [dx, dy, da]
        self.trajectory_x = 0.0
        self.trajectory_y = 0.0
        self.trajectory_a = 0.0
        self.history = []

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Stabilize the incoming frame relative to previous frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        # Detect corner features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3,
        )

        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = gray
            return frame

        # Track features to current frame via Lucas-Kanade optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)

        valid_prev = prev_pts[status == 1]
        valid_curr = curr_pts[status == 1]

        if len(valid_prev) < 10:
            self.prev_gray = gray
            return frame

        # Estimate rigid affine transformation (translation + rotation)
        transform, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
        self.prev_gray = gray

        if transform is None:
            return frame

        # Extract translation dx, dy and rotation da
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])

        self.trajectory_x += dx
        self.trajectory_y += dy
        self.trajectory_a += da
        self.history.append((self.trajectory_x, self.trajectory_y, self.trajectory_a))

        # Compute smoothed trajectory via moving window
        window = self.history[-min(len(self.history), self.smoothing_radius):]
        smooth_x = np.mean([p[0] for p in window])
        smooth_y = np.mean([p[1] for p in window])
        smooth_a = np.mean([p[2] for p in window])

        # Difference between smooth trajectory and current trajectory
        diff_x = smooth_x - self.trajectory_x
        diff_y = smooth_y - self.trajectory_y
        diff_a = smooth_a - self.trajectory_a

        # Adjust the current transform matrix to compensate high-frequency camera shake
        comp_dx = dx + diff_x
        comp_dy = dy + diff_y
        comp_da = da + diff_a

        comp_transform = np.zeros((2, 3), dtype=np.float32)
        comp_transform[0, 0] = np.cos(comp_da)
        comp_transform[0, 1] = -np.sin(comp_da)
        comp_transform[1, 0] = np.sin(comp_da)
        comp_transform[1, 1] = np.cos(comp_da)
        comp_transform[0, 2] = comp_dx
        comp_transform[1, 2] = comp_dy

        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame,
            comp_transform,
            (w, h),
            borderMode=cv2.BORDER_REFLECT,
        )
        return stabilized


class FramePreprocessor:
    """Preprocesses video frames with adaptive lighting enhancement and optional stabilization."""

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = (1280, 720),
        normalize_contrast: bool = True,
        enable_stabilization: bool = False,
    ):
        self.target_size = target_size
        self.normalize_contrast = normalize_contrast
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if normalize_contrast else None
        self.stabilizer = VideoStabilizer() if enable_stabilization else None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply the preprocessing pipeline to a frame."""
        processed = frame.copy()

        # 1. Resize if target dimension specified
        if self.target_size is not None:
            w, h = self.target_size
            if processed.shape[1] != w or processed.shape[0] != h:
                processed = cv2.resize(processed, (w, h), interpolation=cv2.INTER_LINEAR)

        # 2. Drone video stabilization
        if self.stabilizer is not None:
            processed = self.stabilizer.process(processed)

        # 3. CLAHE Contrast & Brightness normalization (LAB color space)
        if self.normalize_contrast and self.clahe is not None:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self.clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return processed
