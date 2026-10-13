"""Track ROI module for RailGuard.

Detects left and right rails using Canny edge detection and Hough line transform,
constructs a perspective trapezoid Region of Interest (ROI), applies temporal
smoothing across frames, and handles fallbacks.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import RailGuardConfig

logger = logging.getLogger("rail-guard.track_roi")


class TrackROIDetector:
    """Detects railway rails and maintains a smoothed trapezoidal Region of Interest."""

    def __init__(self, config: Optional[RailGuardConfig] = None):
        self.config = config or RailGuardConfig()
        self.smoothed_roi: Optional[np.ndarray] = None  # Shape (4, 2) float32
        self.consecutive_failures: int = 0

    def detect_roi(self, frame: np.ndarray) -> np.ndarray:
        """Detect the track ROI trapezoid from a frame.

        Returns:
            np.ndarray: Polygon vertices of shape (4, 2) with dtype int32:
                        [top_left, top_right, bottom_right, bottom_left]
        """
        h, w = frame.shape[:2]
        detected_poly = self._detect_rails_hough(frame)

        if detected_poly is None:
            self.consecutive_failures += 1
            fallback_poly = self._get_fallback_trapezoid(w, h)
            target_poly = fallback_poly
        else:
            self.consecutive_failures = 0
            target_poly = detected_poly

        # Apply temporal smoothing across frames
        if self.smoothed_roi is None:
            self.smoothed_roi = target_poly.astype(np.float32)
        else:
            alpha = self.config.roi_smoothing_alpha
            self.smoothed_roi = alpha * target_poly.astype(np.float32) + (1.0 - alpha) * self.smoothed_roi

        return np.round(self.smoothed_roi).astype(np.int32)

    def _detect_rails_hough(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Attempt to find left and right rails via Canny edges + Hough lines."""
        h, w = frame.shape[:2]

        # Restrict edge search to lower ~60% of frame where rails reside
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        mask_poly = np.array([
            [int(w * 0.1), int(h * 0.40)],
            [int(w * 0.9), int(h * 0.40)],
            [w, h],
            [0, h]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [mask_poly], 255)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.config.roi_canny_low, self.config.roi_canny_high)
        masked_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.roi_hough_threshold,
            minLineLength=self.config.roi_min_line_len,
            maxLineGap=self.config.roi_max_line_gap,
        )

        if lines is None or len(lines) < 2:
            return None

        left_lines = []
        right_lines = []

        center_x = w / 2.0

        for line in lines:
            line_flat = np.asarray(line).ravel()
            if len(line_flat) < 4:
                continue
            x1, y1, x2, y2 = int(line_flat[0]), int(line_flat[1]), int(line_flat[2]), int(line_flat[3])
            if x2 == x1:
                slope = 999.0
            else:
                slope = (y2 - y1) / (x2 - x1)

            # Filter out almost-horizontal lines
            if abs(slope) < 0.4 or abs(slope) > 10.0:
                continue

            # In image coords, left rail slopes down-left to up-right (negative dy/dx if going up,
            # or positive if looking at standard image plane: left rail has negative slope (x increases as y decreases))
            mid_x = (x1 + x2) / 2.0
            length = np.hypot(x2 - x1, y2 - y1)

            # Left rail candidates: located mostly on left half with negative slope
            if slope < -0.4 and mid_x < center_x + (w * 0.15):
                left_lines.append((x1, y1, x2, y2, length))
            # Right rail candidates: located mostly on right half with positive slope
            elif slope > 0.4 and mid_x > center_x - (w * 0.15):
                right_lines.append((x1, y1, x2, y2, length))

        if not left_lines or not right_lines:
            return None

        # Fit representative lines for left and right rails
        left_line = self._fit_representative_line(left_lines, h)
        right_line = self._fit_representative_line(right_lines, h)

        if left_line is None or right_line is None:
            return None

        (lx_top, ly_top), (lx_bot, ly_bot) = left_line
        (rx_top, ry_top), (rx_bot, ry_bot) = right_line

        # Ensure left rail is to the left of right rail
        if lx_top >= rx_top or lx_bot >= rx_bot:
            return None

        # Sanity check widths
        top_w = rx_top - lx_top
        bot_w = rx_bot - lx_bot
        if bot_w <= top_w or bot_w < w * 0.15 or top_w < w * 0.05:
            return None

        # Construct trapezoid: [top-left, top-right, bottom-right, bottom-left]
        poly = np.array([
            [lx_top, ly_top],
            [rx_top, ry_top],
            [rx_bot, ry_bot],
            [lx_bot, ly_bot]
        ], dtype=np.int32)

        return poly

    def _fit_representative_line(self, lines, frame_h: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Fits a single line through weighted line segments."""
        pts = []
        weights = []
        for x1, y1, x2, y2, length in lines:
            pts.append([x1, y1])
            pts.append([x2, y2])
            weights.extend([length, length])

        pts = np.array(pts, dtype=np.float32)
        if len(pts) < 2:
            return None

        # Fit line: [vx, vy, x0, y0]
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])

        if abs(vy) < 1e-4:
            return None

        y_top = int(frame_h * 0.45)
        y_bot = int(frame_h * 0.95)

        x_top = int(x0 + (y_top - y0) * (vx / vy))
        x_bot = int(x0 + (y_bot - y0) * (vx / vy))

        return (x_top, y_top), (x_bot, y_bot)

    def _get_fallback_trapezoid(self, width: int, height: int) -> np.ndarray:
        """Constructs fallback trapezoid using normalized configuration coordinates."""
        coords = self.config.fallback_roi_normalized
        poly = np.array([
            [int(coords[0][0] * width), int(coords[0][1] * height)],
            [int(coords[1][0] * width), int(coords[1][1] * height)],
            [int(coords[2][0] * width), int(coords[2][1] * height)],
            [int(coords[3][0] * width), int(coords[3][1] * height)],
        ], dtype=np.int32)
        return poly

    def check_point_in_roi(self, x: float, y: float, roi_poly: np.ndarray) -> bool:
        """Check if an (x, y) point lies inside the track ROI polygon."""
        dist = cv2.pointPolygonTest(roi_poly, (float(x), float(y)), False)
        return dist >= 0

    def check_bbox_overlap(self, bbox: Tuple[int, int, int, int], roi_poly: np.ndarray) -> Tuple[bool, float]:
        """Check if a bounding box overlaps the track ROI.

        Uses the bottom-center footpoint (ground contact) and bottom corners,
        plus box area intersection.

        Args:
            bbox: (x1, y1, x2, y2)
            roi_poly: Track ROI polygon (4, 2)

        Returns:
            Tuple[bool, float]: (overlaps, overlap_score)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        foot_point = (cx, float(y2))

        # Check if bottom-center footpoint is inside ROI
        if self.check_point_in_roi(foot_point[0], foot_point[1], roi_poly):
            return True, 1.0

        # Check bottom corners
        if self.check_point_in_roi(x1, y2, roi_poly) or self.check_point_in_roi(x2, y2, roi_poly):
            return True, 0.8

        # Check center of box
        cy = (y1 + y2) / 2.0
        if self.check_point_in_roi(cx, cy, roi_poly):
            return True, 0.7

        # Check polygon intersection via bounding mask
        h_max = max(roi_poly[:, 1].max(), y2) + 10
        w_max = max(roi_poly[:, 0].max(), x2) + 10

        mask_box = np.zeros((h_max, w_max), dtype=np.uint8)
        mask_roi = np.zeros((h_max, w_max), dtype=np.uint8)

        cv2.rectangle(mask_box, (x1, y1), (x2, y2), 255, -1)
        cv2.fillPoly(mask_roi, [roi_poly], 255)

        inter = cv2.bitwise_and(mask_box, mask_roi)
        inter_area = np.count_nonzero(inter)
        box_area = max(1, (x2 - x1) * (y2 - y1))
        overlap_ratio = inter_area / float(box_area)

        return (overlap_ratio > 0.15), overlap_ratio

    def estimate_danger_severity(self, bbox: Tuple[int, int, int, int], roi_poly: np.ndarray) -> str:
        """Estimate danger severity (HIGH, MEDIUM, LOW) based on perspective depth inside the ROI.

        Near zone (bottom fraction) -> HIGH severity
        Middle zone -> MEDIUM severity
        Far zone (near horizon) -> LOW severity
        """
        _, _, _, y2 = bbox
        y_min = float(np.min(roi_poly[:, 1]))
        y_max = float(np.max(roi_poly[:, 1]))

        if y_max <= y_min:
            return "MEDIUM"

        # Normalized depth (0.0 = horizon / far away, 1.0 = near train)
        depth = (float(y2) - y_min) / (y_max - y_min)
        depth = np.clip(depth, 0.0, 1.0)

        if depth <= self.config.far_zone_ratio:
            return "LOW"
        elif depth <= self.config.mid_zone_ratio:
            return "MEDIUM"
        else:
            return "HIGH"

    def draw_roi_overlay(self, frame: np.ndarray, roi_poly: np.ndarray, alert_active: bool = False) -> np.ndarray:
        """Draws semi-transparent colored track corridor with zone dividers on the frame."""
        overlay = frame.copy()
        color = (0, 0, 220) if alert_active else (40, 200, 40)  # Red if alert, Green if clear

        cv2.fillPoly(overlay, [roi_poly], color)
        # 25% opacity
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        # Draw corridor boundary lines
        cv2.polylines(frame, [roi_poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Draw depth zone dividers
        y_min = float(np.min(roi_poly[:, 1]))
        y_max = float(np.max(roi_poly[:, 1]))
        h_span = y_max - y_min

        # Far/Mid divider
        y_div1 = int(y_min + self.config.far_zone_ratio * h_span)
        # Mid/Near divider
        y_div2 = int(y_min + self.config.mid_zone_ratio * h_span)

        for y_div, label in [(y_div1, "FAR | LOW RISK"), (y_div2, "MID | MEDIUM RISK")]:
            cv2.line(frame, (roi_poly[3][0], y_div), (roi_poly[2][0], y_div), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                label,
                (roi_poly[3][0] + 10, y_div - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return frame
