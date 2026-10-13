"""Configuration module for RailGuard.

Defines default thresholds, detection classes, paths, ROI parameters,
and severity level mappings.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Setup unified logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rail-guard")


# Default class taxonomy
DEFECT_CLASSES: Set[str] = {
    "crack",
    "broken_rail",
    "missing_clip",
    "sleeper_damage",
}

OBSTACLE_CLASSES: Set[str] = {
    "person",
    "cow",
    "vehicle",
    "debris",
    # Pretrained COCO fallbacks for broader compatibility
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "horse",
    "sheep",
    "elephant",
    "bear",
}

# Base defect severity defaults (can be overridden by ROI position or confidence)
DEFAULT_DEFECT_SEVERITY: Dict[str, str] = {
    "broken_rail": "HIGH",
    "crack": "HIGH",
    "sleeper_damage": "MEDIUM",
    "missing_clip": "LOW",
}


@dataclass
class RailGuardConfig:
    """Central configuration dataclass for the RailGuard pipeline."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    model_path: str = "models/best.pt"
    output_dir: str = "outputs"
    data_dir: str = "data/videos"

    # Hardware & Model
    device: str = "cpu"  # 'cpu', 'cuda', 'mps'
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    frame_skip: int = 1  # 1 = process every frame, 2 = every 2nd frame, etc.

    # Preprocessing
    resize_width: int = 1280
    resize_height: int = 720
    normalize_contrast: bool = True
    stabilize_drone_video: bool = False

    # Track ROI Detection
    roi_canny_low: int = 50
    roi_canny_high: int = 150
    roi_hough_threshold: int = 50
    roi_min_line_len: int = 60
    roi_max_line_gap: int = 20
    roi_smoothing_alpha: float = 0.25  # EMA weight for new frame ROI (0.0=frozen, 1.0=no smoothing)

    # Fixed fallback trapezoid (normalized coords [0..1]: (top_left_x, top_y, top_right_x, top_y, bottom_right_x, bot_y, bottom_left_x, bot_y))
    # Perspective typical for rail onboard camera:
    fallback_roi_normalized: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.38, 0.45),  # Top-left (horizon)
        (0.62, 0.45),  # Top-right (horizon)
        (0.92, 0.95),  # Bottom-right (near train)
        (0.08, 0.95),  # Bottom-left (near train)
    )

    # Danger/Severity Y-axis splits inside ROI (relative depth from top to bottom)
    # 0.0 is top of ROI (far away), 1.0 is bottom of ROI (right in front of train)
    far_zone_ratio: float = 0.35   # 0.00 to 0.35 -> LOW severity
    mid_zone_ratio: float = 0.70   # 0.35 to 0.70 -> MEDIUM severity
    # > 0.70 -> HIGH severity (imminent danger)

    # Tracking
    tracker_type: str = "bytetrack.yaml"
    track_buffer: int = 30

    def resolve_path(self, rel_or_abs_path: str) -> Path:
        """Resolve a path relative to project root if not absolute."""
        p = Path(rel_or_abs_path)
        if not p.is_absolute():
            return (self.project_root / p).resolve()
        return p.resolve()

    def get_output_dir_path(self) -> Path:
        """Get output directory as absolute Path and ensure it exists."""
        p = self.resolve_path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
