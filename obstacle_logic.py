"""Obstacle logic and danger severity evaluation for RailGuard.

Determines whether detected obstacles intersect the track ROI, assesses
collision risk based on perspective depth zones (HIGH, MEDIUM, LOW), and
distinguishes defects from obstacles.
"""

from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple

import numpy as np

from src.config import DEFAULT_DEFECT_SEVERITY, RailGuardConfig
from src.detector import Detection
from src.track_roi import TrackROIDetector

logger = logging.getLogger("rail-guard.obstacle_logic")


@dataclass
class EvaluatedObject:
    """An object evaluated for track presence and risk level."""
    detection: Detection
    in_roi: bool
    severity: str  # 'HIGH', 'MEDIUM', 'LOW', 'NONE'
    alert_triggered: bool
    alert_message: str


@dataclass
class FrameSafetyAssessment:
    """Consolidated safety assessment for a processed frame."""
    has_alert: bool
    highest_severity: str  # 'HIGH', 'MEDIUM', 'LOW', 'CLEAR'
    active_alerts: List[str]
    evaluated_objects: List[EvaluatedObject]
    defects_count: int
    obstacles_in_roi_count: int


class ObstacleEvaluator:
    """Evaluates spatial overlap between detections and the track corridor."""

    def __init__(self, config: Optional[RailGuardConfig] = None):
        self.config = config or RailGuardConfig()
        self.roi_util = TrackROIDetector(config=self.config)

    def evaluate_frame(
        self,
        detections: List[Detection],
        roi_poly: np.ndarray,
    ) -> FrameSafetyAssessment:
        """Evaluates all detections in a frame against the track ROI trapezoid.

        Args:
            detections: List of Detection objects from the detector/tracker.
            roi_poly: (4, 2) array of track ROI trapezoid vertices.

        Returns:
            FrameSafetyAssessment: Summary of alerts, risk severities, and object evaluations.
        """
        evaluated_objects: List[EvaluatedObject] = []
        active_alerts: List[str] = []
        severity_ranks = {"CLEAR": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        highest_rank = 0

        defects_count = 0
        obstacles_in_roi_count = 0

        for det in detections:
            is_defect = (det.category == "defect")

            if is_defect:
                defects_count += 1
                # Defects are inherently on/near the track; calculate severity from class taxonomy
                base_severity = DEFAULT_DEFECT_SEVERITY.get(det.cls_name.lower(), "MEDIUM")
                # Check if defect is inside ROI
                in_roi, _ = self.roi_util.check_bbox_overlap(det.box, roi_poly)
                severity = base_severity if in_roi else "LOW"
                alert_trig = False  # Defects generate defect logs rather than immediate collision emergency sirens
                msg = f"Track defect: {det.cls_name} ({severity})"

                evaluated_objects.append(
                    EvaluatedObject(
                        detection=det,
                        in_roi=in_roi,
                        severity=severity,
                        alert_triggered=alert_trig,
                        alert_message=msg,
                    )
                )

            else:
                # Obstacle evaluation (person, cow, vehicle, debris)
                in_roi, _ = self.roi_util.check_bbox_overlap(det.box, roi_poly)

                if in_roi:
                    obstacles_in_roi_count += 1
                    # Estimate depth zone (HIGH=near, MEDIUM=mid, LOW=far)
                    severity = self.roi_util.estimate_danger_severity(det.box, roi_poly)
                    alert_trig = severity in ("HIGH", "MEDIUM")
                    msg = f"OBSTACLE ON TRACK: {det.cls_name.upper()} [{severity} RISK]"

                    if alert_trig:
                        active_alerts.append(msg)

                    rank = severity_ranks.get(severity, 0)
                    if rank > highest_rank:
                        highest_rank = rank

                    evaluated_objects.append(
                        EvaluatedObject(
                            detection=det,
                            in_roi=True,
                            severity=severity,
                            alert_triggered=alert_trig,
                            alert_message=msg,
                        )
                    )
                else:
                    # Obstacle is outside the rail clearance envelope
                    evaluated_objects.append(
                        EvaluatedObject(
                            detection=det,
                            in_roi=False,
                            severity="NONE",
                            alert_triggered=False,
                            alert_message=f"{det.cls_name} (Clear of rails)",
                        )
                    )

        inv_ranks = {0: "CLEAR", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
        highest_severity_str = inv_ranks.get(highest_rank, "CLEAR")

        return FrameSafetyAssessment(
            has_alert=len(active_alerts) > 0,
            highest_severity=highest_severity_str,
            active_alerts=active_alerts,
            evaluated_objects=evaluated_objects,
            defects_count=defects_count,
            obstacles_in_roi_count=obstacles_in_roi_count,
        )
