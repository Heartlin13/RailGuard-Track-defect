"""Tracker module for RailGuard.

Provides persistent multi-object tracking via ByteTrack and IoU association,
ensuring each defect and obstacle maintains a persistent tracking ID across frames.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detector import Detection

logger = logging.getLogger("rail-guard.tracker")


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
    area_b = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / float(union)


@dataclass
class Track:
    """Represents an active or historical object track."""
    track_id: int
    cls_name: str
    category: str
    last_box: Tuple[int, int, int, int]
    last_conf: float
    max_conf: float
    best_box: Tuple[int, int, int, int]
    first_frame: int
    last_frame: int
    time_since_update: int = 0
    hits: int = 1


class ObjectTracker:
    """Assigns persistent IDs to detected objects across frames."""

    def __init__(self, iou_threshold: float = 0.30, max_age: int = 25):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id: int = 1
        self.tracks: Dict[int, Track] = {}

    def update_with_detections(
        self, detections: List[Detection], frame_idx: int
    ) -> List[Detection]:
        """Updates tracks with current frame detections and assigns persistent track_ids.

        Args:
            detections: List of raw detections.
            frame_idx: Current video frame index.

        Returns:
            List[Detection]: Detections with assigned track_id.
        """
        # If model already provided persistent track IDs (e.g. from Ultralytics ByteTrack)
        has_builtin_ids = any(d.track_id is not None for d in detections)
        if has_builtin_ids:
            for d in detections:
                if d.track_id is not None:
                    tid = int(d.track_id)
                    if tid in self.tracks:
                        t = self.tracks[tid]
                        t.last_box = d.box
                        t.last_conf = d.conf
                        t.last_frame = frame_idx
                        t.time_since_update = 0
                        t.hits += 1
                        if d.conf > t.max_conf:
                            t.max_conf = d.conf
                            t.best_box = d.box
                    else:
                        self.tracks[tid] = Track(
                            track_id=tid,
                            cls_name=d.cls_name,
                            category=d.category,
                            last_box=d.box,
                            last_conf=d.conf,
                            max_conf=d.conf,
                            best_box=d.box,
                            first_frame=frame_idx,
                            last_frame=frame_idx,
                        )
            return detections

        # IoU tracking association
        # Age existing tracks
        for t in self.tracks.values():
            t.time_since_update += 1

        active_track_ids = [tid for tid, t in self.tracks.items() if t.time_since_update <= self.max_age]

        unmatched_detections = set(range(len(detections)))
        matched_pairs: List[Tuple[int, int]] = []

        if active_track_ids and detections:
            # Build IoU matrix (len(active_track_ids) x len(detections))
            iou_matrix = np.zeros((len(active_track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(active_track_ids):
                t_box = self.tracks[tid].last_box
                for j, det in enumerate(detections):
                    if det.cls_name == self.tracks[tid].cls_name or det.category == self.tracks[tid].category:
                        iou_matrix[i, j] = compute_iou(t_box, det.box)

            # Greedy matching on highest IoU
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                row, col = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched_pairs.append((active_track_ids[row], col))
                unmatched_detections.discard(col)
                iou_matrix[row, :] = -1.0
                iou_matrix[:, col] = -1.0

        # Update matched tracks
        for tid, det_idx in matched_pairs:
            det = detections[det_idx]
            det.track_id = tid
            track = self.tracks[tid]
            track.last_box = det.box
            track.last_conf = det.conf
            track.last_frame = frame_idx
            track.time_since_update = 0
            track.hits += 1
            if det.conf > track.max_conf:
                track.max_conf = det.conf
                track.best_box = det.box

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            tid = self.next_id
            self.next_id += 1
            det.track_id = tid

            self.tracks[tid] = Track(
                track_id=tid,
                cls_name=det.cls_name,
                category=det.category,
                last_box=det.box,
                last_conf=det.conf,
                max_conf=det.conf,
                best_box=det.box,
                first_frame=frame_idx,
                last_frame=frame_idx,
            )

        # Purge stale tracks exceeding max_age
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.time_since_update <= self.max_age}

        return detections
