"""Pipeline module for RailGuard.

Integrates video ingestion, preprocessing, track ROI detection, YOLOv8 inference,
persistent multi-object tracking, obstacle danger evaluation, HUD annotation,
and de-duplicated CSV reporting.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import RailGuardConfig
from src.detector import Detection, YOLOv8Detector
from src.obstacle_logic import FrameSafetyAssessment, ObstacleEvaluator
from src.preprocess import FramePreprocessor
from src.reporter import DefectReporter
from src.track_roi import TrackROIDetector
from src.tracker import ObjectTracker
from src.video_io import VideoReader, VideoWriter

logger = logging.getLogger("rail-guard.pipeline")


@dataclass
class PipelineResult:
    """Contains paths and metrics from a completed pipeline run."""
    video_path: Path
    annotated_video_path: Path
    report_csv_path: Path
    total_frames: int
    processed_frames: int
    fps: float
    duration_sec: float
    statistics: Dict[str, any]
    dataframe: pd.DataFrame
    reporter: Optional[DefectReporter] = None


class RailGuardPipeline:
    """End-to-end railway track inspection and obstacle alert pipeline."""

    def __init__(self, config: Optional[RailGuardConfig] = None):
        self.config = config or RailGuardConfig()
        self.preprocessor = FramePreprocessor(
            target_size=(self.config.resize_width, self.config.resize_height),
            normalize_contrast=self.config.normalize_contrast,
            enable_stabilization=self.config.stabilize_drone_video,
        )
        self.roi_detector = TrackROIDetector(config=self.config)
        self.detector = YOLOv8Detector(
            model_path=self.config.model_path,
            device=self.config.device,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            config=self.config,
        )
        self.tracker = ObjectTracker(iou_threshold=self.config.iou_threshold)
        self.evaluator = ObstacleEvaluator(config=self.config)

    def process_video(
        self,
        video_path: str | Path,
        output_dir: Optional[str | Path] = None,
        gps_csv_path: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[float, int, int, str], None]] = None,
    ) -> PipelineResult:
        """Runs complete detection, tracking, safety assessment, and rendering on a video file.

        Args:
            video_path: Path to input video.
            output_dir: Destination directory for output video and CSV report.
            gps_csv_path: Optional CSV with GPS points for coordinate mapping.
            progress_callback: Optional callback fn(progress, current_frame, total_frames, status_msg).

        Returns:
            PipelineResult: Complete metadata, stats, and output paths.
        """
        video_path = Path(video_path).resolve()
        out_dir = Path(output_dir).resolve() if output_dir else self.config.get_output_dir_path()
        out_dir.mkdir(parents=True, exist_ok=True)

        video_stem = video_path.stem
        annotated_video_path = out_dir / f"annotated_{video_stem}.mp4"
        report_csv_path = out_dir / f"report_{video_stem}.csv"

        reporter = DefectReporter(gps_csv_path=gps_csv_path)

        with VideoReader(video_path) as reader:
            total_frames = reader.total_frames
            source_fps = reader.fps
            effective_fps = source_fps / max(1, self.config.frame_skip)

            writer = VideoWriter(
                output_path=annotated_video_path,
                width=self.config.resize_width,
                height=self.config.resize_height,
                fps=effective_fps,
            )

            processed_count = 0
            start_wall_time = time.time()
            prev_frame_time = time.time()
            fps_smoothed = effective_fps

            try:
                for frame_idx, timestamp_sec, raw_frame in reader.frames(frame_skip=self.config.frame_skip):
                    t0 = time.time()

                    # 1. Preprocess frame
                    frame = self.preprocessor.preprocess(raw_frame)

                    # 2. Detect & smooth track ROI
                    roi_poly = self.roi_detector.detect_roi(frame)

                    # 3. YOLOv8 Detection
                    raw_detections = self.detector.detect(frame)

                    # 4. Multi-object tracking (assign persistent track_id)
                    tracked_detections = self.tracker.update_with_detections(raw_detections, frame_idx)

                    # 5. Obstacle overlap & severity logic
                    assessment = self.evaluator.evaluate_frame(tracked_detections, roi_poly)

                    # 6. De-duplicated reporting
                    reporter.log_frame(frame_idx, timestamp_sec, assessment.evaluated_objects, total_frames=total_frames)

                    # Calculate instantaneous FPS
                    dt = time.time() - t0
                    inst_fps = 1.0 / dt if dt > 0 else 30.0
                    fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps

                    # 7. Render annotations and HUD onto frame
                    annotated_frame = self._render_annotations(
                        frame=frame,
                        roi_poly=roi_poly,
                        assessment=assessment,
                        frame_idx=frame_idx,
                        total_frames=total_frames,
                        timestamp_sec=timestamp_sec,
                        fps=fps_smoothed,
                        reporter=reporter,
                    )

                    # 8. Write to output video
                    writer.write(annotated_frame)
                    processed_count += 1

                    # 9. Report progress
                    if progress_callback:
                        progress = min(1.0, float(frame_idx + 1) / max(1, total_frames))
                        status = (
                            f"Frame {frame_idx + 1}/{total_frames} | "
                            f"Alerts: {len(assessment.active_alerts)} | "
                            f"{fps_smoothed:.1f} FPS"
                        )
                        progress_callback(progress, frame_idx + 1, total_frames, status)

            finally:
                writer.release()

        # Save CSV report
        reporter.save_csv(report_csv_path)

        total_wall_time = max(0.001, time.time() - start_wall_time)
        avg_fps = processed_count / total_wall_time

        logger.info(
            f"Finished processing video '{video_path.name}'. Processed {processed_count} frames "
            f"in {total_wall_time:.1f}s ({avg_fps:.1f} FPS avg)."
        )

        return PipelineResult(
            video_path=video_path,
            annotated_video_path=annotated_video_path,
            report_csv_path=report_csv_path,
            total_frames=total_frames,
            processed_frames=processed_count,
            fps=avg_fps,
            duration_sec=total_wall_time,
            statistics=reporter.get_statistics(),
            dataframe=reporter.get_summary_dataframe(),
            reporter=reporter,
        )

    def _render_annotations(
        self,
        frame: np.ndarray,
        roi_poly: np.ndarray,
        assessment: FrameSafetyAssessment,
        frame_idx: int,
        total_frames: int,
        timestamp_sec: float,
        fps: float,
        reporter: DefectReporter,
    ) -> np.ndarray:
        """Renders ROI polygon, bounding boxes, labels, HUD, and alert banners onto the frame."""
        h, w = frame.shape[:2]

        # 1. Draw ROI Corridor
        frame = self.roi_detector.draw_roi_overlay(frame, roi_poly, alert_active=assessment.has_alert)

        # 2. Draw Bounding Boxes and Labels for Evaluated Objects
        for obj in assessment.evaluated_objects:
            det = obj.detection
            x1, y1, x2, y2 = det.box
            tid = det.track_id or 0

            # Color scheme
            if det.category == "defect":
                # Defects: Vibrant Cyan / Magenta
                box_color = (255, 200, 0) if obj.severity == "HIGH" else (220, 160, 40)
                badge_text = f"DEFECT #{tid} {det.cls_name.upper()} ({det.conf:.2f}) [{obj.severity}]"
            else:
                # Obstacles: Red for HIGH danger, Orange for MEDIUM, Yellow for LOW, Gray for outside ROI
                if not obj.in_roi:
                    box_color = (160, 160, 160)
                    badge_text = f"#{tid} {det.cls_name} ({det.conf:.2f}) [CLEAR]"
                elif obj.severity == "HIGH":
                    box_color = (0, 0, 240)
                    badge_text = f"🚨 #{tid} {det.cls_name.upper()} ({det.conf:.2f}) [HIGH DANGER]"
                elif obj.severity == "MEDIUM":
                    box_color = (0, 140, 255)
                    badge_text = f"⚠️ #{tid} {det.cls_name.upper()} ({det.conf:.2f}) [MEDIUM]"
                else:
                    box_color = (0, 220, 255)
                    badge_text = f"#{tid} {det.cls_name} ({det.conf:.2f}) [LOW]"

            # Draw box
            thickness = 3 if (obj.in_roi and obj.severity in ("HIGH", "MEDIUM")) else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            # Draw corner accents for high risk
            if obj.in_roi and obj.severity == "HIGH":
                corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
                accent_col = (0, 0, 255)
                # Top-left corner
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), accent_col, 4)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), accent_col, 4)
                # Bottom-right corner
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), accent_col, 4)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), accent_col, 4)

            # Draw Label Tag Header
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, y1 - text_h - 8)
            tag_y2 = y1
            cv2.rectangle(frame, (x1, tag_y1), (x1 + text_w + 10, tag_y2), box_color, -1)
            cv2.putText(
                frame,
                badge_text,
                (x1 + 5, tag_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0) if box_color[1] > 180 else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Top-Left HUD Telemetry Card
        hud_w, hud_h = 360, 115
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (15, 15), (15 + hud_w, 15 + hud_h), (20, 24, 30), -1)
        cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (15, 15), (15 + hud_w, 15 + hud_h), (60, 80, 100), 1)

        # Title
        cv2.putText(frame, "RAILGUARD v1.0 | TRACK SAFETY", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        # Stats rows
        stats = reporter.get_statistics()
        mins = int(timestamp_sec // 60)
        secs = timestamp_sec % 60
        time_str = f"{mins:02d}:{secs:04.1f}"

        row1 = f"Frame: {frame_idx + 1}/{total_frames}  |  Time: {time_str}  |  FPS: {fps:.1f}"
        row2 = f"Unique Defects: {stats['total_defects']}  |  Obstacles in ROI: {stats['total_obstacles_in_roi']}"

        # Status badge
        if assessment.has_alert:
            status_txt = f"STATUS: {assessment.highest_severity} RISK DETECTED"
            status_col = (0, 0, 255) if assessment.highest_severity == "HIGH" else (0, 140, 255)
        else:
            status_txt = "STATUS: TRACK CLEAR"
            status_col = (40, 220, 40)

        cv2.putText(frame, row1, (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, row2, (25, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, status_txt, (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_col, 2, cv2.LINE_AA)

        # 4. Top-Center Alert Banner (Visible when obstacle is inside ROI)
        if assessment.has_alert:
            banner_w = min(720, w - 100)
            bx1 = (w - banner_w) // 2
            bx2 = bx1 + banner_w
            by1, by2 = 18, 62

            # Flashing intensity based on current frame number
            flash = (frame_idx // 6) % 2 == 0
            banner_bg = (0, 0, 220) if (flash or assessment.highest_severity == "HIGH") else (0, 110, 220)

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), banner_bg, -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)

            alert_msg = assessment.active_alerts[0] if assessment.active_alerts else "WARNING: OBSTACLE ON TRACK"
            (msg_w, msg_h), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            msg_x = bx1 + (banner_w - msg_w) // 2
            msg_y = by1 + (by2 - by1 + msg_h) // 2

            cv2.putText(frame, alert_msg, (msg_x, msg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
