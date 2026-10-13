"""Detector module for RailGuard.

Provides a robust YOLOv8 inference wrapper with support for custom weights,
device selection (CPU/GPU), class extraction, and graceful fallbacks.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from src.config import DEFECT_CLASSES, OBSTACLE_CLASSES, RailGuardConfig

logger = logging.getLogger("rail-guard.detector")


@dataclass
class Detection:
    """Represents a single detected object in a frame."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf: float
    cls_id: int
    cls_name: str
    category: str  # 'defect' or 'obstacle'
    track_id: Optional[int] = None

    @property
    def x1(self) -> int:
        return self.box[0]

    @property
    def y1(self) -> int:
        return self.box[1]

    @property
    def x2(self) -> int:
        return self.box[2]

    @property
    def y2(self) -> int:
        return self.box[3]

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Ground contact point for perspective positioning."""
        return ((self.x1 + self.x2) / 2.0, float(self.y2))


class YOLOv8Detector:
    """Ultralytics YOLOv8 inference wrapper for rail defect & obstacle detection."""

    def __init__(
        self,
        model_path: str | Path = "models/best.pt",
        device: str = "cpu",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        config: Optional[RailGuardConfig] = None,
    ):
        self.config = config or RailGuardConfig()
        self.model_path = Path(model_path)
        self.device = str(device).lower()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.model = None
        self.class_names: Dict[int, str] = {}
        self.defect_classes: Set[str] = set(DEFECT_CLASSES)
        self.obstacle_classes: Set[str] = set(OBSTACLE_CLASSES)

        self._load_model()

    def _load_model(self) -> None:
        """Loads YOLOv8 model weights with graceful fallback."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Ultralytics package is not installed. Run 'pip install ultralytics'.")
            self.model = None
            return

        # Attempt to load custom weights
        resolved_path = self.model_path
        if not resolved_path.is_absolute() and self.config:
            resolved_path = self.config.resolve_path(str(self.model_path))

        if resolved_path.exists():
            try:
                logger.info(f"Loading custom weights from {resolved_path} onto device '{self.device}'...")
                self.model = YOLO(str(resolved_path))
                logger.info(f"Successfully loaded custom model: {resolved_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load custom weights from {resolved_path}: {e}")
                self.model = None

        if self.model is None:
            # Fallback to yolov8n.pt
            fallback_name = "yolov8n.pt"
            logger.warning(
                f"Specified model '{self.model_path}' not found or failed to load. "
                f"Falling back to base pretrained model '{fallback_name}'."
            )
            try:
                self.model = YOLO(fallback_name)
                logger.info(f"Loaded fallback YOLO model: {fallback_name}")
            except Exception as e:
                logger.error(f"Failed to load fallback model {fallback_name}: {e}")
                self.model = None

        # Extract class taxonomy from model
        if self.model is not None and hasattr(self.model, "names") and self.model.names:
            self.class_names = self.model.names
            logger.info(f"Model classes recognized ({len(self.class_names)} classes): {list(self.class_names.values())[:10]}...")

            # Merge any recognized defect and obstacle classes from the model
            for _, name in self.class_names.items():
                name_clean = name.lower().replace(" ", "_").replace("-", "_")
                if any(defect_term in name_clean for defect_term in ["crack", "broken", "clip", "sleeper", "defect", "flaw"]):
                    self.defect_classes.add(name)
                elif any(obs_term in name_clean for obs_term in ["person", "cow", "vehicle", "debris", "car", "truck", "animal"]):
                    self.obstacle_classes.add(name)

    def categorize_class(self, cls_name: str) -> str:
        """Determines if a class is a defect or an obstacle."""
        name_lower = cls_name.lower().replace(" ", "_")
        if name_lower in {c.lower() for c in self.defect_classes}:
            return "defect"
        return "obstacle"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Runs inference on a single frame and returns structured detections.

        Args:
            frame: BGR numpy image array.

        Returns:
            List[Detection]: Detected objects with bounding box, score, and class info.
        """
        if self.model is None:
            # Fallback heuristic / demo detector if ultralytics model failed to load
            return self._heuristic_fallback_detect(frame)

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Inference error on device '{self.device}': {e}")
            return []

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = map(int, boxes_xyxy[i])
            conf = float(confs[i])
            cls_id = int(cls_ids[i])
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            category = self.categorize_class(cls_name)

            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    conf=conf,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    category=category,
                )
            )

        return detections

    def _heuristic_fallback_detect(self, frame: np.ndarray) -> List[Detection]:
        """Heuristic vision fallback for defect candidates when model is unavailable."""
        detections: List[Detection] = []
        h, w = frame.shape[:2]

        # Scan lower track zone for sharp discontinuities (e.g. broken rails or cracks)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        track_strip = gray[int(h * 0.6):int(h * 0.9), int(w * 0.2):int(w * 0.8)]
        thresh = cv2.adaptiveThreshold(
            track_strip, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if 300 < area < 2500:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh) if bh > 0 else 1.0
                if aspect > 2.5:  # Transverse crack candidate
                    abs_x = int(w * 0.2) + bx
                    abs_y = int(h * 0.6) + by
                    detections.append(
                        Detection(
                            box=(abs_x, abs_y, abs_x + bw, abs_y + bh),
                            conf=0.72,
                            cls_id=0,
                            cls_name="crack",
                            category="defect",
                        )
                    )
                    break

        return detections
