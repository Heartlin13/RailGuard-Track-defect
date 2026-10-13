"""Reporter and CSV logging module for RailGuard.

Handles de-duplicated defect logging using persistent track IDs,
optional GPS interpolation, and summary analytics for dashboard consumption.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.obstacle_logic import EvaluatedObject

logger = logging.getLogger("rail-guard.reporter")


@dataclass
class LogRecord:
    """Record representing a detected defect or track obstacle."""
    frame: int
    timestamp: float
    track_id: int
    category: str
    class_name: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    severity: str
    in_roi: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class GPSLookup:
    """Interpolates GPS coordinates from a CSV file or DataFrame based on video timestamp or frame."""

    def __init__(self, gps_source: Optional[str | Path | pd.DataFrame] = None):
        self.df: Optional[pd.DataFrame] = None
        self.lat_col: Optional[str] = None
        self.lon_col: Optional[str] = None
        if gps_source is not None:
            self.load(gps_source)

    def load(self, source: str | Path | pd.DataFrame) -> bool:
        """Load and normalize GPS data from a file path or DataFrame."""
        try:
            if isinstance(source, pd.DataFrame):
                df = source.copy()
            else:
                p = Path(source)
                if not p.exists() or p.stat().st_size == 0:
                    logger.warning(f"GPS file does not exist or is empty: {p}")
                    return False
                df = pd.read_csv(p)

            if df.empty:
                logger.warning("GPS dataset is empty.")
                return False

            # Normalize column names: lowercase, strip whitespace, remove punctuation
            col_map = {}
            for col in df.columns:
                clean_col = str(col).lower().strip().replace(" ", "_").replace("-", "_")
                col_map[col] = clean_col
            df = df.rename(columns=col_map)

            # Detect latitude column
            lat_candidates = ["latitude", "lat", "gps_latitude", "gps_lat", "lat_dd", "lat_deg", "y"]
            lat_col = next((c for c in lat_candidates if c in df.columns), None)
            if not lat_col:
                lat_col = next((c for c in df.columns if "lat" in c), None)

            # Detect longitude column
            lon_candidates = ["longitude", "lon", "lng", "long", "gps_longitude", "gps_lon", "gps_lng", "lon_dd", "x"]
            lon_col = next((c for c in lon_candidates if c in df.columns), None)
            if not lon_col:
                lon_col = next((c for c in df.columns if any(k in c for k in ["lon", "lng", "long"])), None)

            if not lat_col or not lon_col:
                logger.warning(f"Could not identify latitude/longitude columns in GPS data. Columns found: {list(df.columns)}")
                return False

            # Clean and cast coordinates to float
            df[lat_col] = pd.to_numeric(
                df[lat_col].astype(str).str.replace(r"[^\d.-]", "", regex=True),
                errors="coerce"
            )
            df[lon_col] = pd.to_numeric(
                df[lon_col].astype(str).str.replace(r"[^\d.-]", "", regex=True),
                errors="coerce"
            )

            # Drop rows with invalid coordinates
            valid_df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
            if valid_df.empty:
                logger.warning("No valid numeric GPS coordinates found.")
                return False

            self.df = valid_df
            self.lat_col = lat_col
            self.lon_col = lon_col
            logger.info(f"Successfully loaded {len(self.df)} GPS waypoints (Lat: {lat_col}, Lon: {lon_col})")
            return True

        except Exception as e:
            logger.error(f"Error loading GPS data: {e}", exc_info=True)
            return False

    def get_route_dataframe(self) -> pd.DataFrame:
        """Returns clean DataFrame of all route waypoints with 'lat' and 'lon' columns."""
        if self.df is None or self.df.empty:
            return pd.DataFrame(columns=["lat", "lon"])
        return self.df[[self.lat_col, self.lon_col]].rename(columns={self.lat_col: "lat", self.lon_col: "lon"})

    def get_coordinates(
        self,
        timestamp: float,
        frame_idx: int,
        total_frames: Optional[int] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get (latitude, longitude) for a given timestamp or frame index with interpolation."""
        if self.df is None or self.df.empty:
            return None, None

        n_pts = len(self.df)
        lats = self.df[self.lat_col].values.astype(float)
        lons = self.df[self.lon_col].values.astype(float)

        if n_pts == 1:
            return float(lats[0]), float(lons[0])

        # 1. Match by timestamp/time column if available
        time_col = next((c for c in self.df.columns if any(k in c for k in ["timestamp", "time", "sec", "second"])), None)
        if time_col:
            try:
                times = pd.to_numeric(self.df[time_col], errors="coerce").values.astype(float)
                if not np.isnan(times).any() and (times[-1] > times[0]):
                    lat = float(np.interp(timestamp, times, lats))
                    lon = float(np.interp(timestamp, times, lons))
                    return lat, lon
            except Exception:
                pass

        # 2. Match by frame column if available
        frame_col = next((c for c in self.df.columns if "frame" in c), None)
        if frame_col:
            try:
                frames = pd.to_numeric(self.df[frame_col], errors="coerce").values.astype(float)
                if not np.isnan(frames).any() and (frames[-1] > frames[0]):
                    lat = float(np.interp(frame_idx, frames, lats))
                    lon = float(np.interp(frame_idx, frames, lons))
                    return lat, lon
            except Exception:
                pass

        # 3. Proportional interpolation along track corridor
        if total_frames and total_frames > 1:
            ratio = np.clip(float(frame_idx) / float(total_frames - 1), 0.0, 1.0)
            norm_idx = ratio * (n_pts - 1)
            grid = np.arange(n_pts)
            lat = float(np.interp(norm_idx, grid, lats))
            lon = float(np.interp(norm_idx, grid, lons))
            return lat, lon

        # 4. Nearest row fallback
        idx = min(max(0, int(frame_idx)), n_pts - 1)
        return float(lats[idx]), float(lons[idx])


class DefectReporter:
    """Manages de-duplicated logging of defects and track obstacles."""

    def __init__(self, gps_csv_path: Optional[str | Path] = None):
        self.gps_lookup = GPSLookup(gps_csv_path)
        # Store de-duplicated records keyed by track_id
        self.unique_records: Dict[int, LogRecord] = {}
        # Also store full timeline history
        self.all_frame_events: List[LogRecord] = []

    def log_frame(
        self,
        frame_idx: int,
        timestamp: float,
        evaluated_objects: List[EvaluatedObject],
        total_frames: Optional[int] = None,
    ) -> None:
        """Processes evaluated objects for the frame, updating de-duplicated records.

        Args:
            frame_idx: Video frame index.
            timestamp: Video timestamp in seconds.
            evaluated_objects: Objects evaluated by ObstacleEvaluator.
            total_frames: Optional total frame count for normalized spatial interpolation.
        """
        lat, lon = self.gps_lookup.get_coordinates(timestamp, frame_idx, total_frames=total_frames)

        for obj in evaluated_objects:
            det = obj.detection
            tid = det.track_id or (frame_idx * 1000 + det.cls_id)

            rec = LogRecord(
                frame=frame_idx,
                timestamp=round(timestamp, 3),
                track_id=tid,
                category=det.category,
                class_name=det.cls_name,
                confidence=round(det.conf, 4),
                bbox_x1=det.x1,
                bbox_y1=det.y1,
                bbox_x2=det.x2,
                bbox_y2=det.y2,
                severity=obj.severity,
                in_roi=obj.in_roi,
                latitude=lat,
                longitude=lon,
            )

            self.all_frame_events.append(rec)

            # De-duplicate: Keep record with highest confidence for each track_id
            if tid not in self.unique_records or det.conf > self.unique_records[tid].confidence:
                self.unique_records[tid] = rec

    def attach_gps(self, gps_source: str | Path | pd.DataFrame, total_frames: Optional[int] = None) -> bool:
        """Attach or update GPS coordinates for all existing log records from a new GPS source."""
        success = self.gps_lookup.load(gps_source)
        if not success:
            return False
        for tid, rec in self.unique_records.items():
            lat, lon = self.gps_lookup.get_coordinates(rec.timestamp, rec.frame, total_frames=total_frames)
            rec.latitude = lat
            rec.longitude = lon
        for rec in self.all_frame_events:
            lat, lon = self.gps_lookup.get_coordinates(rec.timestamp, rec.frame, total_frames=total_frames)
            rec.latitude = lat
            rec.longitude = lon
        return True

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Returns de-duplicated log as a Pandas DataFrame."""
        if not self.unique_records:
            return pd.DataFrame(
                columns=[
                    "frame", "timestamp", "track_id", "category", "class_name",
                    "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "severity", "in_roi", "latitude", "longitude"
                ]
            )

        data = [asdict(r) for r in self.unique_records.values()]
        df = pd.DataFrame(data)
        # Sort by timestamp
        df = df.sort_values(by=["timestamp", "frame"]).reset_index(drop=True)
        return df

    def save_csv(self, output_csv_path: str | Path) -> Path:
        """Saves de-duplicated log to CSV file."""
        out_path = Path(output_csv_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.get_summary_dataframe()
        df.to_csv(out_path, index=False)
        logger.info(f"Saved defect log with {len(df)} unique records to {out_path}")
        return out_path

    def get_statistics(self) -> Dict[str, any]:
        """Calculates summary statistics for display."""
        df = self.get_summary_dataframe()
        if df.empty:
            return {
                "total_unique_items": 0,
                "total_defects": 0,
                "total_obstacles_in_roi": 0,
                "high_severity_count": 0,
                "medium_severity_count": 0,
                "low_severity_count": 0,
                "class_counts": {},
                "severity_counts": {},
            }

        defects_df = df[df["category"] == "defect"]
        obstacles_in_roi_df = df[(df["category"] == "obstacle") & (df["in_roi"] == True)]

        return {
            "total_unique_items": len(df),
            "total_defects": len(defects_df),
            "total_obstacles_in_roi": len(obstacles_in_roi_df),
            "high_severity_count": int((df["severity"] == "HIGH").sum()),
            "medium_severity_count": int((df["severity"] == "MEDIUM").sum()),
            "low_severity_count": int((df["severity"] == "LOW").sum()),
            "class_counts": df["class_name"].value_counts().to_dict(),
            "severity_counts": df["severity"].value_counts().to_dict(),
        }
