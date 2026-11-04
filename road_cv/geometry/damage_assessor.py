"""
Geometry-Based Damage Assessment and Severity Quantification Module
Estimates physical surface dimensions (length, width, area), road-plane ground projections,
and depth anomalies against local baseline pavement without misusing detector confidence.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

class CameraGeometryModel:
    """
    Manages camera extrinsic/intrinsic geometry and road-plane projection.
    Provides ground meters-per-pixel scaling along the longitudinal road plane.
    """

    def __init__(
        self,
        camera_height_m: float = 1.35,      # typical windshield/dashboard mount height
        pitch_angle_deg: float = 8.0,       # downward tilt in degrees
        focal_length_px: float = 1200.0,    # estimated focal length for 1080p/1440p
        principal_point: Tuple[float, float] = (960.0, 540.0),
        is_calibrated: bool = False
    ):
        self.camera_height_m = camera_height_m
        self.pitch_rad = np.radians(pitch_angle_deg)
        self.focal_length_px = focal_length_px
        self.principal_point = principal_point
        self.is_calibrated = is_calibrated

    def estimate_ground_resolution_at_row(self, row_y: float) -> Tuple[float, float]:
        """
        Estimates (meters_per_pixel_x, meters_per_pixel_y) at a specific image row y
        under a flat road-plane assumption.
        Returns (dx, dy) in meters per pixel.
        """
        cy = self.principal_point[1]
        delta_y = row_y - cy

        # Angle of ray relative to optical axis
        alpha = np.arctan2(delta_y, self.focal_length_px)
        # Total depression angle below horizontal
        theta = self.pitch_rad + alpha

        if theta <= 0.02: # Near or above horizon
            return (0.10, 0.10) # safe clamp

        # Distance along ground plane
        ground_distance_z = self.camera_height_m / np.tan(theta)

        # Spatial resolution at this distance
        m_per_px_x = ground_distance_z / self.focal_length_px
        m_per_px_y = (ground_distance_z**2 / (self.camera_height_m * self.focal_length_px)) * (np.sin(theta)**2)

        return (float(m_per_px_x), float(m_per_px_y))


class GeometryDamageAssessor:
    """
    Assesses physical defect dimensions, surface area, and relative/calibrated depth anomaly.
    Explicitly separates uncalibrated relative estimates from verified physical measurements.
    """

    def __init__(self, camera_model: Optional[CameraGeometryModel] = None):
        self.camera = camera_model or CameraGeometryModel(is_calibrated=False)

    def assess_defect(
        self,
        box: List[float],
        class_name: str,
        mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Performs geometric distress quantification.
        """
        x1, y1, x2, y2 = [float(v) for v in box]
        width_px = max(1.0, x2 - x1)
        height_px = max(1.0, y2 - y1)
        bottom_y = y2

        # 1. Pixel-level measurements
        if mask is not None and np.any(mask):
            surface_area_px = float(np.count_nonzero(mask))
            measurement_source = "segmentation_mask"
        else:
            surface_area_px = float(width_px * height_px)
            measurement_source = "bounding_box_envelope"

        # 2. Metric plane projection
        m_per_px_x, m_per_px_y = self.camera.estimate_ground_resolution_at_row(bottom_y)
        est_width_m = width_px * m_per_px_x
        est_length_m = height_px * m_per_px_y
        est_area_sqm = surface_area_px * (m_per_px_x * m_per_px_y)

        # 3. Depth anomaly analysis (comparing defect interior with surrounding pavement)
        depth_metrics = self._analyze_depth_anomaly(box, mask, depth_map)

        # 4. Objective geometric severity index (NOT based on detector confidence)
        severity_tier, severity_score = self._compute_geometric_severity(
            class_name=class_name,
            area_sqm=est_area_sqm,
            depth_anomaly_ratio=depth_metrics.get("relative_depth_anomaly_ratio")
        )

        return {
            "class_name": class_name,
            "measurement_source": measurement_source,
            "is_metric_calibrated": self.camera.is_calibrated,
            "pixel_dimensions": {
                "width_px": round(width_px, 1),
                "height_px": round(height_px, 1),
                "area_px": round(surface_area_px, 1)
            },
            "estimated_metric_dimensions": {
                "estimated_width_cm": round(est_width_m * 100.0, 1),
                "estimated_length_cm": round(est_length_m * 100.0, 1),
                "estimated_surface_area_sqcm": round(est_area_sqm * 10000.0, 1),
                "disclaimer": "Uncalibrated perspective approximation" if not self.camera.is_calibrated else "Calibrated road plane"
            },
            "depth_analysis": depth_metrics,
            "severity_assessment": {
                "severity_tier": severity_tier, # Low, Medium, High
                "geometric_severity_score": round(severity_score, 2),
                "assessment_basis": "physical area and depth gradient, independent of detector confidence"
            }
        }

    def _analyze_depth_anomaly(
        self,
        box: List[float],
        mask: Optional[np.ndarray],
        depth_map: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Samples depth within defect versus an outer ring of surrounding road.
        """
        if depth_map is None:
            return {"status": "depth_map_not_provided"}

        h, w = depth_map.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return {"status": "invalid_box"}

        # Defect interior region
        if mask is not None and mask.shape == (h, w):
            defect_region = (mask > 0)
        else:
            defect_region = np.zeros((h, w), dtype=bool)
            defect_region[y1:y2, x1:x2] = True

        # Surrounding baseline ring (expanded box minus defect region)
        pad = int(max(8, min(width_px := x2 - x1, height_px := y2 - y1) * 0.3))
        sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
        sx2, sy2 = min(w, x2 + pad), min(h, y2 + pad)

        surrounding_ring = np.zeros((h, w), dtype=bool)
        surrounding_ring[sy1:sy2, sx1:sx2] = True
        surrounding_ring[defect_region] = False

        valid_defect_depth = depth_map[defect_region & np.isfinite(depth_map) & (depth_map > 0)]
        valid_surround_depth = depth_map[surrounding_ring & np.isfinite(depth_map) & (depth_map > 0)]

        if len(valid_defect_depth) < 5 or len(valid_surround_depth) < 10:
            return {"status": "insufficient_valid_depth_pixels"}

        defect_median = float(np.median(valid_defect_depth))
        surround_median = float(np.median(valid_surround_depth))

        # Relative depth ratio (positive indicates defect is deeper than surrounding road)
        anomaly_ratio = (defect_median - surround_median) / max(1e-3, surround_median)

        return {
            "status": "valid_relative_estimate",
            "is_calibrated_physical_depth": False,
            "defect_interior_median_depth": round(defect_median, 3),
            "surrounding_road_median_depth": round(surround_median, 3),
            "relative_depth_anomaly_ratio": round(anomaly_ratio, 4),
            "note": "Relative depth map output only. Physical cm depth requires stereo/sensor ground-truth calibration."
        }

    @staticmethod
    def _compute_geometric_severity(
        class_name: str,
        area_sqm: float,
        depth_anomaly_ratio: Optional[float]
    ) -> Tuple[str, float]:
        """
        Computes reproducible severity based on surface distress area and depth anomaly.
        """
        score = 0.0

        # Area weighting (e.g. > 0.10 sqm is a significant distress zone)
        area_score = min(50.0, (area_sqm / 0.10) * 30.0)
        score += area_score

        # Distress type weighting
        cls_lower = class_name.lower()
        if "pothole" in cls_lower or "d40" in cls_lower:
            score += 25.0
            if depth_anomaly_ratio is not None and depth_anomaly_ratio > 0.05:
                score += min(25.0, depth_anomaly_ratio * 200.0)
        elif "alligator" in cls_lower or "d20" in cls_lower:
            score += 20.0
        elif "crack" in cls_lower:
            score += 10.0

        score = min(100.0, score)

        if score >= 65.0:
            tier = "High"
        elif score >= 35.0:
            tier = "Medium"
        else:
            tier = "Low"

        return (tier, score)

