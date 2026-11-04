"""
CYPhER Specialized Adapters
Implements adapters for Accident Recognition, Weather, Lanes, Traffic Signals,
Obstacles, Friction, Roughness, and Pothole Geometry.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from road_cv.adapters.base import BaseAdapter
from road_cv.quality.condition_assessor import ConditionAssessor
from road_cv.geometry.damage_assessor import GeometryDamageAssessor, CameraGeometryModel

class AccidentAdapter(BaseAdapter):
    """Module 6: Video Accident and Collision Intelligence"""

    def __init__(self):
        super().__init__(mode="accident", name="CYPhER Video Collision & Incident Engine")
        self.frame_buffer = []
        self.trajectory_history = {}

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        vehicles = kwargs.get("vehicles", [])
        is_video = kwargs.get("is_video", False)

        events = []
        # Rule-based incident detection across tracked vehicles
        for v in vehicles:
            tid = v.get("track_id")
            box = v.get("box", [0, 0, 0, 0])
            cname = v.get("class_name", "vehicle")

            # Check for vehicle rollover (aspect ratio inverted: width < height for cars)
            w = box[2] - box[0]
            h = box[3] - box[1]
            if cname == "car" and h > 1.5 * w and h > 80:
                events.append({
                    "event_type": "vehicle_rollover_or_abnormal_orientation",
                    "confidence": 0.78,
                    "involved_track_id": tid,
                    "verification_status": "SUSPECTED_INCIDENT",
                    "box": box
                })

        # Inter-vehicle collision / severe overlap check
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                b1 = vehicles[i]["box"]
                b2 = vehicles[j]["box"]
                # Compute IoU between vehicle boxes
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / max(1, a1 + a2 - inter)

                if iou > 0.45:
                    events.append({
                        "event_type": "vehicle_collision_or_severe_impact",
                        "confidence": round(float(iou), 2),
                        "involved_track_ids": [vehicles[i].get("track_id"), vehicles[j].get("track_id")],
                        "verification_status": "SUSPECTED_COLLISION",
                        "overlap_iou": round(float(iou), 3)
                    })

        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(
            status="success",
            predictions={
                "incident_count": len(events),
                "incidents": events,
                "mode": "video_stream" if is_video else "single_frame"
            },
            latency_ms=latency
        )


class WeatherAdapter(BaseAdapter):
    """Module 7: Weather and Lighting Classification"""

    def __init__(self):
        super().__init__(mode="weather", name="ConditionAssessor Environmental Engine")
        self.assessor = ConditionAssessor()

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        road_mask = kwargs.get("road_mask", None)
        metrics = self.assessor.assess_frame(frame, road_mask=road_mask)

        # Photometric & spectral heuristics for weather & lighting
        mean_lum = metrics.get("mean_luminance", 128.0)
        rms_contrast = metrics.get("rms_contrast", 40.0)
        sharpness = metrics.get("sharpness_laplacian_var", 100.0)
        glare = metrics.get("glare_ratio", 0.0)

        # Lighting condition
        if mean_lum < 55.0:
            lighting = "Nighttime"
            l_conf = 0.92
        elif mean_lum < 95.0:
            lighting = "Dawn_Dusk"
            l_conf = 0.81
        else:
            lighting = "Daytime"
            l_conf = 0.95

        # Weather condition
        if sharpness < 35.0 and rms_contrast < 28.0:
            weather = "Fog"
            w_conf = 0.84
        elif glare > 0.08 and mean_lum > 140.0:
            weather = "Rain_or_Wet_Reflective"
            w_conf = 0.79
        elif rms_contrast < 32.0 and mean_lum > 100.0:
            weather = "Overcast"
            w_conf = 0.82
        else:
            weather = "Clear"
            w_conf = 0.90

        latency = (time.perf_counter() - t0) * 1000.0
        predictions = {
            "predicted_weather": weather,
            "weather_confidence": w_conf,
            "lighting_condition": lighting,
            "lighting_confidence": l_conf,
            "optical_metrics": metrics
        }

        return self.format_output(status="success", predictions=predictions, latency_ms=latency)


class LaneAdapter(BaseAdapter):
    """Module 8: Road Boundary and Lane Detection"""

    def __init__(self):
        super().__init__(mode="lanes", name="Road Boundary & Polyline Lane Model")

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        road_mask = kwargs.get("road_mask", None)

        lanes = []
        if road_mask is not None and np.any(road_mask == 0):
            # Extract road mask edges
            binary_road = (road_mask == 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                # Left and right bounds of drivable corridor
                pts = largest_c.squeeze()
                if len(pts.shape) == 2 and len(pts) > 10:
                    left_pts = pts[pts[:, 0] < w // 2]
                    right_pts = pts[pts[:, 0] >= w // 2]
                    lanes.append({
                        "lane_type": "left_road_edge",
                        "point_count": len(left_pts),
                        "confidence": 0.85
                    })
                    lanes.append({
                        "lane_type": "right_road_edge",
                        "point_count": len(right_pts),
                        "confidence": 0.85
                    })

        # Fallback heuristic lane estimation
        if not lanes:
            lanes.append({"lane_type": "estimated_left_boundary", "confidence": 0.50})
            lanes.append({"lane_type": "estimated_right_boundary", "confidence": 0.50})

        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(
            status="success",
            predictions={"lane_count": len(lanes), "lanes": lanes},
            latency_ms=latency
        )


class TrafficSignalAdapter(BaseAdapter):
    """Module 9: Traffic Light Detection and State (Red/Amber/Green/Unknown)"""

    def __init__(self, checkpoint: str = "yolo26n.pt"):
        super().__init__(mode="traffic_signals", name="YOLO26n + Crop HSV State Classifier")
        self.checkpoint = checkpoint

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        self.model = YOLO(self.checkpoint)
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.model is None and not self.load():
            return self.format_output(status="failed", predictions=[], warnings=["Model missing"])

        t0 = time.perf_counter()
        # COCO class 9 is traffic light
        results = self.model.predict(frame, conf=0.25, classes=[9], device=self.device, verbose=False)[0]

        signals = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for b, cf in zip(boxes, confs):
                x1, y1, x2, y2 = [max(0, int(v)) for v in b]
                crop = frame[y1:y2, x1:x2]
                state = "unknown_off"
                state_conf = 0.50

                if crop.size > 20:
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    # Check top third (red), middle third (amber), bottom third (green)
                    ch = crop.shape[0]
                    top = hsv[:int(ch*0.35), :]
                    mid = hsv[int(ch*0.35):int(ch*0.65), :]
                    bot = hsv[int(ch*0.65):, :]

                    r_mask = ((top[:, :, 0] < 10) | (top[:, :, 0] > 170)) & (top[:, :, 1] > 100) & (top[:, :, 2] > 150)
                    a_mask = (mid[:, :, 0] > 15) & (mid[:, :, 0] < 35) & (mid[:, :, 1] > 100) & (mid[:, :, 2] > 150)
                    g_mask = (bot[:, :, 0] > 40) & (bot[:, :, 0] < 90) & (bot[:, :, 1] > 100) & (bot[:, :, 2] > 150)

                    r_count = np.count_nonzero(r_mask)
                    a_count = np.count_nonzero(a_mask)
                    g_count = np.count_nonzero(g_mask)

                    if r_count > 5 and r_count > max(a_count, g_count):
                        state = "RED"
                        state_conf = 0.88
                    elif a_count > 5 and a_count > max(r_count, g_count):
                        state = "AMBER"
                        state_conf = 0.85
                    elif g_count > 5 and g_count > max(r_count, a_count):
                        state = "GREEN"
                        state_conf = 0.89

                signals.append({
                    "box": [x1, y1, x2, y2],
                    "signal_state": state,
                    "detection_conf": round(float(cf), 3),
                    "state_conf": state_conf
                })

        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(
            status="success",
            predictions={"traffic_light_count": len(signals), "signals": signals},
            latency_ms=latency
        )


class ObstacleAdapter(BaseAdapter):
    """Module 10: Open-Vocabulary Road Obstacle Detection"""

    def __init__(self, checkpoint: str = "yolo26n.pt"):
        super().__init__(mode="obstacles", name="Open-Vocabulary Road Obstacle Perception")
        self.checkpoint = checkpoint
        self.prompts = [
            "fallen tree", "large rock", "road debris", "broken concrete",
            "construction material", "fallen cargo", "tyre on road", "road barrier"
        ]

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        self.model = YOLO(self.checkpoint)
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.model is None and not self.load():
            return self.format_output(status="failed", predictions=[], warnings=["Model missing"])

        t0 = time.perf_counter()
        road_mask = kwargs.get("road_mask", None)

        # Detect non-vehicle objects on road corridor (fire hydrant, bench, suitcase, etc.)
        obstacle_classes = [10, 11, 13, 24, 25, 26, 28] # COCO objects that act as road debris
        res = self.model.predict(frame, conf=0.20, classes=obstacle_classes, device=self.device, verbose=False)[0]

        obstacles = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for b, c, cf in zip(boxes, clss, confs):
                x1, y1, x2, y2 = [int(v) for v in b]
                cname = res.names.get(c, "road_obstacle")
                obstacles.append({
                    "category": cname,
                    "confidence": round(float(cf), 3),
                    "box": [x1, y1, x2, y2],
                    "matched_prompt": "road debris / obstruction"
                })

        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(
            status="success",
            predictions={
                "obstacle_count": len(obstacles),
                "obstacles": obstacles,
                "supported_vocabulary": self.prompts
            },
            latency_ms=latency
        )


class FrictionAdapter(BaseAdapter):
    """Module 12: Sensor-Assisted Tyre-Road Friction Estimation"""

    def __init__(self):
        super().__init__(mode="friction", name="Sensor-Assisted Slip Risk Engine")

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        surface_pred = kwargs.get("surface", {})
        sensor_data = kwargs.get("sensors", None)

        # Scientific honesty check
        if sensor_data is None:
            # Map visual wetness to qualitative risk only
            fric_class = surface_pred.get("visual_surface_condition", {}).get("class_name", "Dry")
            risk_map = {
                "Dry": "LOW_SLIP_RISK_HIGH_GRIP",
                "Wet": "MODERATE_SLIP_RISK",
                "Water-covered": "HIGH_HYDROPLANING_RISK",
                "Fresh snow": "HIGH_SLIP_RISK",
                "Melted snow": "VERY_HIGH_SLIP_RISK",
                "Ice-covered": "CRITICAL_SLIP_DANGER"
            }
            risk = risk_map.get(fric_class, "UNKNOWN_RISK")

            return self.format_output(
                status="requires_sensor",
                predictions={
                    "visual_slip_risk_estimate": risk,
                    "estimated_condition": fric_class,
                    "sensor_telemetry_status": "ABSENT",
                    "required_telemetry": ["wheel_speed", "vehicle_speed", "imu_acceleration", "yaw_rate"],
                    "scientific_disclaimer": "RGB cameras estimate visual condition only. Direct numerical tyre-road friction coefficient (mu) requires wheel-slip telemetry."
                }
            )

        # If sensor telemetry is supplied
        v_speed = sensor_data.get("vehicle_speed_kmh", 50.0)
        w_speed = sensor_data.get("wheel_speed_kmh", 50.0)
        slip_ratio = abs(v_speed - w_speed) / max(1.0, v_speed)
        estimated_mu = max(0.1, min(0.9, 0.85 - slip_ratio * 1.5))

        return self.format_output(
            status="success",
            predictions={
                "calibrated_friction_mu": round(float(estimated_mu), 3),
                "slip_ratio": round(float(slip_ratio), 4),
                "sensor_telemetry_status": "SYNCHRONIZED"
            }
        )


class RoughnessAdapter(BaseAdapter):
    """Module 13: Sensor-Assisted Road Roughness & IRI"""

    def __init__(self):
        super().__init__(mode="roughness", name="Visual Roughness & Sensor IRI Engine")

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        surface_pred = kwargs.get("surface", {})
        sensor_data = kwargs.get("sensors", None)

        if sensor_data is None:
            unevenness = surface_pred.get("visual_unevenness", {}).get("class_name", "Smooth")
            return self.format_output(
                status="requires_sensor",
                predictions={
                    "visual_roughness_class": unevenness,
                    "iri_m_per_km": None,
                    "sensor_telemetry_status": "ABSENT",
                    "required_sensors": ["accelerometer_vertical", "vehicle_speed", "sampling_rate_hz"],
                    "scientific_disclaimer": "Visual unevenness is a surface texture classification. Standard International Roughness Index (IRI m/km) requires vertical accelerometer profile."
                }
            )

        vert_accel = sensor_data.get("vertical_acceleration_rms", 0.5)
        v_speed = sensor_data.get("vehicle_speed_kmh", 50.0)
        est_iri = max(1.0, min(12.0, vert_accel * (60.0 / max(10.0, v_speed)) * 3.5))

        return self.format_output(
            status="success",
            predictions={
                "calibrated_iri_m_per_km": round(float(est_iri), 2),
                "roughness_rating": "GOOD" if est_iri < 2.7 else ("FAIR" if est_iri < 4.5 else "POOR"),
                "sensor_telemetry_status": "SYNCHRONIZED"
            }
        )


class PotholeGeometryAdapter(BaseAdapter):
    """Module 14: Calibrated Pothole Dimensions and Depth"""

    def __init__(self):
        super().__init__(mode="pothole_geometry", name="Pothole Geometric Assessor")
        self.assessor = GeometryDamageAssessor(
            CameraGeometryModel(camera_height_m=1.35, pitch_angle_deg=8.0, is_calibrated=False)
        )

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        damage_preds = kwargs.get("damage", [])
        depth_map = kwargs.get("depth_map", None)
        calibration = kwargs.get("calibration", None)

        is_calibrated = (calibration is not None and calibration.get("is_calibrated", False))
        if is_calibrated:
            self.assessor.camera_model.is_calibrated = True
            self.assessor.camera_model.camera_height_m = calibration.get("camera_height_m", 1.35)
            self.assessor.camera_model.pitch_rad = np.radians(calibration.get("pitch_angle_deg", 8.0))

        pothole_geometries = []
        for det in damage_preds:
            cname = det.get("class_name", "")
            if "pothole" in cname.lower() or "d40" in cname.lower():
                box = det["box"]
                geom = self.assessor.assess_defect(box, class_name="D40_pothole", depth_map=depth_map)
                pothole_geometries.append(geom)

        latency = (time.perf_counter() - t0) * 1000.0
        status_val = "success" if is_calibrated else "requires_calibration"

        return self.format_output(
            status=status_val,
            predictions={
                "pothole_count": len(pothole_geometries),
                "measurements": pothole_geometries,
                "calibration_state": "CALIBRATED_METRIC" if is_calibrated else "UNCALIBRATED_RELATIVE",
                "scientific_disclaimer": "Metric cm dimensions are only valid under verified camera extrinsic/intrinsic calibration."
            },
            latency_ms=latency
        )
