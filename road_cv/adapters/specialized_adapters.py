"""
CYPhER V2.2 Specialized Perception Adapters
Upgrades heuristic modes with dedicated neural models and zero-shot architectures:
- AccidentAdapter: TemporalAccidentDetector with 16-frame 3D-CNN and trajectory anomaly checks
- WeatherAdapter: MultiTaskWeatherClassifier (MobileNetV3-Small) with photometric verification
- LaneAdapter: DedicatedLaneDetector (row-wise anchor polyline model)
- TrafficSignalAdapter: Hybrid neural CNN + optical vertical thirds state classifier
- ObstacleAdapter: OpenVocabularyObstacleDetector (YOLO-World zero-shot natural language prompts)
- FrictionAdapter / RoughnessAdapter: Physical sensor safeguards
- PotholeGeometryAdapter: Calibrated camera geometry and relative depth profiling
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from road_cv.adapters.base import BaseAdapter
from road_cv.quality.condition_assessor import ConditionAssessor
from road_cv.geometry.damage_assessor import GeometryDamageAssessor, CameraGeometryModel

# Import V2.2 upgraded modules
from road_cv.accident.temporal_accident_detector import TemporalAccidentDetector
from road_cv.weather.weather_model import MultiTaskWeatherClassifier, WEATHER_CLASSES, LIGHTING_CLASSES
from road_cv.lanes.lane_detector import DedicatedLaneDetector
from road_cv.traffic_signals.signal_classifier import DedicatedTrafficSignalClassifier
from road_cv.obstacles.open_vocabulary_obstacles import OpenVocabularyObstacleDetector, DEFAULT_OBSTACLE_PROMPTS


class AccidentAdapter(BaseAdapter):
    """Module 6: Video Accident and Collision Intelligence (V2.2 Temporal 3D-CNN + Trajectory)"""

    def __init__(self):
        super().__init__(mode="accident", name="CYPhER Video Collision & Incident Engine")
        self.temporal_engine = TemporalAccidentDetector(buffer_size=16)

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        vehicles = kwargs.get("vehicles", [])
        is_video = kwargs.get("is_video", False)
        timestamp = kwargs.get("timestamp", time.time())

        # Execute temporal / geometric anomaly detection
        res = self.temporal_engine.analyze(
            frame=frame,
            detected_vehicles=vehicles,
            timestamp=timestamp,
            is_video=is_video
        )

        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(
            status="success",
            predictions=res,
            latency_ms=latency
        )


class WeatherAdapter(BaseAdapter):
    """Module 7: Environmental Weather and Lighting Classification (V2.2 Multi-Task Neural Engine)"""

    def __init__(self, checkpoint: str = "runs/weather_classifier_v22/best_weather_model.pt"):
        super().__init__(mode="weather", name="Multi-Task MobileNetV3 Weather Classifier")
        self.checkpoint = checkpoint
        self.assessor = ConditionAssessor()
        self.neural_model = None

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        if Path(self.checkpoint).exists():
            try:
                ckpt = torch.load(self.checkpoint, map_location=self.device)
                self.neural_model = MultiTaskWeatherClassifier(pretrained=False)
                self.neural_model.load_state_dict(ckpt["state_dict"])
                self.neural_model.to(self.device).eval()
            except Exception as e:
                print(f"[WeatherAdapter WARNING] Checkpoint load failed: {e}")
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        road_mask = kwargs.get("road_mask", None)
        optical_metrics = self.assessor.assess_frame(frame, road_mask=road_mask)

        # 1. Neural Prediction Path (if checkpoint loaded)
        if self.neural_model is None:
            self.load()

        if self.neural_model is not None:
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = t.unsqueeze(0).to(self.device)

            with torch.no_grad():
                w_logits, l_logits = self.neural_model(t)
                w_probs = torch.softmax(w_logits, dim=-1)[0].cpu().numpy()
                l_probs = torch.softmax(l_logits, dim=-1)[0].cpu().numpy()

                w_idx = int(np.argmax(w_probs))
                l_idx = int(np.argmax(l_probs))

                weather = WEATHER_CLASSES[w_idx]
                w_conf = float(w_probs[w_idx])
                lighting = LIGHTING_CLASSES[l_idx]
                l_conf = float(l_probs[l_idx])
                method = "neural_mobilenet_v3"
        else:
            # 2. Photometric Optical Heuristic Fallback
            mean_lum = optical_metrics.get("mean_luminance", 128.0)
            rms_contrast = optical_metrics.get("rms_contrast", 40.0)
            sharpness = optical_metrics.get("sharpness_laplacian_var", 100.0)
            glare = optical_metrics.get("glare_ratio", 0.0)

            lighting = "Nighttime" if mean_lum < 55.0 else ("Dawn_Dusk" if mean_lum < 95.0 else "Daytime")
            l_conf = 0.85
            if sharpness < 35.0 and rms_contrast < 28.0:
                weather = "Fog"
            elif glare > 0.08 and mean_lum > 140.0:
                weather = "Rain"
            elif rms_contrast < 32.0 and mean_lum > 100.0:
                weather = "Overcast"
            else:
                weather = "Clear"
            w_conf = 0.85
            method = "photometric_optical_fallback"

        latency = (time.perf_counter() - t0) * 1000.0
        predictions = {
            "predicted_weather": weather,
            "weather_confidence": round(w_conf, 4),
            "lighting_condition": lighting,
            "lighting_confidence": round(l_conf, 4),
            "method": method,
            "optical_metrics": optical_metrics
        }
        return self.format_output(status="success", predictions=predictions, latency_ms=latency)


class LaneAdapter(BaseAdapter):
    """Module 8: Dedicated Lane & Polyline Boundary Detector (V2.2 Dedicated Polyline Engine)"""

    def __init__(self, checkpoint: str = "runs/lane_detector_v22/best_lane_model.pt"):
        super().__init__(mode="lanes", name="Dedicated Polyline Lane Detector")
        self.checkpoint = checkpoint
        self.detector = DedicatedLaneDetector(checkpoint_path=self.checkpoint)

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        road_mask = kwargs.get("road_mask", None)
        res = self.detector.predict(frame, road_mask=road_mask)
        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(status="success", predictions=res.get("predictions", {}), latency_ms=latency)


class TrafficSignalAdapter(BaseAdapter):
    """Module 9: Traffic Light Detection & State Classifier (V2.2 Hybrid Neural + HSV)"""

    def __init__(self, detection_ckpt: str = "yolo26n.pt"):
        super().__init__(mode="traffic_signals", name="YOLO26n + Dedicated State Classifier")
        self.detection_ckpt = detection_ckpt
        self.detector = None
        self.classifier = DedicatedTrafficSignalClassifier()

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        self.detector = YOLO(self.detection_ckpt)
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.detector is None and not self.load():
            return self.format_output(status="failed", predictions=[], warnings=["Detector missing"])

        t0 = time.perf_counter()
        results = self.detector.predict(frame, conf=0.25, classes=[9], device=self.device, verbose=False)[0]

        signals = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for b, cf in zip(boxes, confs):
                x1, y1, x2, y2 = [max(0, int(v)) for v in b]
                crop = frame[y1:y2, x1:x2]
                state_res = self.classifier.classify_crop(crop)

                signals.append({
                    "box": [x1, y1, x2, y2],
                    "signal_state": state_res["state"],
                    "state_confidence": state_res["confidence"],
                    "detection_conf": round(float(cf), 3),
                    "method": state_res["method"]
                })

        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(
            status="success",
            predictions={"traffic_light_count": len(signals), "signals": signals},
            latency_ms=latency
        )


class ObstacleAdapter(BaseAdapter):
    """Module 10: Open-Vocabulary Text-Prompted Obstacle Perception (V2.2 YOLO-World)"""

    def __init__(self):
        super().__init__(mode="obstacles", name="YOLO-World Open-Vocabulary Obstacle Perception")
        self.detector = OpenVocabularyObstacleDetector(
            model_path="yolov8s-world.pt",
            prompts=DEFAULT_OBSTACLE_PROMPTS
        )

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        road_mask = kwargs.get("road_mask", None)
        custom_prompts = kwargs.get("prompts", None)

        res = self.detector.predict(frame, road_mask=road_mask, custom_prompts=custom_prompts)
        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(
            status="success",
            predictions=res,
            latency_ms=latency
        )


class FrictionAdapter(BaseAdapter):
    """Module 12: Sensor-Assisted Tyre-Road Friction Estimation"""

    def __init__(self):
        super().__init__(mode="friction", name="Vision-Assisted Slip Risk Estimator")

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sensors = kwargs.get("sensors", None)
        surface_pred = kwargs.get("surface", {})

        condition = "Dry"
        if isinstance(surface_pred, dict) and "visual_surface_condition" in surface_pred:
            cond_data = surface_pred["visual_surface_condition"]
            if isinstance(cond_data, dict):
                condition = cond_data.get("class_name", "Dry")
            elif isinstance(cond_data, str):
                condition = cond_data

        risk_level = "LOW"
        if condition in ["Wet"]:
            risk_level = "MEDIUM"
        elif condition in ["Water-covered", "Melted snow", "Ice-covered", "Mud"]:
            risk_level = "HIGH"

        status_flag = "success" if sensors is not None else "requires_sensor"
        predictions = {
            "visual_slip_risk_estimate": risk_level,
            "visual_surface_condition": condition,
            "measured_friction_mu": None if sensors is None else sensors.get("wheel_slip_friction_mu", 0.72),
            "sensor_telemetry_provided": sensors is not None,
            "scientific_disclaimer": (
                "True physical tire-road friction coefficient mu requires wheel speed encoders, "
                "chassis accelerometers, and powertrain torque data. Single-camera RGB provides visual slip risk only."
            )
        }
        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(status=status_flag, predictions=predictions, latency_ms=latency)


class RoughnessAdapter(BaseAdapter):
    """Module 13: Sensor-Assisted Road Roughness & IRI Class"""

    def __init__(self):
        super().__init__(mode="roughness", name="Vision-Assisted Roughness Estimator")

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sensors = kwargs.get("sensors", None)
        surface_pred = kwargs.get("surface", {})

        roughness_class = "Smooth"
        if isinstance(surface_pred, dict) and "visual_unevenness" in surface_pred:
            u_data = surface_pred["visual_unevenness"]
            if isinstance(u_data, dict):
                roughness_class = u_data.get("class_name", "Smooth")
            elif isinstance(u_data, str):
                roughness_class = u_data

        status_flag = "success" if sensors is not None else "requires_sensor"
        predictions = {
            "visual_roughness_class": roughness_class,
            "iri_m_per_km": None if sensors is None else sensors.get("quarter_car_iri_m_per_km", 2.1),
            "sensor_telemetry_provided": sensors is not None,
            "scientific_disclaimer": (
                "International Roughness Index (IRI m/km) requires vertical accelerometer integration "
                "measuring vehicle chassis oscillation. Single-camera RGB estimates visual unevenness texture only."
            )
        }
        latency = (time.perf_counter() - t0) * 1000.0
        return self.format_output(status=status_flag, predictions=predictions, latency_ms=latency)


class PotholeGeometryAdapter(BaseAdapter):
    """Module 14: Calibrated Pothole Dimensions and Volumetric Depth"""

    def __init__(self):
        super().__init__(mode="pothole_geometry", name="Calibrated Pothole Geometry Profiler")
        self.assessor = GeometryDamageAssessor()

    def load(self, device: Optional[str] = None) -> bool:
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        damage_list = kwargs.get("damage", [])
        depth_map = kwargs.get("depth_map", None)
        calibration = kwargs.get("calibration", None)

        potholes = [d for d in damage_list if "D40" in str(d.get("class_name", "")) or "pothole" in str(d.get("class_name", "")).lower()]

        if calibration is None:
            # Uncalibrated relative measurements
            rel_measurements = []
            for p in potholes:
                b = p["box"]
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                aspect = bw / max(1, bh)

                depth_stat = 0.5
                if depth_map is not None:
                    cx = int((b[0] + b[2]) / 2)
                    cy = int((b[1] + b[3]) / 2)
                    if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                        depth_stat = float(depth_map[cy, cx])

                rel_measurements.append({
                    "box": b,
                    "relative_width_px": bw,
                    "relative_aspect_ratio": round(aspect, 2),
                    "relative_depth_gradient": round(depth_stat, 3),
                    "metric_dimensions_available": False
                })

            latency = (time.perf_counter() - t0) * 1000.0
            return self.format_output(
                status="requires_calibration",
                predictions={
                    "calibration_state": "UNCALIBRATED_RELATIVE",
                    "pothole_count": len(rel_measurements),
                    "potholes": rel_measurements,
                    "scientific_disclaimer": (
                        "Camera focal length, mount height, and pitch angle are required to calculate "
                        "metric dimensions (cm, m). Reporting uncalibrated relative dimensions."
                    )
                },
                latency_ms=latency
            )
        else:
            # Calibrated metric measurements
            metric_results = []
            cam_model = CameraGeometryModel(
                focal_length_px=calibration.get("focal_length_px", 1000.0),
                camera_height_m=calibration.get("camera_height_m", 1.4),
                pitch_angle_deg=calibration.get("pitch_angle_deg", 5.0)
            )

            for p in potholes:
                b = p["box"]
                dim = self.assessor.estimate_pothole_dimensions_calibrated(b, depth_map, cam_model)
                metric_results.append(dim)

            latency = (time.perf_counter() - t0) * 1000.0
            return self.format_output(
                status="success",
                predictions={
                    "calibration_state": "CALIBRATED_METRIC",
                    "pothole_count": len(metric_results),
                    "potholes": metric_results
                },
                latency_ms=latency
            )
