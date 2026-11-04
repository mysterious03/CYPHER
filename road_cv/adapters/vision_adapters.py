"""
CYPhER Core Vision Model Adapters
Implements adapters for Road Damage, Vehicles, Road Scene, Waterlogging, Depth, and Surface.
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
from road_cv.surface.surface_predictor import RoadSurfacePredictor
from road_cv.water.water_analyzer import WaterHazardAnalyzer

class DamageAdapter(BaseAdapter):
    """Module 1: Road Damage Detection (D00, D10, D20, D40)"""

    def __init__(self, checkpoint: str = "runs/detect/runs/road_damage/B0_baseline/weights/best.pt", conf_thresh: float = 0.25):
        super().__init__(mode="damage", name="Custom YOLO26n Road Damage Detector")
        self.checkpoint = checkpoint
        self.conf_thresh = conf_thresh

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        if not Path(self.checkpoint).exists():
            self.status = "CHECKPOINT_MISSING"
            return False
        self.model = YOLO(self.checkpoint)
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.model is None and not self.load():
            return self.format_output(status="failed", predictions=[], warnings=["Checkpoint missing"])

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        conf = kwargs.get("conf", self.conf_thresh)
        imgsz = kwargs.get("imgsz", 640)

        results = self.model.predict(frame, imgsz=imgsz, conf=conf, device=self.device, verbose=False)[0]
        latency = (time.perf_counter() - t0) * 1000.0

        predictions = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for b, c, cf in zip(boxes, clss, confs):
                x1, y1, x2, y2 = [int(v) for v in b]
                bw = x2 - x1
                bh = y2 - y1
                area_px = bw * bh

                # Detection scale classification
                if area_px < (32 * 32):
                    scale = "small"
                elif area_px < (96 * 96):
                    scale = "medium"
                else:
                    scale = "large"

                cname = results.names.get(c, str(c))
                predictions.append({
                    "class_id": int(c),
                    "class_name": cname,
                    "confidence": round(float(cf), 3),
                    "box": [x1, y1, x2, y2],
                    "area_px": int(area_px),
                    "scale": scale
                })

        return self.format_output(
            status="success",
            predictions=predictions,
            latency_ms=latency,
            detection_count=len(predictions)
        )


class VehicleAdapter(BaseAdapter):
    """Module 2: Vehicle & Road User Detection and Tracking"""

    def __init__(self, checkpoint: str = "yolo26n-seg.pt", conf_thresh: float = 0.35):
        super().__init__(mode="vehicles", name="YOLO26n Vehicle Instance Segmenter & Tracker")
        self.checkpoint = checkpoint
        self.conf_thresh = conf_thresh
        self.track_history = {}

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        if not Path(self.checkpoint).exists():
            self.status = "CHECKPOINT_MISSING"
            return False
        self.model = YOLO(self.checkpoint)
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.model is None and not self.load():
            return self.format_output(status="failed", predictions=[], warnings=["Checkpoint missing"])

        t0 = time.perf_counter()
        conf = kwargs.get("conf", self.conf_thresh)
        is_video = kwargs.get("is_video", False)

        # Vehicle & road user classes from COCO: person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
        target_classes = [0, 1, 2, 3, 5, 7]

        if is_video:
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf,
                classes=target_classes,
                device=self.device,
                verbose=False
            )[0]
        else:
            results = self.model.predict(
                frame,
                conf=conf,
                classes=target_classes,
                device=self.device,
                verbose=False
            )[0]

        latency = (time.perf_counter() - t0) * 1000.0
        predictions = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int) if (results.boxes.id is not None) else None

            for i, (b, c, cf) in enumerate(zip(boxes, clss, confs)):
                tid = int(track_ids[i]) if (track_ids is not None and i < len(track_ids)) else (i + 1)
                cname = results.names.get(c, str(c))
                predictions.append({
                    "track_id": tid,
                    "class_id": int(c),
                    "class_name": cname,
                    "confidence": round(float(cf), 3),
                    "box": [int(v) for v in b]
                })

        return self.format_output(
            status="success",
            predictions=predictions,
            latency_ms=latency,
            vehicle_count=len(predictions)
        )


class RoadSceneAdapter(BaseAdapter):
    """Module 3: Road & Environment Semantic Segmentation"""

    def __init__(self, checkpoint: str = "yolo26n-sem.pt"):
        super().__init__(mode="road_scene", name="YOLO26n Road Scene Semantic Segmenter")
        self.checkpoint = checkpoint

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        if not Path(self.checkpoint).exists():
            self.status = "CHECKPOINT_MISSING"
            return False
        self.model = YOLO(self.checkpoint)
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.model is None and not self.load():
            return self.format_output(status="failed", predictions={}, warnings=["Checkpoint missing"])

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        res = self.model.predict(frame, imgsz=768, device=self.device, verbose=False)[0]
        latency = (time.perf_counter() - t0) * 1000.0

        if hasattr(res, "semantic_mask") and res.semantic_mask is not None:
            raw_mask = res.semantic_mask.data.cpu().numpy().squeeze().astype(np.uint8)
            full_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            full_mask = np.zeros((h, w), dtype=np.uint8)

        # Class 0 is road in Cityscapes
        road_pixels = int(np.count_nonzero(full_mask == 0))
        total_pixels = h * w
        road_coverage_pct = round((road_pixels / max(1, total_pixels)) * 100.0, 2)

        predictions = {
            "road_coverage_pct": road_coverage_pct,
            "classes_detected": [int(c) for c in np.unique(full_mask)],
            "mask_shape": list(full_mask.shape)
        }

        return self.format_output(
            status="success",
            predictions=predictions,
            latency_ms=latency,
            road_mask=full_mask
        )


class WaterAdapter(BaseAdapter):
    """Module 4: Waterlogging & Wet-Road Segmentation"""

    def __init__(self, checkpoint: str = "runs/water_segmenter/best_water_model.pt"):
        super().__init__(mode="water", name="LRASPP Road Waterlogging Segmenter")
        self.checkpoint = checkpoint
        self.analyzer = None

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        if not Path(self.checkpoint).exists():
            self.status = "CHECKPOINT_MISSING"
            return False
        self.analyzer = WaterHazardAnalyzer(checkpoint_path=self.checkpoint, device=self.device)
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.analyzer is None and not self.load():
            return self.format_output(status="failed", predictions={}, warnings=["Checkpoint missing"])

        t0 = time.perf_counter()
        detected_potholes = kwargs.get("detected_potholes", [])
        road_mask = kwargs.get("road_mask", None)

        analysis = self.analyzer.analyze_image(
            frame,
            road_mask=road_mask,
            detected_potholes=detected_potholes
        )
        latency = (time.perf_counter() - t0) * 1000.0

        return self.format_output(
            status="success",
            predictions=analysis,
            latency_ms=latency
        )


class DepthAdapter(BaseAdapter):
    """Module 5: Monocular Depth & Road Surface Geometry"""

    def __init__(self, checkpoint: str = "yolo26n-depth.pt"):
        super().__init__(mode="depth", name="YOLO26n Monocular Depth Estimator")
        self.checkpoint = checkpoint

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        if not Path(self.checkpoint).exists():
            self.status = "CHECKPOINT_MISSING"
            return False
        self.model = YOLO(self.checkpoint)
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.model is None and not self.load():
            return self.format_output(status="failed", predictions={}, warnings=["Checkpoint missing"])

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        res = self.model.predict(frame, imgsz=640, device=self.device, verbose=False)[0]
        latency = (time.perf_counter() - t0) * 1000.0

        depth = res.depth.data.cpu().numpy().astype(np.float32)
        valid = np.isfinite(depth)

        depth_min = float(np.min(depth[valid]))
        depth_max = float(np.max(depth[valid]))
        depth_median = float(np.median(depth[valid]))

        predictions = {
            "measurement_type": "relative_monocular",
            "depth_range": [round(depth_min, 4), round(depth_max, 4)],
            "depth_median": round(depth_median, 4),
            "disclaimer": "Relative ordinal depth only. Calibrated metric depth requires stereo/LiDAR calibration."
        }

        return self.format_output(
            status="success",
            predictions=predictions,
            latency_ms=latency,
            depth_map=depth
        )


class SurfaceAdapter(BaseAdapter):
    """Module 11: Visual Surface Condition & Material Classification"""

    def __init__(self, checkpoint: str = "runs/surface_classifier/best_surface_model.pt"):
        super().__init__(mode="surface", name="Multi-Task Surface Classifier")
        self.checkpoint = checkpoint
        self.predictor = None

    def load(self, device: Optional[str] = None) -> bool:
        self.device = device or self.device
        if not Path(self.checkpoint).exists():
            self.status = "CHECKPOINT_MISSING"
            return False
        self.predictor = RoadSurfacePredictor(checkpoint_path=self.checkpoint, device=self.device)
        self.status = "READY"
        return True

    def predict(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        if self.predictor is None and not self.load():
            return self.format_output(status="failed", predictions={}, warnings=["Checkpoint missing"])

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        # Sample road patch from the bottom 40% of the frame (drivable corridor)
        road_patch = frame[int(h * 0.6):, int(w * 0.2):int(w * 0.8)]
        if road_patch.size == 0:
            road_patch = frame

        results = self.predictor.predict_patch(road_patch)
        latency = (time.perf_counter() - t0) * 1000.0

        return self.format_output(
            status="success",
            predictions=results,
            latency_ms=latency
        )

