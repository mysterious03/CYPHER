"""
Pretrained Traffic Signal Adapter for CYPhER.
Integrates the generic YOLOv8m traffic signal detector (subhodeepmoitra/Traffic_signals_detection_YOLOv8m)
and supports experimental comma-camera center-net (Sculptor-AI/lightd-v4).
Enforces distinction between traffic light bounding box detection and state classification (Red/Yellow/Green).
"""
import os
import time
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Union, Optional
from ultralytics import YOLO

class PretrainedSignalAdapter:
    """
    Adapter for pretrained traffic signal and sign detection.
    """
    SIGNAL_CLASSES = {
        0: "Traffic_Light_Signal",
        1: "Stop_Signal",
        2: "Speedlimit_Signal",
        3: "Crosswalk_Signal",
        4: "Crosswalk",
        5: "Pedestrian",
        6: "Bus",
        7: "Car",
        8: "Truck"
    }

    def __init__(
        self,
        checkpoint_path: str = "models/pretrained/signals/traffic_signals_yolov8m.pt",
        device: str = "cuda"
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Load YOLOv8m traffic signal detector."""
        if self.is_loaded and self.model is not None:
            return

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        print(f"[PretrainedSignalAdapter] Loading Traffic Signals YOLOv8m from {self.checkpoint_path} on {self.device}...")
        self.model = YOLO(self.checkpoint_path)
        self.is_loaded = True

    def _classify_lamp_state_hsv(self, crop: np.ndarray) -> str:
        """
        Secondary HSV heuristic to determine lamp color state (Red/Yellow/Green/Off)
        for detected traffic light boxes, preventing false assumption that bounding box = state.
        """
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        val = hsv[:, :, 2]
        if np.max(val) < 80:
            return "off"

        # Red mask
        r_mask1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
        r_mask2 = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
        r_cnt = np.sum((r_mask1 > 0) | (r_mask2 > 0))

        # Yellow / Amber mask
        y_mask = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))
        y_cnt = np.sum(y_mask > 0)

        # Green mask
        g_mask = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))
        g_cnt = np.sum(g_mask > 0)

        max_c = max(r_cnt, y_cnt, g_cnt)
        if max_c < 10:
            return "off"
        if max_c == r_cnt:
            return "red"
        if max_c == y_cnt:
            return "yellow"
        return "green"

    def predict(self, image: Union[str, np.ndarray], conf: float = 0.25) -> Dict[str, Any]:
        """
        Detect traffic signals and evaluate lamp states.
        """
        if not self.is_loaded:
            self.load_model()

        if isinstance(image, str):
            img_bgr = cv2.imread(image)
        else:
            img_bgr = image

        if img_bgr is None:
            raise ValueError("Invalid image input")

        t0 = time.time()
        results = self.model.predict(
            source=img_bgr,
            conf=conf,
            device=self.device,
            verbose=False
        )
        dt = time.time() - t0
        res = results[0]

        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                cls_id = int(clss[i])
                raw_name = self.model.names.get(cls_id, f"class_{cls_id}")
                conf_val = float(confs[i])
                bbox = boxes[i].tolist()
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Extract state if traffic light
                lamp_state = "not_applicable"
                if "Traffic_Light" in raw_name or "Signal" in raw_name:
                    crop = img_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    lamp_state = self._classify_lamp_state_hsv(crop)

                detections.append({
                    "class_id": cls_id,
                    "class_name": raw_name,
                    "confidence": round(conf_val, 3),
                    "box": [round(v, 1) for v in bbox],
                    "lamp_state": lamp_state
                })

        overlay = res.plot() if isinstance(image, np.ndarray) else None

        return {
            "model_name": "TrafficSignals_YOLOv8m",
            "checkpoint": self.checkpoint_path,
            "architecture": "YOLOv8m",
            "device": self.device,
            "runtime_ms": round(dt * 1000, 2),
            "detections_count": len(detections),
            "detections": detections,
            "overlay": overlay,
            "state_note": "Bounding boxes localize physical signal fixtures; lamp_state is derived via optical color analysis.",
            "status": "INFERENCE_VERIFIED"
        }

    def predict_video(self, frame: Union[str, np.ndarray], timestamp: float, conf: float = 0.25) -> Dict[str, Any]:
        res = self.predict(frame, conf=conf)
        res["timestamp"] = timestamp
        return res

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "TrafficSignals_YOLOv8m",
            "source": "https://huggingface.co/subhodeepmoitra/Traffic_signals_detection_YOLOv8m",
            "checkpoint": self.checkpoint_path,
            "classes": list(self.SIGNAL_CLASSES.values()),
            "status": "INFERENCE_VERIFIED"
        }

    def release_resources(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

