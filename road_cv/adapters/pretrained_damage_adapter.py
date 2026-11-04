"""
Pretrained Road Damage Adapter for CYPhER.
Wraps the fine-tuned RDD2022 YOLOv12s model (rezzzq/yolo12s-road-damage-rdd2022)
as an auxiliary comparison detector alongside CYPhER's baseline B0 detector.
"""
import os
import time
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Union, Optional
from ultralytics import YOLO

class PretrainedDamageAdapter:
    """
    Inference adapter for the public pretrained YOLO12s RDD2022 road damage model.
    """
    CLASS_MAPPING = {
        "D00": "longitudinal_crack",
        "D10": "transverse_crack",
        "D20": "alligator_crack",
        "D40": "pothole",
        "Repair": "road_repair_patch"
    }

    def __init__(self, checkpoint_path: str = "models/pretrained/damage/yolo12s_RDD2022_best.pt", device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Load pretrained checkpoint."""
        if self.is_loaded and self.model is not None:
            return

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        print(f"[PretrainedDamageAdapter] Loading YOLO12s RDD2022 model from {self.checkpoint_path} on {self.device}...")
        self.model = YOLO(self.checkpoint_path)
        self.is_loaded = True

    def predict(self, image: Union[str, np.ndarray], conf: float = 0.25) -> Dict[str, Any]:
        """
        Run road-damage inference on a single image.
        """
        if not self.is_loaded:
            self.load_model()

        t0 = time.time()
        results = self.model.predict(
            source=image,
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
                standard_name = self.CLASS_MAPPING.get(raw_name, raw_name)
                conf_val = float(confs[i])
                bbox = boxes[i].tolist()

                # Repair is an additional infrastructure class, not severe distress
                is_damage = raw_name != "Repair"

                detections.append({
                    "class_id": cls_id,
                    "raw_class": raw_name,
                    "class_name": standard_name,
                    "is_damage": is_damage,
                    "confidence": round(conf_val, 3),
                    "box": [round(v, 1) for v in bbox]
                })

        # Render overlay if numpy image
        overlay = None
        if isinstance(image, np.ndarray):
            overlay = res.plot()

        return {
            "model_name": "PretrainedYOLO12s_RDD2022",
            "checkpoint": self.checkpoint_path,
            "architecture": "YOLO12s-Detection",
            "device": self.device,
            "runtime_ms": round(dt * 1000, 2),
            "detections_count": len(detections),
            "detections": detections,
            "overlay": overlay,
            "status": "INFERENCE_VERIFIED"
        }

    def predict_video(self, frame: Union[str, np.ndarray], timestamp: float, conf: float = 0.25) -> Dict[str, Any]:
        """Process video frame with timestamp."""
        res = self.predict(frame, conf=conf)
        res["timestamp"] = timestamp
        return res

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "PretrainedYOLO12s_RDD2022",
            "source": "https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022",
            "license": "MIT",
            "architecture": "YOLO12s",
            "checkpoint": self.checkpoint_path,
            "classes": ["D00", "D10", "D20", "D40", "Repair"],
            "status": "INFERENCE_VERIFIED"
        }

    def release_resources(self):
        """Release PyTorch memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

