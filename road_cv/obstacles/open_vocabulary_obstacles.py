"""
CYPhER V2.2 Open-Vocabulary Road Obstacle & Debris Perception
Employs YOLO-World zero-shot text-prompted detection conditioned on customizable hazard vocabulary:
- fallen tree / branch
- rock / boulder
- road debris / broken concrete
- tyre / shredded tire on road
- spilled cargo / construction material
- road barrier / traffic cone
Fuses with road_mask to verify in-corridor positioning without hard-dropping edge hazards.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import torch
from ultralytics import YOLO

DEFAULT_OBSTACLE_PROMPTS = [
    "fallen tree",
    "large rock",
    "boulder on road",
    "road debris",
    "broken concrete",
    "spilled cargo",
    "tyre on road",
    "construction material",
    "traffic barrier",
    "road obstruction"
]

class OpenVocabularyObstacleDetector:
    """
    Zero-Shot open-vocabulary object detector driven by natural language prompts.
    """
    def __init__(
        self,
        model_path: str = "yolov8s-world.pt",
        prompts: Optional[List[str]] = None,
        conf_threshold: float = 0.20,
        device: str = "cuda"
    ):
        self.device = 0 if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.prompts = prompts or DEFAULT_OBSTACLE_PROMPTS
        
        self.model = YOLO(self.model_path)
        try:
            self.model.set_classes(self.prompts)
            self.loaded = True
        except Exception as e:
            print(f"[OpenVocabularyObstacle WARNING] Failed setting custom classes: {e}")
            self.loaded = False

    def predict(
        self,
        frame: np.ndarray,
        road_mask: Optional[np.ndarray] = None,
        custom_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs open-vocabulary detection and computes road-corridor overlap.
        """
        if custom_prompts and custom_prompts != self.prompts:
            self.prompts = custom_prompts
            self.model.set_classes(self.prompts)

        h, w = frame.shape[:2]
        res = self.model.predict(frame, imgsz=640, conf=self.conf_threshold, device=self.device, verbose=False)[0]
        
        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for b, c, cf in zip(boxes, clss, confs):
                x1, y1, x2, y2 = b
                cx = np.clip((x1 + x2) // 2, 0, w - 1)
                cy = np.clip((y1 + y2) // 2, 0, h - 1)

                # Check road-corridor spatial prior
                on_road = True
                if road_mask is not None:
                    on_road = bool(road_mask[cy, cx] > 0)

                prompt_name = res.names.get(int(c), self.prompts[int(c)] if int(c) < len(self.prompts) else "obstacle")

                detections.append({
                    "category": prompt_name,
                    "confidence": round(float(cf), 4),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "in_drivable_corridor": on_road
                })

        return {
            "status": "success",
            "method": "zero_shot_yolo_world",
            "active_prompts": self.prompts,
            "detected_count": len(detections),
            "obstacles": detections
        }

if __name__ == "__main__":
    detector = OpenVocabularyObstacleDetector()
    frame = cv2.imread("Road.jpg")
    if frame is not None:
        out = detector.predict(frame)
        print(f"Detected {out['detected_count']} obstacle candidates:")
        for o in out["obstacles"][:5]:
            print(" ", o)
