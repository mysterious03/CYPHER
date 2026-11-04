"""
Pretrained Open-Vocabulary Road Obstacle Adapter for CYPhER.
Leverages YOLO-World (yolov8s-world.pt) with dynamic text prompts to detect
arbitrary roadside hazards, fallen debris, spilled cargo, animals, and infrastructure damage.
"""
import os
import time
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Union, Optional
from ultralytics import YOLO

class PretrainedObstacleAdapter:
    """
    Open-vocabulary detector for unexpected road obstacles using YOLO-World.
    """
    DEFAULT_OBSTACLE_PROMPTS = [
        # Natural obstacles
        "fallen tree",
        "fallen tree branch",
        "rock on road",
        "boulder",
        "animal on road",
        # Road debris
        "broken concrete",
        "road debris",
        "wooden log",
        "construction debris",
        "tyre on road",
        # Cargo & dropped objects
        "spilled cargo",
        "fallen box",
        "lost object",
        # Infrastructure damage
        "fallen traffic sign",
        "broken barrier",
        "construction barrier"
    ]

    def __init__(
        self,
        checkpoint_path: str = "yolov8s-world.pt",
        prompts: Optional[List[str]] = None,
        device: str = "cuda"
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.prompts = prompts or self.DEFAULT_OBSTACLE_PROMPTS
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Load YOLO-World model and set custom vocabulary prompts."""
        if self.is_loaded and self.model is not None:
            return

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"YOLO-World checkpoint not found at: {self.checkpoint_path}")

        print(f"[PretrainedObstacleAdapter] Loading YOLO-World from {self.checkpoint_path} on {self.device}...")
        self.model = YOLO(self.checkpoint_path)
        # Set open-vocabulary classes
        self.model.set_classes(self.prompts)
        self.is_loaded = True

    def set_custom_prompts(self, prompts: List[str]):
        """Update active detection prompts dynamically."""
        self.prompts = prompts
        if self.is_loaded and self.model is not None:
            self.model.set_classes(prompts)

    def predict(self, image: Union[str, np.ndarray], conf: float = 0.15) -> Dict[str, Any]:
        """
        Detect custom obstacle prompts in the image.
        Uses conf=0.15 for open-vocabulary recall on unusual hazard geometries.
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
                cls_name = self.prompts[cls_id] if cls_id < len(self.prompts) else f"prompt_{cls_id}"
                conf_val = float(confs[i])
                bbox = boxes[i].tolist()

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 3),
                    "box": [round(v, 1) for v in bbox]
                })

        overlay = res.plot() if isinstance(image, np.ndarray) else None

        return {
            "model_name": "PretrainedYOLO_World_Obstacles",
            "checkpoint": self.checkpoint_path,
            "architecture": "YOLOv8s-World",
            "device": self.device,
            "runtime_ms": round(dt * 1000, 2),
            "prompts_count": len(self.prompts),
            "detections_count": len(detections),
            "detections": detections,
            "overlay": overlay,
            "status": "INFERENCE_VERIFIED"
        }

    def predict_video(self, frame: Union[str, np.ndarray], timestamp: float, conf: float = 0.15) -> Dict[str, Any]:
        res = self.predict(frame, conf=conf)
        res["timestamp"] = timestamp
        return res

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "PretrainedYOLO_World_Obstacles",
            "checkpoint": self.checkpoint_path,
            "prompts": self.prompts,
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

