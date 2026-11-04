"""
Pretrained Road Wetness Segmentation Adapter for CYPhER.
Wraps the pretrained Lithuanian YOLOv8n road wetness segmentation model
(martinkenas/Road-Segmentation-for-type-wetness-and-roughness).
Outputs dry road / wet road masks, wet-area ratio, and visual overlay.
"""
import os
import time
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Union, Optional
from ultralytics import YOLO

class PretrainedWetnessAdapter:
    """
    Adapter for the YOLOv8n road wetness segmenter.
    Maps native Lithuanian classes ('sausas'=dry, 'slapias'=wet) to standardized terms.
    """
    CLASS_MAP = {
        0: "dry_road",   # 'sausas'
        1: "wet_road"    # 'slapias'
    }

    def __init__(self, checkpoint_path: str = "models/pretrained/water/road_wetness_yolov8n_seg.pt", device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Load YOLOv8n segmentation checkpoint."""
        if self.is_loaded and self.model is not None:
            return

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        print(f"[PretrainedWetnessAdapter] Loading wetness model from {self.checkpoint_path} on {self.device}...")
        self.model = YOLO(self.checkpoint_path)
        self.is_loaded = True

    def predict(self, image: Union[str, np.ndarray], conf: float = 0.05) -> Dict[str, Any]:
        """
        Run wetness segmentation on an image.
        Uses conf=0.05 default because specialized segmentation models may yield lower peak confidence on cross-domain imagery.
        """
        if not self.is_loaded:
            self.load_model()

        if isinstance(image, str):
            img_bgr = cv2.imread(image)
        else:
            img_bgr = image

        if img_bgr is None:
            raise ValueError("Invalid image input")

        h, w = img_bgr.shape[:2]
        t0 = time.time()
        results = self.model.predict(
            source=img_bgr,
            conf=conf,
            device=self.device,
            verbose=False
        )
        dt = time.time() - t0
        res = results[0]

        dry_mask = np.zeros((h, w), dtype=np.uint8)
        wet_mask = np.zeros((h, w), dtype=np.uint8)

        detections = []
        max_conf_class = "dry_road"
        max_conf = 0.0

        if res.masks is not None and len(res.masks) > 0:
            masks = res.masks.data.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()

            for i in range(len(clss)):
                cls_id = int(clss[i])
                std_cls = self.CLASS_MAP.get(cls_id, f"class_{cls_id}")
                conf_val = float(confs[i])
                box = boxes[i].tolist()

                # Resize mask to original image size
                m_raw = masks[i]
                m_resized = cv2.resize(m_raw, (w, h), interpolation=cv2.INTER_LINEAR)
                binary_m = (m_resized > 0.5).astype(np.uint8)

                if cls_id == 1:  # wet
                    wet_mask = np.maximum(wet_mask, binary_m)
                else:            # dry
                    dry_mask = np.maximum(dry_mask, binary_m)

                if conf_val > max_conf:
                    max_conf = conf_val
                    max_conf_class = std_cls

                detections.append({
                    "class_id": cls_id,
                    "native_class": res.names.get(cls_id, ""),
                    "class_name": std_cls,
                    "confidence": round(conf_val, 3),
                    "box": [round(v, 1) for v in box]
                })

        total_pixels = h * w
        dry_pixels = int(np.sum(dry_mask))
        wet_pixels = int(np.sum(wet_mask))
        wet_area_ratio = round(wet_pixels / max(1, total_pixels), 4)
        dry_area_ratio = round(dry_pixels / max(1, total_pixels), 4)

        # Composite segmentation overlay
        overlay = img_bgr.copy()
        if dry_pixels > 0 or wet_pixels > 0:
            color_mask = np.zeros_like(img_bgr)
            color_mask[dry_mask > 0] = [0, 200, 0]      # Green for dry
            color_mask[wet_mask > 0] = [255, 100, 0]    # Blue for wet
            overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

        return {
            "model_name": "PretrainedWetness_YOLOv8n_Seg",
            "checkpoint": self.checkpoint_path,
            "architecture": "YOLOv8n-Seg",
            "device": self.device,
            "runtime_ms": round(dt * 1000, 2),
            "predicted_condition": max_conf_class,
            "confidence": round(max_conf, 3),
            "wet_area_ratio": wet_area_ratio,
            "dry_area_ratio": dry_area_ratio,
            "has_wet_regions": wet_pixels > 0,
            "dry_mask": dry_mask,
            "wet_mask": wet_mask,
            "overlay": overlay,
            "limitations_note": "Detects visual road wetness vs dry surface; does not measure water depth or hydroplaning risk without calibrated physical sensors.",
            "status": "PRETRAINED_NOT_VALIDATED"
        }

    def predict_video(self, frame: Union[str, np.ndarray], timestamp: float, conf: float = 0.05) -> Dict[str, Any]:
        """Video frame inference with timestamp."""
        res = self.predict(frame, conf=conf)
        res["timestamp"] = timestamp
        return res

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "PretrainedWetness_YOLOv8n_Seg",
            "source": "https://github.com/martinkenas/Road-Segmentation-for-type-wetness-and-roughness",
            "classes": ["dry_road (sausas)", "wet_road (slapias)"],
            "dataset": "1,387 annotated road images",
            "status": "PRETRAINED_NOT_VALIDATED"
        }

    def release_resources(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

