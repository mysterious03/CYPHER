"""
Pretrained Multiclass Water-Puddle Segmentation Adapter for CYPhER.
Audits the Shrinidhi1/Multiclass-Semantic-Segmentation-for-Road-Surface-Detection repository.
Records CHECKPOINT_UNAVAILABLE honestly if only training notebooks are provided without downloadable weights.
"""
import os
import numpy as np
from typing import Dict, Any, Union

class PretrainedPuddleAdapter:
    """
    Adapter representing the external multiclass road puddle U-Net model candidate.
    """
    def __init__(self, checkpoint_path: str = "models/pretrained/water/road_surface_multiclass_unet.pt"):
        self.checkpoint_path = checkpoint_path
        self.is_loaded = False
        self.status = "CHECKPOINT_UNAVAILABLE"
        self.reason = (
            "Audited Shrinidhi1/Multiclass-Semantic-Segmentation-for-Road-Surface-Detection. "
            "The repository's Models/ directory contains Jupyter training notebooks (1. Unet.ipynb, 2. UnetBackbone.ipynb) "
            "but no serialized checkpoint weights (.pt/.pth) are hosted or released. "
            "Per CYPhER technical integrity directives, no fake weights are fabricated."
        )

    def load_model(self):
        """Report honest status of checkpoint availability."""
        if not os.path.exists(self.checkpoint_path):
            self.status = "CHECKPOINT_UNAVAILABLE"
            return False
        self.is_loaded = True
        return True

    def predict(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Return unavailable status with explicit fallback advice."""
        return {
            "model_name": "MulticlassWaterPuddle_UNet",
            "checkpoint": self.checkpoint_path,
            "status": "CHECKPOINT_UNAVAILABLE",
            "reason": self.reason,
            "has_puddle": False,
            "puddle_mask": None,
            "puddle_area_ratio": 0.0,
            "fallback": "Use CYPhER native water perception / YOLOv8n wetness segmenter."
        }

    def predict_video(self, frame: Union[str, np.ndarray], timestamp: float) -> Dict[str, Any]:
        res = self.predict(frame)
        res["timestamp"] = timestamp
        return res

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "MulticlassWaterPuddle_UNet",
            "source": "https://github.com/Shrinidhi1/Multiclass-Semantic-Segmentation-for-Road-Surface-Detection",
            "target_classes": ["Asphalt", "Paved road", "Water puddle", "Pothole", "Cracks"],
            "status": "CHECKPOINT_UNAVAILABLE",
            "audit_note": self.reason
        }

    def release_resources(self):
        pass

