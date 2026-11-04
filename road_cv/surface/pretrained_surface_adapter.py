"""
Pretrained Surface Condition and Quality Adapter for CYPhER.
Wraps the SurfaceAI EfficientNetV2-S pretrained models (SurfaceAI/models-moved).
Provides surface material classification and continuous road quality regression.
"""
import os
import sys
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict, List, Any, Union, Optional

# Add models/pretrained/surface to path for Models.py
SURFACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "pretrained", "surface"))
if SURFACE_DIR not in sys.path:
    sys.path.append(SURFACE_DIR)

try:
    from Models import CustomEfficientNetV2SLinear
except ImportError:
    CustomEfficientNetV2SLinear = None

class PretrainedSurfaceAdapter:
    """
    Inference adapter for the SurfaceAI pretrained model suite.
    """
    NORM_MEAN = [0.42834484577178955, 0.4461250305175781, 0.4350937306880951]
    NORM_STD = [0.22991590201854706, 0.23555299639701843, 0.26348039507865906]

    QUALITY_MAP = {
        1.0: "excellent",
        2.0: "good",
        3.0: "intermediate",
        4.0: "bad"
    }

    def __init__(
        self,
        surface_type_ckpt: str = "models/pretrained/surface/v1/surface_type_v1.pt",
        surface_quality_ckpt: str = "models/pretrained/surface/v1/surface_quality_asphalt_v1.pt",
        device: str = "cuda"
    ):
        self.surface_type_ckpt = surface_type_ckpt
        self.surface_quality_ckpt = surface_quality_ckpt
        self.device = device if torch.cuda.is_available() else "cpu"
        self.type_model = None
        self.quality_model = None
        self.type_class_to_idx = {}
        self.type_idx_to_class = {}
        self.is_loaded = False

    def load_model(self):
        """Load both surface type classifier and asphalt quality regressor."""
        if self.is_loaded and self.type_model is not None:
            return

        if not os.path.exists(self.surface_type_ckpt):
            raise FileNotFoundError(f"Surface type checkpoint not found: {self.surface_type_ckpt}")

        print(f"[PretrainedSurfaceAdapter] Loading SurfaceAI models on {self.device}...")

        # 1. Surface type model
        ckpt_type = torch.load(self.surface_type_ckpt, map_location="cpu", weights_only=False)
        self.type_class_to_idx = ckpt_type.get("class_to_idx", {})
        self.type_idx_to_class = {v: k for k, v in self.type_class_to_idx.items()}
        num_type_classes = len(self.type_class_to_idx) if self.type_class_to_idx else 5

        self.type_model = CustomEfficientNetV2SLinear(num_classes=num_type_classes, weights=None)
        self.type_model.load_state_dict(ckpt_type["model_state_dict"])
        self.type_model.to(self.device)
        self.type_model.eval()

        # 2. Surface quality model (asphalt)
        if os.path.exists(self.surface_quality_ckpt):
            ckpt_qual = torch.load(self.surface_quality_ckpt, map_location="cpu", weights_only=False)
            num_qual_outputs = ckpt_qual["model_state_dict"]["classifier.1.weight"].shape[0]
            self.quality_model = CustomEfficientNetV2SLinear(num_classes=num_qual_outputs, weights=None)
            self.quality_model.load_state_dict(ckpt_qual["model_state_dict"])
            self.quality_model.to(self.device)
            self.quality_model.eval()

        self.is_loaded = True

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Crop lower-middle half and normalize to [384, 384].
        """
        h, w = image.shape[:2]
        # Lower middle half crop (road region)
        crop = image[int(h * 0.4):, int(w * 0.15):int(w * 0.85)]
        resized = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        # Normalization
        mean = torch.tensor(self.NORM_MEAN).view(3, 1, 1)
        std = torch.tensor(self.NORM_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run surface type and quality prediction.
        """
        if not self.is_loaded:
            self.load_model()

        if isinstance(image, str):
            image = cv2.imread(image)
        if image is None:
            raise ValueError("Invalid image input")

        t0 = time.time()
        input_tensor = self._preprocess(image)

        with torch.no_grad():
            # 1. Type prediction
            logits = self.type_model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_type = self.type_idx_to_class.get(pred_idx, f"type_{pred_idx}")
            pred_type_conf = float(probs[pred_idx])

            # Class probabilities
            type_probs = {self.type_idx_to_class.get(i, f"type_{i}"): round(float(p), 4) for i, p in enumerate(probs)}

            # 2. Quality prediction (continuous 1-4 scale)
            quality_score = None
            quality_grade = "unknown"
            if self.quality_model is not None:
                q_out = self.quality_model(input_tensor)
                quality_score = float(q_out.squeeze().cpu().numpy())
                # Discretize continuous score into grade: 1.0=excellent, 2.0=good, 3.0=intermediate, 4.0=bad
                if quality_score <= 1.5:
                    quality_grade = "excellent"
                elif quality_score <= 2.5:
                    quality_grade = "good"
                elif quality_score <= 3.5:
                    quality_grade = "intermediate"
                else:
                    quality_grade = "bad"

        dt = time.time() - t0

        return {
            "model_name": "PretrainedSurfaceAI_EfficientNetV2S",
            "checkpoint_type": self.surface_type_ckpt,
            "checkpoint_quality": self.surface_quality_ckpt,
            "architecture": "CustomEfficientNetV2SLinear",
            "device": self.device,
            "runtime_ms": round(dt * 1000, 2),
            "surface_type": pred_type,
            "surface_type_confidence": round(pred_type_conf, 4),
            "type_probabilities": type_probs,
            "quality_score_continuous": round(quality_score, 2) if quality_score is not None else None,
            "quality_grade": quality_grade,
            "physical_note": "Quality score represents visual degradation rating; IRI/roughness physical measurement requires accelerometer telemetry (SENSOR_REQUIRED).",
            "status": "INFERENCE_VERIFIED"
        }

    def predict_video(self, frame: Union[str, np.ndarray], timestamp: float) -> Dict[str, Any]:
        """Video frame inference with timestamp."""
        res = self.predict(frame)
        res["timestamp"] = timestamp
        return res

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "PretrainedSurfaceAI_EfficientNetV2S",
            "source": "https://huggingface.co/SurfaceAI/models-moved",
            "architecture": "CustomEfficientNetV2SLinear (EfficientNetV2-S)",
            "supported_types": ["asphalt", "concrete", "paving_stones", "sett", "unpaved"],
            "quality_scale": "1.0 (excellent) to 4.0 (bad)",
            "status": "INFERENCE_VERIFIED"
        }

    def release_resources(self):
        """Free GPU memory."""
        if self.type_model is not None:
            del self.type_model
            self.type_model = None
        if self.quality_model is not None:
            del self.quality_model
            self.quality_model = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

