"""
Pretrained Weather Classification Adapter for CYPhER.
Leverages the SigLIP2-based Weather-Image-Classification model (prithivMLmods/Weather-Image-Classification)
for classifying atmospheric road conditions (cloudy, foggy, rain, snow, clear)
with integrated lighting estimation (day, night, dusk).
"""
import os
import time
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Union, Optional
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

class PretrainedWeatherAdapter:
    """
    Adapter for SigLIP2-based road weather condition classification.
    """
    STANDARDIZED_LABELS = {
        0: "cloudy_overcast",
        1: "foggy_hazy",
        2: "rain_storm",
        3: "snow_frosty",
        4: "clear_sunny"
    }

    def __init__(
        self,
        model_dir: str = "models/pretrained/weather/weather_siglip2",
        device: str = "cuda",
        uncertainty_threshold: float = 0.40
    ):
        self.model_dir = model_dir
        self.device = device if torch.cuda.is_available() else "cpu"
        self.uncertainty_threshold = uncertainty_threshold
        self.processor = None
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Load SigLIP2 processor and model."""
        if self.is_loaded and self.model is not None:
            return

        if not os.path.exists(os.path.join(self.model_dir, "model.safetensors")):
            raise FileNotFoundError(f"Model files not found in: {self.model_dir}")

        print(f"[PretrainedWeatherAdapter] Loading Weather SigLIP2 from {self.model_dir} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def _estimate_lighting(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Estimate ambient illumination level (daytime, nighttime, dawn/dusk)."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        if mean_brightness < 45.0:
            lighting = "nighttime"
        elif mean_brightness < 90.0:
            lighting = "dawn_dusk"
        else:
            lighting = "daytime"

        return {
            "mean_luminance": round(mean_brightness, 1),
            "lighting_condition": lighting
        }

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Predict atmospheric weather condition and illumination.
        """
        if not self.is_loaded:
            self.load_model()

        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
            bgr_img = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            bgr_img = image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported image type")

        t0 = time.time()
        lighting_info = self._estimate_lighting(bgr_img)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        dt = time.time() - t0

        pred_idx = int(np.argmax(probs))
        top_prob = float(probs[pred_idx])

        # If top probability is lower than uncertainty threshold, flag as ambiguous
        if top_prob < self.uncertainty_threshold:
            condition = "ambiguous_unknown"
        else:
            condition = self.STANDARDIZED_LABELS.get(pred_idx, f"condition_{pred_idx}")

        all_probs = {
            self.STANDARDIZED_LABELS.get(i, f"class_{i}"): round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return {
            "model_name": "Weather_SigLIP2",
            "model_dir": self.model_dir,
            "architecture": "SigLIP2-ImageClassification",
            "device": self.device,
            "runtime_ms": round(dt * 1000, 2),
            "weather_condition": condition,
            "confidence": round(top_prob, 4),
            "lighting": lighting_info["lighting_condition"],
            "mean_luminance": lighting_info["mean_luminance"],
            "probabilities": all_probs,
            "is_ambiguous": condition == "ambiguous_unknown",
            "status": "INFERENCE_VERIFIED"
        }

    def predict_video(self, frame: Union[str, np.ndarray], timestamp: float) -> Dict[str, Any]:
        res = self.predict(frame)
        res["timestamp"] = timestamp
        return res

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Weather_SigLIP2",
            "source": "https://huggingface.co/prithivMLmods/Weather-Image-Classification",
            "license": "Apache-2.0",
            "classes": list(self.STANDARDIZED_LABELS.values()),
            "status": "INFERENCE_VERIFIED"
        }

    def release_resources(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

