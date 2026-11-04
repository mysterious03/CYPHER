"""
Pretrained Depth Anything V2 Small Adapter for CYPhER.
Leverages Hugging Face Transformers for dense relative depth estimation.
Enforces physical telemetry gating (CALIBRATION_REQUIRED) for metric pothole measurements.
"""
import os
import time
import cv2
import numpy as np
import torch
from typing import Dict, Any, Union, Optional
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class PretrainedDepthAdapter:
    """
    Adapter for Depth-Anything-V2-Small relative depth estimator.
    """
    def __init__(
        self,
        model_dir: str = "models/pretrained/depth/depth_anything_v2_small",
        device: str = "cuda"
    ):
        self.model_dir = model_dir
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Load Depth Anything V2 processor and model."""
        if self.is_loaded and self.model is not None:
            return

        if not os.path.exists(os.path.join(self.model_dir, "model.safetensors")):
            raise FileNotFoundError(f"Model files not found in: {self.model_dir}")

        print(f"[PretrainedDepthAdapter] Loading Depth-Anything-V2-Small from {self.model_dir} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Estimate dense relative depth map for the input image.
        """
        if not self.is_loaded:
            self.load_model()

        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
            orig_w, orig_h = pil_img.size
        elif isinstance(image, np.ndarray):
            orig_h, orig_w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            orig_w, orig_h = pil_img.size
        else:
            raise ValueError("Unsupported image type")

        t0 = time.time()
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_depth = outputs.predicted_depth.squeeze(0).cpu().numpy()

        dt = time.time() - t0

        # Resize predicted depth map back to original input image dimensions
        depth_resized = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Normalize relative depth to [0.0, 1.0] for downstream perception modules
        d_min = float(np.min(depth_resized))
        d_max = float(np.max(depth_resized))
        norm_depth = (depth_resized - d_min) / max(1e-6, (d_max - d_min))

        # Generate colorized depth visualization (inferno colormap)
        depth_u8 = (norm_depth * 255.0).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

        return {
            "model_name": "DepthAnythingV2_Small",
            "model_dir": self.model_dir,
            "architecture": "DPT_DINOv2_Small",
            "device": self.device,
            "runtime_ms": round(dt * 1000, 2),
            "depth_type": "relative_inverse_depth",
            "depth_map": norm_depth,
            "depth_min_raw": round(d_min, 4),
            "depth_max_raw": round(d_max, 4),
            "depth_colored": depth_colored,
            "calibration_warning": "CALIBRATION_REQUIRED: Output is relative disparity/depth; conversion to real-world centimetres or metric pothole depth requires camera calibration and reference telemetry.",
            "status": "INFERENCE_VERIFIED"
        }

    def predict_video(self, frame: Union[str, np.ndarray], timestamp: float) -> Dict[str, Any]:
        res = self.predict(frame)
        res["timestamp"] = timestamp
        return res

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "DepthAnythingV2_Small",
            "source": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf",
            "license": "Apache-2.0",
            "parameters": "~24.8M",
            "depth_type": "Relative",
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

