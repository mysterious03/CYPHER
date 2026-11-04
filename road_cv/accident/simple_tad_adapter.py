"""
Simple-TAD VideoMAE Temporal Traffic Anomaly / Accident Adapter for CYPhER.
Leverages VideoMAE ViT-Small fine-tuned on DoTA (Detection of Traffic Anomalies)
to evaluate sliding temporal windows (16 frames) on video streams.
"""
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Union, Optional

ACCIDENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "pretrained", "accident"))
if ACCIDENT_DIR not in sys.path:
    sys.path.append(ACCIDENT_DIR)

try:
    import simple_tad_model
except ImportError:
    simple_tad_model = None

class SimpleTADAdapter:
    """
    Adapter for the VideoMAE ViT-S traffic anomaly detector.
    Processes 16-frame spatio-temporal video clips.
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        checkpoint_path: str = "models/pretrained/accident/simpletad_ft-dota_vm1-s_auroc.pth",
        device: str = "cuda",
        anomaly_threshold: float = 0.50
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.anomaly_threshold = anomaly_threshold
        self.model = None
        self.is_loaded = False
        # 16-frame sliding buffer
        self.frame_buffer: List[np.ndarray] = []
        self.timestamp_buffer: List[float] = []

    def load_model(self):
        """Load VideoMAE ViT-S checkpoint."""
        if self.is_loaded and self.model is not None:
            return

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        if simple_tad_model is None:
            raise ImportError("simple_tad_model module could not be loaded")

        print(f"[SimpleTADAdapter] Loading VideoMAE ViT-S from {self.checkpoint_path} on {self.device}...")
        self.model = simple_tad_model.get_video_vit_small(with_flash=False)
        sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def _prepare_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Resize to 224x224, convert to RGB, and normalize."""
        resized = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Transpose to (C, H, W)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
        # Normalize
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std

    def reset_buffer(self):
        """Clear temporal frame buffer."""
        self.frame_buffer.clear()
        self.timestamp_buffer.clear()

    def predict(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Enforce temporal integrity: an isolated static image cannot constitute a video traffic anomaly.
        """
        return {
            "model_name": "SimpleTAD_VideoMAE_ViTS",
            "checkpoint": self.checkpoint_path,
            "status": "IMAGE_INPUT_UNSUPPORTED",
            "warning": "Temporal anomaly and accident recognition requires sequential video frames (minimum 16 frames). Single static image input cannot verify a temporal collision.",
            "is_anomaly": False,
            "anomaly_score": 0.0
        }

    def predict_video(self, frame: Union[str, np.ndarray], timestamp: float) -> Dict[str, Any]:
        """
        Ingest a video frame into the 16-frame sliding window and evaluate temporal anomaly score.
        """
        if not self.is_loaded:
            self.load_model()

        if isinstance(frame, str):
            frame_bgr = cv2.imread(frame)
        else:
            frame_bgr = frame

        if frame_bgr is None:
            raise ValueError("Invalid frame input")

        self.frame_buffer.append(frame_bgr)
        self.timestamp_buffer.append(timestamp)

        # Maintain 16-frame sliding window
        if len(self.frame_buffer) > 16:
            self.frame_buffer.pop(0)
            self.timestamp_buffer.pop(0)

        # If buffer not yet filled, return accumulating status
        if len(self.frame_buffer) < 16:
            return {
                "model_name": "SimpleTAD_VideoMAE_ViTS",
                "checkpoint": self.checkpoint_path,
                "status": "ACCUMULATING_FRAMES",
                "frames_buffered": len(self.frame_buffer),
                "frames_required": 16,
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "timestamp": timestamp
            }

        t0 = time.time()
        # Construct spatio-temporal tensor: (1, 3, 16, 224, 224)
        clip_tensor = torch.zeros((1, 3, 16, 224, 224), dtype=torch.float32)
        for i, f in enumerate(self.frame_buffer):
            clip_tensor[0, :, i, :, :] = self._prepare_frame(f)

        clip_tensor = clip_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(clip_tensor)
            # Output is [1, 2] where index 1 corresponds to anomaly/accident event probability
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            anomaly_score = float(probs[1]) if len(probs) > 1 else float(probs[0])

        dt = time.time() - t0
        is_anomaly = anomaly_score >= self.anomaly_threshold

        return {
            "model_name": "SimpleTAD_VideoMAE_ViTS",
            "checkpoint": self.checkpoint_path,
            "architecture": "VideoMAE_ViTS",
            "device": self.device,
            "runtime_ms": round(dt * 1000, 2),
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": is_anomaly,
            "threshold": self.anomaly_threshold,
            "window_start_time": self.timestamp_buffer[0],
            "window_end_time": self.timestamp_buffer[-1],
            "note": "Anomaly score reflects spatio-temporal deviation on DoTA benchmark; not a confirmed physical collision probability.",
            "status": "INFERENCE_VERIFIED"
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "SimpleTAD_VideoMAE_ViTS",
            "source": "https://huggingface.co/tue-mps/simple-tad",
            "dataset": "DoTA (Detection of Traffic Anomalies)",
            "license": "CC BY-NC 4.0",
            "input_shape": "(1, 3, 16, 224, 224)",
            "status": "INFERENCE_VERIFIED"
        }

    def release_resources(self):
        """Free GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        self.reset_buffer()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

