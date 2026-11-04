"""
CYPhER V2.2 Dedicated Traffic Signal State Classifier
Classifies cropped traffic light boxes into discrete states:
0: red
1: amber_yellow
2: green
3: unknown_or_off
Combines a dedicated neural classifier with fallback to HSV vertical thirds rule.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn

SIGNAL_STATES = ["red", "amber_yellow", "green", "unknown_or_off"]

class TrafficSignalCNN(nn.Module):
    """
    Lightweight CNN for small traffic signal patch classification (32x64).
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x32

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)) # 64 * 4 = 256
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv(x)
        flat = torch.flatten(f, 1)
        return self.fc(flat)

class DedicatedTrafficSignalClassifier:
    """
    Hybrid neural + optical traffic signal state classifier.
    """
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.checkpoint_path = checkpoint_path or "runs/traffic_signals_v22/best_signal_model.pt"
        self.model = TrafficSignalCNN()
        
        if Path(self.checkpoint_path).exists():
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                if "state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["state_dict"])
                self.model.to(self.device).eval()
                self.loaded = True
            except Exception as e:
                print(f"[TrafficSignal WARNING] Failed loading {self.checkpoint_path}: {e}")
                self.loaded = False
        else:
            self.loaded = False

    def classify_crop(self, crop: np.ndarray) -> Dict[str, Any]:
        """
        Classifies state of a single cropped traffic light image patch.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return {"state": "unknown_or_off", "confidence": 0.0, "method": "rejected_invalid_crop"}

        # Method A: Neural Inference
        if self.loaded:
            resized = cv2.resize(crop, (32, 64))
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            t = t.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(t)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                if conf >= 0.40:
                    return {
                        "state": SIGNAL_STATES[idx],
                        "confidence": round(conf, 4),
                        "method": "neural_cnn",
                        "distribution": {s: round(float(p), 4) for s, p in zip(SIGNAL_STATES, probs)}
                    }

        # Method B: Optical HSV Vertical Thirds Fallback
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_crop, w_crop = crop.shape[:2]
        third = max(1, h_crop // 3)

        top = hsv[0:third, :]
        mid = hsv[third:2*third, :]
        bot = hsv[2*third:h_crop, :]

        # Red mask
        r1 = cv2.inRange(top, np.array([0, 70, 70]), np.array([10, 255, 255]))
        r2 = cv2.inRange(top, np.array([170, 70, 70]), np.array([180, 255, 255]))
        red_cnt = cv2.countNonZero(r1) + cv2.countNonZero(r2)

        # Yellow mask
        y_mask = cv2.inRange(mid, np.array([15, 70, 70]), np.array([35, 255, 255]))
        yellow_cnt = cv2.countNonZero(y_mask)

        # Green mask
        g_mask = cv2.inRange(bot, np.array([40, 70, 70]), np.array([90, 255, 255]))
        green_cnt = cv2.countNonZero(g_mask)

        counts = {"red": red_cnt, "amber_yellow": yellow_cnt, "green": green_cnt}
        max_color = max(counts, key=counts.get)
        max_val = counts[max_color]

        if max_val < 8:
            return {"state": "unknown_or_off", "confidence": 0.50, "method": "optical_hsv_fallback"}

        total = sum(counts.values())
        return {
            "state": max_color,
            "confidence": round(float(max_val / max(1, total)), 3),
            "method": "optical_hsv_fallback",
            "pixel_counts": counts
        }

if __name__ == "__main__":
    out_dir = Path("runs/traffic_signals_v22")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_signal_model.pt"

    model = TrafficSignalCNN()
    torch.save({
        "state_dict": model.state_dict(),
        "classes": SIGNAL_STATES,
        "architecture": "TrafficSignalCNN"
    }, str(ckpt_path))
    print(f"Exported traffic signal model to: {ckpt_path}")
