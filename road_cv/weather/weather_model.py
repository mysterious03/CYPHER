"""
CYPhER V2.2 Multi-Task Weather & Illumination Classifier
Backbone: MobileNetV3-Small with dual task heads:
1. Environmental Weather Head: Clear, Rain, Fog, Snow, Overcast
2. Lighting Condition Head: Daytime, Nighttime, Dawn_Dusk
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Dict, Any, Tuple

WEATHER_CLASSES = ["Clear", "Rain", "Fog", "Snow", "Overcast"]
LIGHTING_CLASSES = ["Daytime", "Nighttime", "Dawn_Dusk"]

class MultiTaskWeatherClassifier(nn.Module):
    """
    Lightweight dual-head environmental perception model.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        
        in_dim = 576
        self.weather_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, len(WEATHER_CLASSES))
        )
        
        self.lighting_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, len(LIGHTING_CLASSES))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x)
        pooled = self.avgpool(f)
        flat = torch.flatten(pooled, 1)
        w_logits = self.weather_head(flat)
        l_logits = self.lighting_head(flat)
        return w_logits, l_logits

    def predict_image(self, img_tensor: torch.Tensor, device: str = "cuda") -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            w_logits, l_logits = self.forward(img_tensor)
            w_probs = torch.softmax(w_logits, dim=-1)[0].cpu().numpy()
            l_probs = torch.softmax(l_logits, dim=-1)[0].cpu().numpy()

            w_idx = int(np.argmax(w_probs))
            l_idx = int(np.argmax(l_probs))

            return {
                "weather": {
                    "class": WEATHER_CLASSES[w_idx],
                    "confidence": round(float(w_probs[w_idx]), 4),
                    "distribution": {c: round(float(p), 4) for c, p in zip(WEATHER_CLASSES, w_probs)}
                },
                "lighting": {
                    "class": LIGHTING_CLASSES[l_idx],
                    "confidence": round(float(l_probs[l_idx]), 4),
                    "distribution": {c: round(float(p), 4) for c, p in zip(LIGHTING_CLASSES, l_probs)}
                }
            }

if __name__ == "__main__":
    import numpy as np
    model = MultiTaskWeatherClassifier(pretrained=True)
    out_dir = Path("runs/weather_classifier_v22")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_weather_model.pt"
    
    torch.save({
        "state_dict": model.state_dict(),
        "weather_classes": WEATHER_CLASSES,
        "lighting_classes": LIGHTING_CLASSES,
        "architecture": "MobileNetV3-Small Multi-Task"
    }, str(ckpt_path))
    print(f"Exported weather model checkpoint to: {ckpt_path}")
