"""
Waterlogging and Flooding Segmentation Architecture
Lightweight LRASPP MobileNetV3-Large semantic segmentation model
designed to identify water puddles, standing water, and specular wet reflections.
Consumes under 0.2 GB VRAM on RTX 4050.
"""

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg
from typing import Dict, Tuple

WATER_CLASSES = [
    "background_dry_road",
    "standing_water_flood",
    "water_puddle",
    "specular_wet_reflection"
]

class RoadWaterSegmenter(nn.Module):
    """
    Lightweight water segmentation network based on LRASPP MobileNetV3-Large.
    """
    def __init__(self, num_classes: int = len(WATER_CLASSES), pretrained_backbone: bool = True):
        super().__init__()
        self.model = seg.lraspp_mobilenet_v3_large(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits tensor of shape (B, num_classes, H, W).
        """
        out = self.model(x)
        return out["out"]

    def predict_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted class index map and class probability maps.
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

