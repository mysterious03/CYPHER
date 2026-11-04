"""
Surface Friction, Material, and Unevenness Multi-Task Architecture
Uses a shared lightweight backbone (MobileNetV3-Large or EfficientNetV2-S) with 3 decoupled prediction heads:
1. Friction Condition (Dry, Wet, Water-covered, Fresh snow, Melted snow, Ice-covered)
2. Road Material (Asphalt, Concrete, Gravel, Dirt, Mud, Loose stone)
3. Surface Unevenness (Smooth, Slightly uneven, Severely uneven / rutted, Deteriorated)

Strictly preserves scientific validity: visual predictions represent visual surface status,
never uncalibrated physical tyre-road friction coefficients.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple

FRICTION_CLASSES = [
    "Dry",
    "Wet",
    "Water-covered",
    "Fresh snow",
    "Melted snow",
    "Ice-covered"
]

MATERIAL_CLASSES = [
    "Asphalt",
    "Concrete",
    "Gravel",
    "Dirt",
    "Mud",
    "Loose stone"
]

UNEVENNESS_CLASSES = [
    "Smooth",
    "Slightly uneven",
    "Severely uneven",
    "Deteriorated"
]

class MultiTaskSurfaceClassifier(nn.Module):
    """
    Multi-task road surface classifier with shared feature backbone.
    Memory-efficient architecture designed for RTX 4050 6GB VRAM.
    """
    def __init__(self, backbone: str = "mobilenet_v3_large", pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "mobilenet_v3_large":
            base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = 960
        elif backbone == "efficientnet_v2_s":
            base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'mobilenet_v3_large' or 'efficientnet_v2_s'")

        # Decoupled Prediction Heads
        self.friction_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, len(FRICTION_CLASSES))
        )

        self.material_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, len(MATERIAL_CLASSES))
        )

        self.unevenness_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 128),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(128, len(UNEVENNESS_CLASSES))
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)

        return {
            "friction_logits": self.friction_head(feat),
            "material_logits": self.material_head(feat),
            "unevenness_logits": self.unevenness_head(feat)
        }

    def predict_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        return {
            "friction_probs": torch.softmax(logits["friction_logits"], dim=-1),
            "material_probs": torch.softmax(logits["material_logits"], dim=-1),
            "unevenness_probs": torch.softmax(logits["unevenness_logits"], dim=-1)
        }

