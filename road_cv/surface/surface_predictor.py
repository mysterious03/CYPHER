"""
Road Surface Condition Predictor & Scientific Inference Interface
Runs multi-task inference for friction, material, and unevenness.
Separates visual surface conditions from physical tyre-road friction coefficients.
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional
import torchvision.transforms as T

from road_cv.surface.friction_classifier import MultiTaskSurfaceClassifier, FRICTION_CLASSES, MATERIAL_CLASSES, UNEVENNESS_CLASSES

class RoadSurfacePredictor:
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiTaskSurfaceClassifier(backbone="mobilenet_v3_large", pretrained=False)

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded surface classifier from: {checkpoint_path}")
        else:
            print("Surface predictor initialized without weights (inference will use base weights)")

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_patch(self, patch: np.ndarray) -> Dict:
        """
        Predicts surface properties for a single RGB patch.
        """
        if patch is None or patch.size == 0:
            raise ValueError("Input patch is empty")

        if len(patch.shape) == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        elif patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        tensor = self.transform(patch).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.model.predict_probabilities(tensor)

        f_probs = probs["friction_probs"][0].cpu().numpy()
        m_probs = probs["material_probs"][0].cpu().numpy()
        u_probs = probs["unevenness_probs"][0].cpu().numpy()

        best_f_idx = int(np.argmax(f_probs))
        best_m_idx = int(np.argmax(m_probs))
        best_u_idx = int(np.argmax(u_probs))

        return {
            "visual_surface_condition": {
                "class_name": FRICTION_CLASSES[best_f_idx],
                "confidence": float(f_probs[best_f_idx]),
                "all_probabilities": {FRICTION_CLASSES[i]: float(f_probs[i]) for i in range(len(FRICTION_CLASSES))}
            },
            "road_material": {
                "class_name": MATERIAL_CLASSES[best_m_idx],
                "confidence": float(m_probs[best_m_idx]),
                "all_probabilities": {MATERIAL_CLASSES[i]: float(m_probs[i]) for i in range(len(MATERIAL_CLASSES))}
            },
            "visual_unevenness": {
                "class_name": UNEVENNESS_CLASSES[best_u_idx],
                "confidence": float(u_probs[best_u_idx]),
                "all_probabilities": {UNEVENNESS_CLASSES[i]: float(u_probs[i]) for i in range(len(UNEVENNESS_CLASSES))}
            },
            # Explicit physical separation as required by scientific guidelines:
            "estimated_tire_road_friction": None,
            "physical_friction_disclaimer": "Physical tyre-road friction coefficient (mu) requires vehicle dynamics (wheel slip, lateral/longitudinal acceleration). RGB predictions indicate visual surface status only.",
            "iri_roughness_disclaimer": "Visual unevenness category is not a certified International Roughness Index (IRI) measurement."
        }

