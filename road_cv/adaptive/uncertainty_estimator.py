"""
CYPhER Adaptive Perception: Calibrated Uncertainty Estimator
Estimates predictive uncertainty from detector logits, decision margins,
scale degradation factors, and photometric image quality.
"""

from typing import Dict, Any, List, Optional
import numpy as np

class CalibratedUncertaintyEstimator:
    """
    Computes an explainable uncertainty metric U in [0.0, 1.0] for object detections.
    Higher values trigger high-resolution crop refinement or multi-pass verification.
    """

    def __init__(
        self,
        entropy_weight: float = 0.35,
        margin_weight: float = 0.35,
        scale_weight: float = 0.20,
        photometric_weight: float = 0.10
    ):
        self.entropy_weight = entropy_weight
        self.margin_weight = margin_weight
        self.scale_weight = scale_weight
        self.photometric_weight = photometric_weight

    def estimate_uncertainty(
        self,
        box: List[float],
        confidence: float,
        image_shape: tuple,
        photometric_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Estimates multi-factor predictive uncertainty.
        
        Args:
            box: [x1, y1, x2, y2]
            confidence: detector sigmoid output in [0.0, 1.0]
            image_shape: (H, W)
            photometric_metrics: dict from ConditionAssessor
        """
        # 1. Decision Margin Uncertainty: Maximum at conf=0.50, drops to 0 at conf=0.0 or 1.0
        # U_margin = 1.0 - 2.0 * |conf - 0.5|
        margin_uncertainty = 1.0 - 2.0 * abs(confidence - 0.50)
        margin_uncertainty = float(np.clip(margin_uncertainty, 0.0, 1.0))

        # 2. Binary Entropy
        p = np.clip(confidence, 1e-6, 1.0 - 1e-6)
        binary_entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        entropy_uncertainty = float(np.clip(binary_entropy, 0.0, 1.0))

        # 3. Scale Uncertainty: Defects smaller than 32x32 pixels have high scale ambiguity
        w_px = max(1.0, box[2] - box[0])
        h_px = max(1.0, box[3] - box[1])
        area_px = w_px * h_px
        ref_area = 32.0 * 32.0
        scale_uncertainty = float(np.clip(1.0 - (area_px / ref_area), 0.0, 1.0))

        # 4. Photometric Degradation Factor
        photo_penalty = 0.0
        if photometric_metrics:
            # Low contrast penalty
            contrast = photometric_metrics.get("contrast_rms", 50.0)
            if contrast < 30.0:
                photo_penalty += 0.40
            # Low sharpness / blur penalty
            sharpness = photometric_metrics.get("sharpness_laplacian", 200.0)
            if sharpness < 80.0:
                photo_penalty += 0.40
            # Glare / overexposure penalty
            glare = photometric_metrics.get("glare_ratio", 0.0)
            if glare > 0.15:
                photo_penalty += 0.20
        photo_uncertainty = float(np.clip(photo_penalty, 0.0, 1.0))

        # Composite Calibrated Uncertainty
        composite_u = (
            self.margin_weight * margin_uncertainty +
            self.entropy_weight * entropy_uncertainty +
            self.scale_weight * scale_uncertainty +
            self.photometric_weight * photo_uncertainty
        )
        composite_u = round(float(np.clip(composite_u, 0.0, 1.0)), 4)

        return {
            "uncertainty_score": composite_u,
            "is_uncertain": composite_u >= 0.45,
            "components": {
                "margin_uncertainty": round(margin_uncertainty, 3),
                "entropy_uncertainty": round(entropy_uncertainty, 3),
                "scale_uncertainty": round(scale_uncertainty, 3),
                "photometric_uncertainty": round(photo_uncertainty, 3)
            }
        }
