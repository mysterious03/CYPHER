"""
Cross-Condition Assessment and Environmental Robustness Module
Analyzes illumination, contrast, blur, shadow, and wet reflections to categorize
environmental driving conditions and report condition-stratified performance.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2

class ConditionAssessor:
    """
    Computes objective visual quality metrics and classifies image frames
    into operational road-scene conditions.
    """

    @staticmethod
    def assess_frame(
        image: np.ndarray,
        road_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Extracts photometric and geometric condition indicators.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # If road mask provided, focus analysis on the road surface
        if road_mask is not None and np.any(road_mask):
            road_pixels = gray[road_mask > 0]
        else:
            road_pixels = gray

        # 1. Illumination & Exposure
        mean_lum = float(np.mean(road_pixels))
        p10 = float(np.percentile(road_pixels, 10))
        p90 = float(np.percentile(road_pixels, 90))
        dynamic_range = p90 - p10

        # 2. Contrast (RMS contrast)
        rms_contrast = float(np.std(road_pixels))

        # 3. Blur / Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = float(laplacian.var())

        # 4. Glare and Specular Reflections (high intensity saturation in road surface)
        glare_ratio = float(np.count_nonzero(road_pixels > 240) / max(1, road_pixels.size))

        # 5. Shadow analysis (strong contrast bimodality on road surface)
        shadow_ratio = float(np.count_nonzero(road_pixels < 45) / max(1, road_pixels.size))

        # Condition Categorization
        primary_condition = "daylight_normal"
        tags = []

        if mean_lum < 55.0:
            primary_condition = "night_low_light"
            tags.append("low_illumination")
        elif glare_ratio > 0.08:
            primary_condition = "wet_rain_reflections"
            tags.append("specular_glare")
        elif shadow_ratio > 0.20 and dynamic_range > 120:
            primary_condition = "harsh_sunlight_shadows"
            tags.append("severe_shadows")
        elif sharpness_score < 80.0:
            primary_condition = "motion_blur"
            tags.append("blurred")

        return {
            "primary_condition": primary_condition,
            "condition_tags": tags,
            "metrics": {
                "mean_luminance": round(mean_lum, 2),
                "dynamic_range": round(dynamic_range, 2),
                "rms_contrast": round(rms_contrast, 2),
                "sharpness_laplacian": round(sharpness_score, 2),
                "glare_ratio": round(glare_ratio, 4),
                "shadow_ratio": round(shadow_ratio, 4)
            }
        }


class ConditionPerformanceTracker:
    """
    Maintains condition-stratified evaluation records to diagnose domain shift.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record_sample(
        self,
        image_id: str,
        condition: str,
        tp: int,
        fp: int,
        fn: int,
        details: Optional[Dict[str, Any]] = None
    ):
        self.records.append({
            "image_id": image_id,
            "condition": condition,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "details": details or {}
        })

    def generate_report(self) -> Dict[str, Any]:
        """
        Summarizes precision, recall, and F1 broken down by operational condition.
        """
        conditions = set(r["condition"] for r in self.records)
        summary = {}

        for cond in conditions:
            subset = [r for r in self.records if r["condition"] == cond]
            total_tp = sum(r["tp"] for r in subset)
            total_fp = sum(r["fp"] for r in subset)
            total_fn = sum(r["fn"] for r in subset)

            prec = total_tp / max(1, total_tp + total_fp)
            rec = total_tp / max(1, total_tp + total_fn)
            f1 = (2 * prec * rec) / max(1e-5, prec + rec)

            summary[cond] = {
                "sample_count": len(subset),
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4)
            }

        return summary

