"""
Scale-Adaptive, Confidence-Triggered Refinement Module
Coordinates high-resolution local crops on small or uncertain defect candidates,
merges detections using Weighted Box Fusion, and manages polygon-level mask structures.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from .edge_preservation import EdgePreservingEnhancer

class AdaptiveRefiner:
    """
    Inference controller that identifies small/uncertain defect regions,
    dispatches targeted high-resolution crops within a bounded compute budget,
    and merges multi-scale predictions.
    """

    def __init__(
        self,
        max_crops_per_image: int = 4,
        crop_size: int = 640,
        uncertainty_conf_range: Tuple[float, float] = (0.15, 0.50),
        small_defect_area_ratio: float = 0.02, # box area < 2% of frame area
        iou_merge_threshold: float = 0.50
    ):
        self.max_crops_per_image = max_crops_per_image
        self.crop_size = crop_size
        self.uncertainty_conf_range = uncertainty_conf_range
        self.small_defect_area_ratio = small_defect_area_ratio
        self.iou_merge_threshold = iou_merge_threshold
        self.enhancer = EdgePreservingEnhancer()

    def select_refinement_regions(
        self,
        image_shape: Tuple[int, int],
        full_frame_detections: List[Dict[str, Any]],
        road_mask: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Filters and prioritizes regions that justify high-resolution crop inference.
        Prioritizes:
        1. Small defect bounding boxes
        2. Predictions within the uncertainty confidence window
        3. Detections falling inside the drivable road mask
        """
        h, w = image_shape[:2]
        total_frame_area = float(h * w)
        candidates = []

        for det in full_frame_detections:
            box = det["box"] # [x1, y1, x2, y2]
            conf = float(det.get("confidence", 0.0))
            cls_name = det.get("class_name", "")

            box_area = (box[2] - box[0]) * (box[3] - box[1])
            area_ratio = box_area / total_frame_area

            is_small = area_ratio <= self.small_defect_area_ratio
            is_uncertain = (self.uncertainty_conf_range[0] <= conf <= self.uncertainty_conf_range[1])
            is_thin_crack = cls_name in ["D00", "D10", "longitudinal_crack", "transverse_crack"]

            # Road mask check
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            on_road = True
            if road_mask is not None and 0 <= cy < h and 0 <= cx < w:
                on_road = bool(road_mask[cy, cx])

            if (is_small or is_uncertain or is_thin_crack) and on_road:
                # Priority score: smaller area and medium confidence get higher priority
                priority = (1.0 - area_ratio) * 0.6 + (1.0 - abs(conf - 0.35)) * 0.4
                candidates.append({
                    "center": (cx, cy),
                    "box": box,
                    "priority": priority,
                    "is_thin_crack": is_thin_crack
                })

        # Sort by priority and cap at max_crops_per_image
        candidates.sort(key=lambda x: x["priority"], reverse=True)
        selected = candidates[:self.max_crops_per_image]

        crops = []
        half_crop = self.crop_size // 2

        for item in selected:
            cx, cy = item["center"]
            x1 = max(0, cx - half_crop)
            y1 = max(0, cy - half_crop)
            x2 = min(w, x1 + self.crop_size)
            y2 = min(h, y1 + self.crop_size)

            # Adjust x1, y1 if near image boundaries to keep crop square
            if x2 - x1 < self.crop_size and w >= self.crop_size:
                x1 = max(0, x2 - self.crop_size)
            if y2 - y1 < self.crop_size and h >= self.crop_size:
                y1 = max(0, y2 - self.crop_size)

            crops.append({
                "crop_box": [x1, y1, x2, y2],
                "is_thin_crack": item["is_thin_crack"]
            })

        return crops

    def run_crop_inference(
        self,
        image: np.ndarray,
        detector_model: Any,
        crops: List[Dict[str, Any]],
        apply_edge_enhancement: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes detector inference on each targeted high-resolution crop
        and projects predicted coordinates back to global frame coordinates.
        """
        projected_detections = []

        for crop_info in crops:
            cx1, cy1, cx2, cy2 = crop_info["crop_box"]
            crop_img = image[cy1:cy2, cx1:cx2].copy()

            if apply_edge_enhancement and crop_info.get("is_thin_crack", False):
                crop_img = self.enhancer.enhance_thin_cracks(crop_img, alpha=0.30)

            # Detector inference on the crop
            results = detector_model.predict(
                source=crop_img,
                imgsz=self.crop_size,
                verbose=False
            )[0]

            if results.boxes is not None and len(results.boxes) > 0:
                c_boxes = results.boxes.xyxy.cpu().numpy()
                c_confs = results.boxes.conf.cpu().numpy()
                c_clss = results.boxes.cls.cpu().numpy().astype(int)

                for b, conf, cls_id in zip(c_boxes, c_confs, c_clss):
                    # Project box to global frame
                    global_box = [
                        float(b[0] + cx1),
                        float(b[1] + cy1),
                        float(b[2] + cx1),
                        float(b[3] + cy1)
                    ]
                    cls_name = detector_model.names.get(cls_id, str(cls_id))
                    projected_detections.append({
                        "box": global_box,
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": cls_name,
                        "source": "crop_refinement"
                    })

        return projected_detections

    def merge_predictions_wbf(
        self,
        full_frame_preds: List[Dict[str, Any]],
        crop_preds: List[Dict[str, Any]],
        full_weight: float = 1.0,
        crop_weight: float = 1.3
    ) -> List[Dict[str, Any]]:
        """
        Merges global-frame and crop-level predictions using Weighted Box Fusion (WBF).
        Crop predictions receive slightly higher weight for small defects because they
        were inspected at higher effective pixel resolution.
        """
        all_preds = []
        for p in full_frame_preds:
            p_copy = dict(p)
            p_copy["weight"] = full_weight
            p_copy["source"] = p.get("source", "full_frame")
            all_preds.append(p_copy)
        for p in crop_preds:
            p_copy = dict(p)
            p_copy["weight"] = crop_weight
            p_copy["source"] = "crop_refinement"
            all_preds.append(p_copy)

        if not all_preds:
            return []

        # Group by class
        merged = []
        classes = set(p.get("class_name", "") for p in all_preds)

        for cls in classes:
            cls_preds = [p for p in all_preds if p.get("class_name", "") == cls]
            cls_preds.sort(key=lambda x: x["confidence"], reverse=True)

            clusters: List[List[Dict[str, Any]]] = []

            for pred in cls_preds:
                matched = False
                for cluster in clusters:
                    # Check IoU with representative of cluster
                    rep_box = cluster[0]["box"]
                    iou = self._iou(pred["box"], rep_box)
                    if iou >= self.iou_merge_threshold:
                        cluster.append(pred)
                        matched = True
                        break
                if not matched:
                    clusters.append([pred])

            # Merge clusters using confidence weights
            for cluster in clusters:
                total_w = sum(p["confidence"] * p["weight"] for p in cluster)
                if total_w <= 0:
                    continue

                w_x1 = sum(p["box"][0] * p["confidence"] * p["weight"] for p in cluster) / total_w
                w_y1 = sum(p["box"][1] * p["confidence"] * p["weight"] for p in cluster) / total_w
                w_x2 = sum(p["box"][2] * p["confidence"] * p["weight"] for p in cluster) / total_w
                w_y2 = sum(p["box"][3] * p["confidence"] * p["weight"] for p in cluster) / total_w

                # Fused confidence is maximum or weighted mean
                max_conf = max(p["confidence"] for p in cluster)
                sources = list(set(p.get("source", "") for p in cluster))

                merged.append({
                    "box": [round(w_x1, 1), round(w_y1, 1), round(w_x2, 1), round(w_y2, 1)],
                    "confidence": round(float(max_conf), 4),
                    "class_name": cls,
                    "class_id": cluster[0].get("class_id", 0),
                    "cluster_size": len(cluster),
                    "fused_sources": sources
                })

        return merged

    @staticmethod
    def _iou(boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        denom = areaA + areaB - inter
        return inter / denom if denom > 0 else 0.0


class PolygonDatasetHelper:
    """
    Manages true polygon annotations for precise boundary evaluation.
    Strictly forbids pseudo-converting bounding boxes to masks.
    """

    @staticmethod
    def validate_polygon_annotation(polygon: List[Tuple[float, float]]) -> bool:
        """
        Verifies that a polygon annotation contains at least 3 valid non-collinear vertices.
        """
        if len(polygon) < 3:
            return False
        pts = np.array(polygon, dtype=np.int32)
        area = cv2.contourArea(pts)
        return area > 4.0

    @staticmethod
    def polygon_to_mask(polygon: List[Tuple[float, float]], shape: Tuple[int, int]) -> np.ndarray:
        """
        Renders verified polygon coordinates to a binary segmentation mask.
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        return mask

    @staticmethod
    def compute_mask_iou(maskA: np.ndarray, maskB: np.ndarray) -> float:
        """
        Computes standard Intersection-over-Union between two binary masks.
        """
        intersection = np.logical_and(maskA > 0, maskB > 0).sum()
        union = np.logical_or(maskA > 0, maskB > 0).sum()
        return float(intersection / union) if union > 0 else 0.0

