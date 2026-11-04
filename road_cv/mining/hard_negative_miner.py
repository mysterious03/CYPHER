"""
Hard-Negative Mining and False-Positive Tracking Module
Identifies, catalogues, and leverages false-positive candidates (shadows, tar seams,
lane markings, manholes, stains, joints) to improve model discriminability.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

class HardNegativeMiner:
    """
    Identifies high-confidence false detections on clean or distractor pavement regions,
    extracts diagnostic patches, and formats background images for YOLO fine-tuning.
    """
    
    DEFAULT_CATEGORIES = [
        "shadow_edge",
        "tar_repair_seam",
        "lane_marking",
        "manhole_joint",
        "water_stain_reflection",
        "unclassified_background"
    ]
    
    def __init__(
        self,
        output_dir: str = "data/hard_negatives",
        iou_threshold: float = 0.10,
        min_conf_to_flag: float = 0.25
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir = self.output_dir / "patches"
        self.patches_dir.mkdir(exist_ok=True)
        self.yolo_bg_dir = self.output_dir / "yolo_backgrounds"
        self.yolo_bg_dir.mkdir(exist_ok=True)
        
        self.iou_threshold = iou_threshold
        self.min_conf_to_flag = min_conf_to_flag
        self.catalog_file = self.output_dir / "mined_negatives_catalog.json"
        self.catalog: List[Dict[str, Any]] = self._load_catalog()

    def _load_catalog(self) -> List[Dict[str, Any]]:
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_catalog(self):
        with open(self.catalog_file, "w", encoding="utf-8") as f:
            json.dump(self.catalog, f, indent=2)

    @staticmethod
    def _compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0.0

    def mine_image(
        self,
        image_path: str,
        predictions: List[Dict[str, Any]],
        ground_truth_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
        road_mask: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Identifies false positive detections on an image.
        If ground_truth_boxes is empty or None, any detection above min_conf_to_flag
        on a confirmed negative image is flagged as a hard negative.
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
        h, w = image.shape[:2]
        gt_boxes = ground_truth_boxes or []
        
        flagged_negatives = []
        
        for det in predictions:
            box = det["box"] # [x1, y1, x2, y2]
            conf = float(det.get("confidence", 0.0))
            cls_name = det.get("class_name", "defect")
            
            if conf < self.min_conf_to_flag:
                continue
                
            # Check maximum IoU against known ground truth
            max_iou = 0.0
            for gt in gt_boxes:
                iou = self._compute_iou(box, gt)
                if iou > max_iou:
                    max_iou = iou
                    
            # If no ground truth overlap, it's a false positive candidate
            if max_iou <= self.iou_threshold:
                # Classify visual context (heuristic negative category)
                neg_category = self._classify_negative_patch(image, box, road_mask)
                
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                patch_id = f"fp_{int(time.time()*1000)}_{len(flagged_negatives)}"
                patch_filename = f"{patch_id}_{neg_category}.jpg"
                patch_path = self.patches_dir / patch_filename
                
                if (x2 > x1) and (y2 > y1):
                    patch_img = image[y1:y2, x1:x2]
                    cv2.imwrite(str(patch_path), patch_img)
                
                record = {
                    "patch_id": patch_id,
                    "source_image": str(image_path),
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "predicted_class": cls_name,
                    "negative_category": neg_category,
                    "patch_path": str(patch_path),
                    "timestamp": time.time()
                }
                flagged_negatives.append(record)
                self.catalog.append(record)

        self._save_catalog()
        return flagged_negatives

    def _classify_negative_patch(
        self,
        image: np.ndarray,
        box: List[float],
        road_mask: Optional[np.ndarray]
    ) -> str:
        """
        Classifies the false positive based on local color, contrast, and geometry.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return "unclassified_background"

        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Color variance & brightness
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        
        # Aspect ratio of the box
        aspect_ratio = (x2 - x1) / max(1, (y2 - y1))
        
        # Heuristic checks
        if mean_val < 50 and std_val < 25:
            return "shadow_edge"
        elif mean_val > 180 and std_val < 35:
            return "lane_marking"
        elif (aspect_ratio > 3.0 or aspect_ratio < 0.33) and std_val > 30:
            return "tar_repair_seam"
        elif abs(crop.shape[0] - crop.shape[1]) < min(crop.shape[:2]) * 0.3:
            # Circular or square gradient check
            edges = cv2.Canny(gray, 50, 150)
            if np.count_nonzero(edges) / edges.size > 0.15:
                return "manhole_joint"
        return "unclassified_background"

    def export_yolo_negative_images(self, source_images: List[str]) -> int:
        """
        Copies verified negative images to yolo_backgrounds/ and creates empty .txt label files.
        YOLO natively treats empty label files as pure background/negative images.
        """
        exported_count = 0
        img_out = self.yolo_bg_dir / "images"
        lbl_out = self.yolo_bg_dir / "labels"
        img_out.mkdir(exist_ok=True)
        lbl_out.mkdir(exist_ok=True)

        for img_p in source_images:
            p = Path(img_p)
            if not p.exists():
                continue
            dest_img = img_out / p.name
            dest_lbl = lbl_out / f"{p.stem}.txt"
            
            shutil.copy2(p, dest_img)
            # Create empty label file
            with open(dest_lbl, "w") as f:
                pass
            exported_count += 1
            
        return exported_count


class FalsePositiveTracker:
    """
    Tracks false-positive reduction across training iterations (t0, t1, ...).
    """
    
    def __init__(self, log_file: str = "data/hard_negatives/fp_history.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def record_iteration(
        self,
        iteration_name: str,
        total_eval_images: int,
        fp_count: int,
        fp_by_category: Optional[Dict[str, int]] = None
    ):
        fp_per_image = fp_count / max(1, total_eval_images)
        entry = {
            "iteration": iteration_name,
            "timestamp": time.time(),
            "total_images": total_eval_images,
            "fp_count": fp_count,
            "fp_per_image": round(fp_per_image, 4),
            "fp_by_category": fp_by_category or {}
        }
        self.history.append(entry)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def compute_reduction_summary(self) -> Dict[str, Any]:
        """
        Calculates reduction percentages between the earliest and latest iterations.
        """
        if len(self.history) < 2:
            return {"status": "insufficient_iterations", "history_length": len(self.history)}
            
        initial = self.history[0]
        latest = self.history[-1]
        
        init_rate = initial["fp_per_image"]
        latest_rate = latest["fp_per_image"]
        
        reduction = (init_rate - latest_rate) / max(1e-5, init_rate) * 100.0
        
        return {
            "initial_iteration": initial["iteration"],
            "initial_fp_per_image": init_rate,
            "latest_iteration": latest["iteration"],
            "latest_fp_per_image": latest_rate,
            "reduction_percentage": round(reduction, 2),
            "total_recorded_iterations": len(self.history)
        }

