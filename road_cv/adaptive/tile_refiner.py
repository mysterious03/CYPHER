"""
CYPhER Adaptive Perception: Dual-Path High-Resolution Tile Refiner
Implements:
1. Candidate-Triggered Zoom: High-resolution zoom on small or uncertain detections.
2. Independent Road-Tile Proposals: Slices the drivable road corridor into high-resolution tiles
   to discover thin cracks or small defects missed by low-resolution global inference.
3. Coordinate re-projection and Weighted Box Fusion (WBF) / NMS merging.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
import torch

class DualPathTileRefiner:
    """
    Coordinates scale-adaptive inference by merging global context with targeted high-resolution crops.
    """

    def __init__(
        self,
        tile_size: int = 640,
        iou_merge_threshold: float = 0.45,
        max_candidate_crops: int = 3,
        max_independent_tiles: int = 3
    ):
        self.tile_size = tile_size
        self.iou_merge_threshold = iou_merge_threshold
        self.max_candidate_crops = max_candidate_crops
        self.max_independent_tiles = max_independent_tiles

    def generate_candidate_crops(
        self,
        image_shape: Tuple[int, int],
        candidates: List[Dict[str, Any]],
        uncertainty_scores: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates zoomed bounding crops around high-uncertainty or small defect candidates.
        """
        h, w = image_shape[:2]
        crop_specs = []
        half = self.tile_size // 2

        # Sort candidates by uncertainty score if available, or area ratio
        scored_candidates = []
        for i, cand in enumerate(candidates):
            u_score = uncertainty_scores[i] if uncertainty_scores and i < len(uncertainty_scores) else 0.50
            box = cand.get("box", [0, 0, 10, 10])
            area = (box[2] - box[0]) * (box[3] - box[1])
            area_ratio = area / float(h * w)
            priority = u_score * 0.6 + (1.0 - min(1.0, area_ratio * 20.0)) * 0.4
            scored_candidates.append((priority, cand))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:self.max_candidate_crops]

        for priority, cand in top_candidates:
            box = cand["box"]
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, x1 + self.tile_size)
            y2 = min(h, y1 + self.tile_size)

            if x2 - x1 < self.tile_size and w >= self.tile_size:
                x1 = max(0, x2 - self.tile_size)
            if y2 - y1 < self.tile_size and h >= self.tile_size:
                y1 = max(0, y2 - self.tile_size)

            crop_specs.append({
                "type": "candidate_zoom",
                "coords": (x1, y1, x2, y2),
                "trigger_box": box,
                "priority": round(priority, 3)
            })

        return crop_specs

    def map_tile_boxes_to_global(
        self,
        tile_boxes: List[Dict[str, Any]],
        tile_coords: Tuple[int, int, int, int]
    ) -> List[Dict[str, Any]]:
        """
        Maps local tile detections back into the full-frame coordinate space.
        """
        tx1, ty1, tx2, ty2 = tile_coords
        global_boxes = []

        for det in tile_boxes:
            local_box = det["box"] # [lx1, ly1, lx2, ly2]
            gx1 = local_box[0] + tx1
            gy1 = local_box[1] + ty1
            gx2 = local_box[2] + tx1
            gy2 = local_box[3] + ty1

            mapped_det = dict(det)
            mapped_det["box"] = [round(float(gx1), 1), round(float(gy1), 1), round(float(gx2), 1), round(float(gy2), 1)]
            mapped_det["source"] = "tile_refinement"
            global_boxes.append(mapped_det)

        return global_boxes

    def merge_detections_wbf(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Merges multi-source and multi-scale detections using Weighted Box Fusion.
        """
        if not detections:
            return []

        # Group by class
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for d in detections:
            cid = int(d.get("class_id", d.get("cls", 0)))
            by_class.setdefault(cid, []).append(d)

        merged_results = []

        for cid, class_dets in by_class.items():
            # Sort descending by confidence
            class_dets.sort(key=lambda x: float(x.get("confidence", x.get("conf", 0.0))), reverse=True)
            clusters: List[List[Dict[str, Any]]] = []

            for det in class_dets:
                b1 = det["box"]
                matched_cluster = False
                for cluster in clusters:
                    cb = cluster[0]["box"]
                    # Calculate IoU
                    ix1, iy1 = max(b1[0], cb[0]), max(b1[1], cb[1])
                    ix2, iy2 = min(b1[2], cb[2]), min(b1[3], cb[3])
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = ((b1[2]-b1[0])*(b1[3]-b1[1]) + (cb[2]-cb[0])*(cb[3]-cb[1])) - inter
                    iou = inter / max(1e-6, union)

                    if iou >= iou_threshold:
                        cluster.append(det)
                        matched_cluster = True
                        break

                if not matched_cluster:
                    clusters.append([det])

            # Merge each cluster by weighted box fusion
            for cluster in clusters:
                total_conf = sum(float(x.get("confidence", x.get("conf", 0.0))) for x in cluster)
                weights = [float(x.get("confidence", x.get("conf", 0.0))) / max(1e-6, total_conf) for x in cluster]

                avg_x1 = sum(w * x["box"][0] for w, x in zip(weights, cluster))
                avg_y1 = sum(w * x["box"][1] for w, x in zip(weights, cluster))
                avg_x2 = sum(w * x["box"][2] for w, x in zip(weights, cluster))
                avg_y2 = sum(w * x["box"][3] for w, x in zip(weights, cluster))

                max_conf = max(float(x.get("confidence", x.get("conf", 0.0))) for x in cluster)
                rep = cluster[0]

                merged_results.append({
                    "class_id": cid,
                    "class_name": rep.get("class_name", f"class_{cid}"),
                    "confidence": round(float(max_conf), 4),
                    "box": [round(float(avg_x1), 1), round(float(avg_y1), 1), round(float(avg_x2), 1), round(float(avg_y2), 1)],
                    "merged_count": len(cluster)
                })

        return merged_results
