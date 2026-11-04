"""
CYPhER V2.2 Master Adaptive Perception Pipeline
Coordinates:
1. Low-Resolution Global Context (640x640)
2. Road-Aware Spatial Priors (Drivable corridor soft weights)
3. Calibrated Uncertainty Estimation (Entropy, Margin, Scale, Photometric)
4. Dual-Path High-Resolution Refinement:
   - Path A: Candidate Zoom on Small / High-Uncertainty Detections
   - Path B: Independent Road Corridor Tiling (Discovers defects missed at 640x640)
5. Weighted Box Fusion (WBF) Multi-Scale Merging
6. Compute-Aware Scheduling (Fast, Balanced, High-Accuracy)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from road_cv.adaptive.spatial_selector import RoadAwareSpatialSelector
from road_cv.adaptive.uncertainty_estimator import CalibratedUncertaintyEstimator
from road_cv.adaptive.tile_refiner import DualPathTileRefiner
from road_cv.adaptive.profile_scheduler import AdaptiveProfileScheduler
from road_cv.quality.condition_assessor import ConditionAssessor

class AdaptivePerceptionPipeline:
    """
    Unified Adaptive Perception Engine for CYPhER V2.2.
    """

    def __init__(
        self,
        damage_model_path: str = "runs/detect/road_damage_v22/weights/best.pt",
        scene_model_path: str = "yolo26n-sem.pt",
        default_profile: str = "balanced",
        device: str = "cuda"
    ):
        self.device = 0 if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.damage_model = YOLO(damage_model_path)
        self.scene_model = YOLO(scene_model_path) if scene_model_path else None
        
        self.spatial_selector = RoadAwareSpatialSelector(boundary_dilation_px=40)
        self.uncertainty_estimator = CalibratedUncertaintyEstimator()
        self.tile_refiner = DualPathTileRefiner(tile_size=640, iou_merge_threshold=0.45)
        self.scheduler = AdaptiveProfileScheduler(default_profile=default_profile)
        self.condition_assessor = ConditionAssessor()

    def run_adaptive_inference(
        self,
        frame: np.ndarray,
        profile: Optional[str] = None,
        road_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes adaptive road perception on a single frame.
        """
        t_start = time.perf_counter()
        cfg = self.scheduler.get_config(profile)
        h, w = frame.shape[:2]
        total_pixels_processed = 0

        # Step 1: Optical Quality Assessment
        photo_metrics = self.condition_assessor.assess_frame(frame, road_mask=road_mask)

        # Step 2: Semantic Road Scene Segmentation (if mask not passed)
        if road_mask is None and self.scene_model is not None:
            t_sem = time.perf_counter()
            sem_res = self.scene_model.predict(frame, imgsz=cfg["imgsz"], device=self.device, verbose=False)[0]
            if sem_res.masks is not None and len(sem_res.masks) > 0:
                raw_mask = sem_res.masks.data[0].cpu().numpy()
                road_mask = (raw_mask == 0).astype(np.uint8) # Class 0 = road
                road_mask = cv2.resize(road_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            total_pixels_processed += cfg["imgsz"] * cfg["imgsz"]

        # Step 3: Global Coarse Detection (Base resolution 640x640)
        t_global = time.perf_counter()
        global_res = self.damage_model.predict(
            frame,
            imgsz=cfg["imgsz"],
            conf=cfg["confidence_threshold"],
            device=self.device,
            verbose=False
        )[0]
        total_pixels_processed += cfg["imgsz"] * cfg["imgsz"]

        raw_detections = []
        if global_res.boxes is not None and len(global_res.boxes) > 0:
            boxes = global_res.boxes.xyxy.cpu().numpy()
            clss = global_res.boxes.cls.cpu().numpy().astype(int)
            confs = global_res.boxes.conf.cpu().numpy()

            for b, c, cf in zip(boxes, clss, confs):
                raw_detections.append({
                    "class_id": int(c),
                    "class_name": global_res.names[int(c)],
                    "confidence": float(cf),
                    "box": [float(v) for v in b],
                    "source": "global_coarse"
                })

        # Step 4: Apply Road Spatial Prior and Calibrated Uncertainty Estimation
        scored_detections = []
        uncertainties = []
        for det in raw_detections:
            # Soft spatial prior
            spatial_prior = self.spatial_selector.compute_soft_spatial_prior(
                box=det["box"],
                road_mask=road_mask,
                image_shape=(h, w)
            ) if cfg["enable_road_prior"] else 1.0

            # Uncertainty estimation
            u_info = self.uncertainty_estimator.estimate_uncertainty(
                box=det["box"],
                confidence=det["confidence"],
                image_shape=(h, w),
                photometric_metrics=photo_metrics
            )
            det["spatial_prior"] = spatial_prior
            det["uncertainty"] = u_info["uncertainty_score"]
            det["uncertainty_breakdown"] = u_info["components"]

            # Effective confidence weighted by spatial prior
            det["effective_confidence"] = round(det["confidence"] * spatial_prior, 4)
            scored_detections.append(det)
            uncertainties.append(u_info["uncertainty_score"])

        all_detections_pool = list(scored_detections)
        crops_dispatched = 0
        tiles_dispatched = 0

        # Step 5: Path A — Candidate-Triggered Zoom Crops
        if cfg["enable_candidate_zoom"]:
            zoom_candidates = [
                d for d in scored_detections
                if d["uncertainty"] >= cfg["uncertainty_trigger_threshold"] or
                   (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) < (32 * 32)
            ]
            crop_specs = self.tile_refiner.generate_candidate_crops(
                image_shape=(h, w),
                candidates=zoom_candidates,
                uncertainty_scores=[d["uncertainty"] for d in zoom_candidates]
            )[:cfg["max_candidate_crops"]]

            for cs in crop_specs:
                x1, y1, x2, y2 = cs["coords"]
                crop_img = frame[y1:y2, x1:x2]
                if crop_img.size > 0:
                    crop_res = self.damage_model.predict(
                        crop_img,
                        imgsz=640,
                        conf=cfg["confidence_threshold"],
                        device=self.device,
                        verbose=False
                    )[0]
                    total_pixels_processed += 640 * 640
                    crops_dispatched += 1

                    if crop_res.boxes is not None and len(crop_res.boxes) > 0:
                        c_boxes = crop_res.boxes.xyxy.cpu().numpy()
                        c_clss = crop_res.boxes.cls.cpu().numpy().astype(int)
                        c_confs = crop_res.boxes.conf.cpu().numpy()

                        tile_dets = [
                            {"class_id": int(c), "class_name": crop_res.names[int(c)], "confidence": float(cf), "box": list(b)}
                            for b, c, cf in zip(c_boxes, c_clss, c_confs)
                        ]
                        mapped = self.tile_refiner.map_tile_boxes_to_global(tile_dets, (x1, y1, x2, y2))
                        all_detections_pool.extend(mapped)

        # Step 6: Path B — Independent Road-Corridor Tile Proposals
        if cfg["enable_independent_tiling"]:
            road_tiles = self.spatial_selector.generate_road_tiling_grid(
                road_mask=road_mask,
                image_shape=(h, w),
                tile_size=640,
                overlap_ratio=cfg["tile_overlap_ratio"]
            )[:cfg["max_independent_tiles"]]

            for tile_box in road_tiles:
                tx1, ty1, tx2, ty2 = tile_box
                tile_img = frame[ty1:ty2, tx1:tx2]
                if tile_img.size > 0:
                    tile_res = self.damage_model.predict(
                        tile_img,
                        imgsz=640,
                        conf=cfg["confidence_threshold"],
                        device=self.device,
                        verbose=False
                    )[0]
                    total_pixels_processed += 640 * 640
                    tiles_dispatched += 1

                    if tile_res.boxes is not None and len(tile_res.boxes) > 0:
                        t_boxes = tile_res.boxes.xyxy.cpu().numpy()
                        t_clss = tile_res.boxes.cls.cpu().numpy().astype(int)
                        t_confs = tile_res.boxes.conf.cpu().numpy()

                        tile_dets = [
                            {"class_id": int(c), "class_name": tile_res.names[int(c)], "confidence": float(cf), "box": list(b)}
                            for b, c, cf in zip(t_boxes, t_clss, t_confs)
                        ]
                        mapped = self.tile_refiner.map_tile_boxes_to_global(tile_dets, (tx1, ty1, tx2, ty2))
                        all_detections_pool.extend(mapped)

        # Step 7: Multi-Scale Weighted Box Fusion
        merged_detections = self.tile_refiner.merge_detections_wbf(
            detections=all_detections_pool,
            iou_threshold=0.45
        )

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        newly_discovered_defects = max(0, len(merged_detections) - len(raw_detections))

        return {
            "status": "success",
            "profile_executed": cfg["name"],
            "profile_key": profile or self.scheduler.current_profile,
            "total_latency_ms": round(total_latency_ms, 2),
            "pixels_processed": total_pixels_processed,
            "refinement_stats": {
                "global_coarse_detections": len(raw_detections),
                "candidate_zoom_crops": crops_dispatched,
                "independent_road_tiles": tiles_dispatched,
                "total_merged_detections": len(merged_detections),
                "defects_discovered_by_refinement": newly_discovered_defects
            },
            "detections": merged_detections
        }

if __name__ == "__main__":
    pipeline = AdaptivePerceptionPipeline()
    frame = cv2.imread("Road.jpg")
    if frame is not None:
        print("Executing Adaptive Perception across 3 Profiles on Road.jpg:")
        for prof in ["fast", "balanced", "high_accuracy"]:
            res = pipeline.run_adaptive_inference(frame, profile=prof)
            print(f"Profile: {prof:<14} | Time: {res['total_latency_ms']:.1f}ms | Pixels: {res['pixels_processed']} | Detections: {res['refinement_stats']['total_merged_detections']} (New: {res['refinement_stats']['defects_discovered_by_refinement']})")
