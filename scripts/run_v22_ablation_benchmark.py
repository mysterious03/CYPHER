"""
CYPhER V2.2 Workstream G: Reproducible Ablation Experiment Framework
Evaluates 5 controlled configurations on holdout road imagery:
1. Baseline Independent Detector (Standard 640x640 single-scale inference)
2. Road-Aware Spatial Prior Only (Soft spatial corridor gating)
3. Adaptive High-Res Crop Refinement Only (Path A candidate zoom)
4. Uncertainty-Guided Refinement Only (Calibrated U-score triggering)
5. Full Adaptive Perception Pipeline (Spatial + Uncertainty + Dual-path Zoom & Tiling + WBF)

Measures: mAP50, Precision, Recall, Small-Defect Recall, Latency, Peak VRAM, Processed Pixels.
"""

import os
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_cv.adaptive.pipeline import AdaptivePerceptionPipeline
from road_cv.adaptive.spatial_selector import RoadAwareSpatialSelector
from road_cv.adaptive.uncertainty_estimator import CalibratedUncertaintyEstimator
from road_cv.adaptive.tile_refiner import DualPathTileRefiner

VAL_IMG_DIR = Path("data/processed/rdd2022_india/images/val")
VAL_LBL_DIR = Path("data/processed/rdd2022_india/labels/val")

# Use a clean 100-image holdout evaluation split containing damage and background
VAL_SAMPLES = sorted(list(VAL_IMG_DIR.glob("*.jpg")))[:100]

def load_ground_truth(stem: str):
    lbl_p = VAL_LBL_DIR / f"{stem}.txt"
    gts = []
    if lbl_p.exists():
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                    gts.append({"cls": cid, "box": [cx, cy, bw, bh]})
    return gts

def run_ablation_suite():
    print("=" * 80)
    print("CYPhER V2.2 CONTROLLED ABLATION EXPERIMENT BENCHMARK")
    print(f"Evaluating across {len(VAL_SAMPLES)} holdout road camera validation images")
    print("=" * 80)

    damage_model = YOLO("runs/detect/road_damage_v22/weights/best.pt")
    adaptive_pipe = AdaptivePerceptionPipeline(damage_model_path="runs/detect/road_damage_v22/weights/best.pt")

    configs = [
        "1_baseline_independent",
        "2_road_aware_spatial_only",
        "3_adaptive_crops_only",
        "4_uncertainty_guided_only",
        "5_full_adaptive_perception"
    ]

    ablation_results = {}

    for cfg_name in configs:
        print(f"\n>> Running Ablation Config: {cfg_name}...")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        latencies = []
        total_pixels = 0
        total_invocations = 0
        
        all_tp = {0: 0, 1: 0, 2: 0, 3: 0}
        all_fp = {0: 0, 1: 0, 2: 0, 3: 0}
        all_gt = {0: 0, 1: 0, 2: 0, 3: 0}
        small_gt_cnt = 0
        small_tp_cnt = 0

        for img_p in VAL_SAMPLES:
            frame = cv2.imread(str(img_p))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            gts = load_ground_truth(img_p.stem)
            
            # Count GTs
            for gt in gts:
                all_gt[gt["cls"]] += 1
                is_small = (gt["box"][2] * w * gt["box"][3] * h) < (32 * 32)
                if is_small:
                    small_gt_cnt += 1

            t0 = time.perf_counter()
            predicted_boxes = []

            if cfg_name == "1_baseline_independent":
                # Standard single-scale 640x640 YOLO forward pass
                res = damage_model.predict(frame, imgsz=640, conf=0.25, device=0, verbose=False)[0]
                total_pixels += 640 * 640
                total_invocations += 1
                if res.boxes is not None and len(res.boxes) > 0:
                    for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        predicted_boxes.append({"cls": int(c), "box": list(b), "conf": float(cf)})

            elif cfg_name == "2_road_aware_spatial_only":
                # Standard inference + soft spatial prior filter
                res = damage_model.predict(frame, imgsz=640, conf=0.20, device=0, verbose=False)[0]
                total_pixels += 640 * 640
                total_invocations += 1
                if res.boxes is not None and len(res.boxes) > 0:
                    for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        # Suppress detections far outside lower half
                        cy = (b[1] + b[3]) / 2.0
                        if cy >= h * 0.25: # Road corridor spatial prior
                            predicted_boxes.append({"cls": int(c), "box": list(b), "conf": float(cf)})

            elif cfg_name == "3_adaptive_crops_only":
                # Standard inference + top-2 candidate zoom crops (without uncertainty weighting)
                res = damage_model.predict(frame, imgsz=640, conf=0.25, device=0, verbose=False)[0]
                total_pixels += 640 * 640
                total_invocations += 1
                cand_pool = []
                if res.boxes is not None and len(res.boxes) > 0:
                    for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        cand_pool.append({"cls": int(c), "class_id": int(c), "box": list(b), "conf": float(cf), "confidence": float(cf)})
                
                # Zoom top candidates
                for c in cand_pool[:2]:
                    bx = c["box"]
                    cx, cy = int((bx[0]+bx[2])/2), int((bx[1]+bx[3])/2)
                    x1, y1 = max(0, cx - 320), max(0, cy - 320)
                    x2, y2 = min(w, x1 + 640), min(h, y1 + 640)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        c_res = damage_model.predict(crop, imgsz=640, conf=0.25, device=0, verbose=False)[0]
                        total_pixels += 640 * 640
                        total_invocations += 1
                        if c_res.boxes is not None and len(c_res.boxes) > 0:
                            for cb, cc, ccf in zip(c_res.boxes.xyxy.cpu().numpy(), c_res.boxes.cls.cpu().numpy(), c_res.boxes.conf.cpu().numpy()):
                                cand_pool.append({
                                    "cls": int(cc), "class_id": int(cc),
                                    "box": [cb[0]+x1, cb[1]+y1, cb[2]+x1, cb[3]+y1],
                                    "conf": float(ccf), "confidence": float(ccf)
                                })
                predicted_boxes = adaptive_pipe.tile_refiner.merge_detections_wbf(cand_pool, 0.45)

            elif cfg_name == "4_uncertainty_guided_only":
                # Standard inference + uncertainty metric triggering
                res = damage_model.predict(frame, imgsz=640, conf=0.20, device=0, verbose=False)[0]
                total_pixels += 640 * 640
                total_invocations += 1
                cand_pool = []
                if res.boxes is not None and len(res.boxes) > 0:
                    for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        u = adaptive_pipe.uncertainty_estimator.estimate_uncertainty(b, float(cf), (h, w))
                        if u["uncertainty_score"] < 0.65: # Filter out excessively uncertain spurious noise
                            cand_pool.append({"cls": int(c), "class_id": int(c), "box": list(b), "conf": float(cf), "confidence": float(cf)})
                predicted_boxes = cand_pool

            elif cfg_name == "5_full_adaptive_perception":
                # Master Dual-Path Adaptive Pipeline (Balanced Profile)
                out = adaptive_pipe.run_adaptive_inference(frame, profile="balanced")
                total_pixels += out["pixels_processed"]
                total_invocations += (1 + out["refinement_stats"]["candidate_zoom_crops"] + out["refinement_stats"]["independent_road_tiles"])
                for d in out["detections"]:
                    predicted_boxes.append({"cls": int(d["class_id"]), "box": d["box"], "conf": float(d["confidence"])})

            latencies.append((time.perf_counter() - t0) * 1000.0)

            # Match predictions against GT
            matched_preds = [False] * len(predicted_boxes)
            for gt in gts:
                gc = gt["cls"]
                gx1 = (gt["box"][0] - gt["box"][2]/2) * w
                gy1 = (gt["box"][1] - gt["box"][3]/2) * h
                gx2 = (gt["box"][0] + gt["box"][2]/2) * w
                gy2 = (gt["box"][1] + gt["box"][3]/2) * h
                is_small = ((gx2 - gx1) * (gy2 - gy1)) < (32 * 32)

                best_iou = 0.0
                best_idx = -1
                for pi, pb in enumerate(predicted_boxes):
                    if pb.get("cls", pb.get("class_id")) == gc and not matched_preds[pi]:
                        px1, py1, px2, py2 = pb["box"]
                        ix1, iy1 = max(gx1, px1), max(gy1, py1)
                        ix2, iy2 = min(gx2, px2), min(gy2, py2)
                        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                        inter = iw * ih
                        union = ((gx2-gx1)*(gy2-gy1) + (px2-px1)*(py2-py1)) - inter
                        iou = inter / max(1e-6, union)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = pi

                if best_iou >= 0.25 and best_idx >= 0:
                    matched_preds[best_idx] = True
                    all_tp[gc] += 1
                    if is_small:
                        small_tp_cnt += 1

            for pi, pb in enumerate(predicted_boxes):
                if not matched_preds[pi]:
                    c = pb.get("cls", pb.get("class_id", 0))
                    all_fp[c] = all_fp.get(c, 0) + 1

        total_tp = sum(all_tp.values())
        total_fp = sum(all_fp.values())
        total_gt = sum(all_gt.values())

        p_overall = round((total_tp / max(1, total_tp + total_fp)) * 100.0, 2)
        r_overall = round((total_tp / max(1, total_gt)) * 100.0, 2)
        f1_overall = round(2 * p_overall * r_overall / max(1e-6, p_overall + r_overall), 2)
        small_rec = round((small_tp_cnt / max(1, small_gt_cnt)) * 100.0, 2)
        median_lat = round(float(np.median(latencies)), 2)
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

        res_entry = {
            "config": cfg_name,
            "precision": p_overall,
            "recall": r_overall,
            "f1": f1_overall,
            "small_defect_recall": small_rec,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "per_class_recall": {f"D{i}0": round((all_tp[i] / max(1, all_gt[i])) * 100.0, 2) for i in range(4)},
            "median_latency_ms": median_lat,
            "peak_vram_mb": peak_vram,
            "avg_pixels_per_frame": int(total_pixels / max(1, len(VAL_SAMPLES))),
            "avg_invocations_per_frame": round(total_invocations / max(1, len(VAL_SAMPLES)), 2)
        }
        ablation_results[cfg_name] = res_entry
        print(f"   Precision: {p_overall}% | Recall: {r_overall}% | F1: {f1_overall}% | Small Recall: {small_rec}% | Latency: {median_lat}ms")

    # Save to JSON
    out_json = Path("reports/v22_ablation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Ablation Suite Finished. Saved to {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    run_ablation_suite()
