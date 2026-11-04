"""
Multi-Resolution Ground-Truth Matching Evaluation
Evaluates B0 detector on holdout small-defect validation images across 640, 960, and 1280 resolutions.
Uses true ground-truth bounding box matching (IoU >= 0.25 and IoU >= 0.50) to measure real TP, FP, FN, Precision, and Recall.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def evaluate_resolution_scaling(
    ckpt_path: str = "runs/detect/runs/road_damage/B0_baseline/weights/best.pt",
    resolutions = [640, 960, 1280],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.25
):
    print("=" * 65)
    print("MULTI-RESOLUTION GROUND-TRUTH MATCHING EVALUATION")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Confidence Threshold: {conf_thresh} | IoU Match Threshold: {iou_thresh}")
    print("=" * 65)

    model = YOLO(ckpt_path)
    val_img_dir = Path("data/processed/rdd2022_india/images/val")
    val_lbl_dir = Path("data/processed/rdd2022_india/labels/val")

    # Find validation images that have at least one small defect (<32x32 px at 640 scale)
    small_defect_images = []
    all_val_imgs = sorted(list(val_img_dir.glob("*.jpg")))
    
    for img_p in all_val_imgs:
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        has_small = False
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    bw, bh = float(parts[3]), float(parts[4])
                    if (bw * 640) * (bh * 640) < (32 * 32):
                        has_small = True
                        break
        if has_small:
            small_defect_images.append(img_p)

    print(f"Identified {len(small_defect_images)} validation images containing small defects (<32x32 px).")
    # Evaluate across resolutions
    results = {}

    for res in resolutions:
        print(f"\n---> Evaluating Resolution: {res}x{res}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        total_gt = 0
        total_small_gt = 0
        total_tp = 0
        total_small_tp = 0
        total_fp = 0
        total_fn = 0
        latencies = []

        # Warmup
        dummy = np.zeros((res, res, 3), dtype=np.uint8)
        for _ in range(3):
            _ = model.predict(dummy, imgsz=res, device=0, verbose=False)

        for img_p in small_defect_images:
            lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
            gts = []
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        is_small = ((bw * 640) * (bh * 640)) < (32 * 32)
                        gts.append({"cls": cid, "box": [cx, cy, bw, bh], "is_small": is_small, "matched": False})

            total_gt += len(gts)
            total_small_gt += sum(1 for g in gts if g["is_small"])

            t0 = time.perf_counter()
            pred_res = model.predict(str(img_p), imgsz=res, conf=conf_thresh, device=0, verbose=False)[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)

            pred_boxes = []
            if pred_res.boxes is not None and len(pred_res.boxes) > 0:
                boxes_norm = pred_res.boxes.xywhn.cpu().numpy()
                confs = pred_res.boxes.conf.cpu().numpy()
                clss = pred_res.boxes.cls.cpu().numpy().astype(int)
                for bn, c, cf in zip(boxes_norm, clss, confs):
                    pred_boxes.append({"cls": int(c), "box": bn, "conf": float(cf), "matched": False})

            # IoU matching
            for g in gts:
                gx1 = g["box"][0] - g["box"][2]/2
                gy1 = g["box"][1] - g["box"][3]/2
                gx2 = g["box"][0] + g["box"][2]/2
                gy2 = g["box"][1] + g["box"][3]/2

                best_iou = 0.0
                best_pi = -1

                for pi, pb in enumerate(pred_boxes):
                    if pb["matched"]:
                        continue
                    px1 = pb["box"][0] - pb["box"][2]/2
                    py1 = pb["box"][1] - pb["box"][3]/2
                    px2 = pb["box"][0] + pb["box"][2]/2
                    py2 = pb["box"][1] + pb["box"][3]/2

                    ix1, iy1 = max(gx1, px1), max(gy1, py1)
                    ix2, iy2 = min(gx2, px2), min(gy2, py2)
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (gx2-gx1)*(gy2-gy1) + (px2-px1)*(py2-py1) - inter
                    iou = inter / max(1e-6, union)
                    if iou > best_iou:
                        best_iou = iou
                        best_pi = pi

                if best_iou >= iou_thresh and best_pi >= 0:
                    g["matched"] = True
                    pred_boxes[best_pi]["matched"] = True
                    total_tp += 1
                    if g["is_small"]:
                        total_small_tp += 1

            for g in gts:
                if not g["matched"]:
                    total_fn += 1

            for pb in pred_boxes:
                if not pb["matched"]:
                    total_fp += 1

        p = round((total_tp / max(1, total_tp + total_fp)) * 100.0, 2)
        r = round((total_tp / max(1, total_gt)) * 100.0, 2)
        small_r = round((total_small_tp / max(1, total_small_gt)) * 100.0, 2)
        med_lat = round(float(np.median(latencies)), 2)
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

        results[f"{res}x{res}"] = {
            "resolution": res,
            "total_gt_defects": total_gt,
            "total_small_gt": total_small_gt,
            "true_positives": total_tp,
            "small_defect_true_positives": total_small_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": p,
            "recall": r,
            "small_defect_recall": small_r,
            "median_latency_ms": med_lat,
            "peak_vram_mb": peak_vram
        }

        print(f"Results for {res}x{res}:")
        print(f"  - Precision: {p}% | Recall: {r}% | Small Defect Recall: {small_r}%")
        print(f"  - TP: {total_tp} | Small TP: {total_small_tp}/{total_small_gt} | FP: {total_fp} | FN: {total_fn}")
        print(f"  - Latency: {med_lat} ms | Peak VRAM: {peak_vram} MB")

    with open("reports/resolution_scaling_gt_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    evaluate_resolution_scaling()

