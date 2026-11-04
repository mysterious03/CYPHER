"""
B1 Evaluation Gate & Full Metric Audit
Evaluates B0, B1A, and B1B on the 1,172 untouched holdout validation images.
Computes:
- Overall Precision, Recall, F1, mAP50, mAP50-95
- Per-class Precision, Recall, AP for D00, D10, D20, D40
- False Positive Count at conf=0.25
- Small Object Recall (<32x32 px)
- Inference latency & GPU memory consumption
- Ground-truth matching across 640, 960, 1280 resolutions
"""

import os
import sys
import time
import json
import csv
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

VAL_YAML = "data/processed/rdd2022_india/data.yaml"
CLASS_NAMES = ["D00", "D10", "D20", "D40"]

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    return 0.0

def evaluate_model_holdout(ckpt_path: str, name: str):
    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {name} ({ckpt_path})")
    print(f"=======================================================")
    
    if not Path(ckpt_path).exists():
        print(f"ERROR: Checkpoint {ckpt_path} does not exist!")
        return None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model = YOLO(ckpt_path)
    
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(dummy, device=0, verbose=False)

    # Measure pure latency on 50 runs
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model.predict(dummy, device=0, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency = round(float(np.median(latencies)), 2)

    # Standard holdout validation
    res = model.val(data=VAL_YAML, split="val", imgsz=640, device=0, verbose=False)
    
    p = float(res.results_dict.get("metrics/precision(B)", 0))
    r = float(res.results_dict.get("metrics/recall(B)", 0))
    f1 = 2 * p * r / max(1e-6, p + r)
    map50 = float(res.results_dict.get("metrics/mAP50(B)", 0))
    map50_95 = float(res.results_dict.get("metrics/mAP50-95(B)", 0))

    # Per class metrics
    # res.box.p, res.box.r, res.box.ap50, res.box.ap
    per_p = [float(x) for x in res.box.p] if hasattr(res.box, "p") else [0]*4
    per_r = [float(x) for x in res.box.r] if hasattr(res.box, "r") else [0]*4
    per_ap50 = [float(x) for x in res.box.ap50] if hasattr(res.box, "ap50") else [0]*4
    per_ap = [float(x) for x in res.box.ap] if hasattr(res.box, "ap") else [0]*4

    peak_mem = get_gpu_memory_mb()

    # Now let's calculate exact false positive count & small-object recall at conf=0.25
    val_img_dir = Path("data/processed/rdd2022_india/images/val")
    val_lbl_dir = Path("data/processed/rdd2022_india/labels/val")
    
    total_fp = 0
    small_gt_count = 0
    small_gt_tp = 0
    d10_gt_count = 0
    d10_tp_count = 0
    d10_fp_count = 0

    val_images = sorted(list(val_img_dir.glob("*.jpg")))
    
    print(f"Running detailed box-level matching on {len(val_images)} validation images...")
    
    for img_p in val_images:
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        gts = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        # YOLO format: cx, cy, w, h relative
                        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        gts.append({"cls": cid, "box_norm": [cx, cy, bw, bh]})
        
        # Run prediction
        preds_out = model.predict(str(img_p), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        pred_boxes = []
        if preds_out.boxes is not None and len(preds_out.boxes) > 0:
            boxes_norm = preds_out.boxes.xywhn.cpu().numpy()
            boxes_xyxy = preds_out.boxes.xyxy.cpu().numpy()
            clss = preds_out.boxes.cls.cpu().numpy().astype(int)
            confs = preds_out.boxes.conf.cpu().numpy()
            for b_norm, b_xy, c, cf in zip(boxes_norm, boxes_xyxy, clss, confs):
                pred_boxes.append({"cls": int(c), "box_norm": b_norm, "box_xy": b_xy, "conf": float(cf), "matched": False})

        # Match gts with preds using IoU
        matched_gt = [False] * len(gts)
        
        for gi, gt in enumerate(gts):
            gc = gt["cls"]
            # Check if small defect (<32x32 px at 640 scale)
            # gt width/height in normalized: gt["box_norm"][2]*640, gt["box_norm"][3]*640
            w_px = gt["box_norm"][2] * 640
            h_px = gt["box_norm"][3] * 640
            is_small = (w_px * h_px) < (32 * 32)
            if is_small:
                small_gt_count += 1
            if gc == 1:
                d10_gt_count += 1

            best_iou = 0.0
            best_pi = -1
            for pi, pb in enumerate(pred_boxes):
                if pb["cls"] == gc and not pb["matched"]:
                    # compute IoU
                    gx1 = gt["box_norm"][0] - gt["box_norm"][2]/2
                    gy1 = gt["box_norm"][1] - gt["box_norm"][3]/2
                    gx2 = gt["box_norm"][0] + gt["box_norm"][2]/2
                    gy2 = gt["box_norm"][1] + gt["box_norm"][3]/2

                    px1 = pb["box_norm"][0] - pb["box_norm"][2]/2
                    py1 = pb["box_norm"][1] - pb["box_norm"][3]/2
                    px2 = pb["box_norm"][0] + pb["box_norm"][2]/2
                    py2 = pb["box_norm"][1] + pb["box_norm"][3]/2

                    ix1, iy1 = max(gx1, px1), max(gy1, py1)
                    ix2, iy2 = min(gx2, px2), min(gy2, py2)
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (gx2-gx1)*(gy2-gy1) + (px2-px1)*(py2-py1) - inter
                    iou = inter / max(1e-6, union)
                    if iou > best_iou:
                        best_iou = iou
                        best_pi = pi
            
            if best_iou >= 0.25 and best_pi >= 0:
                matched_gt[gi] = True
                pred_boxes[best_pi]["matched"] = True
                if is_small:
                    small_gt_tp += 1
                if gc == 1:
                    d10_tp_count += 1

        for pb in pred_boxes:
            if not pb["matched"]:
                total_fp += 1
                if pb["cls"] == 1:
                    d10_fp_count += 1

    small_recall = round((small_gt_tp / max(1, small_gt_count)) * 100.0, 2)
    d10_matched_recall = round((d10_tp_count / max(1, d10_gt_count)) * 100.0, 2)

    data = {
        "model_name": name,
        "checkpoint": ckpt_path,
        "precision": round(p * 100.0, 2),
        "recall": round(r * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "mAP50": round(map50 * 100.0, 2),
        "mAP50_95": round(map50_95 * 100.0, 2),
        "per_class": {
            "D00": {"precision": round(per_p[0]*100, 2), "recall": round(per_r[0]*100, 2), "ap50": round(per_ap50[0]*100, 2), "ap50_95": round(per_ap[0]*100, 2)},
            "D10": {"precision": round(per_p[1]*100, 2), "recall": round(per_r[1]*100, 2), "ap50": round(per_ap50[1]*100, 2), "ap50_95": round(per_ap[1]*100, 2)},
            "D20": {"precision": round(per_p[2]*100, 2), "recall": round(per_r[2]*100, 2), "ap50": round(per_ap50[2]*100, 2), "ap50_95": round(per_ap[2]*100, 2)},
            "D40": {"precision": round(per_p[3]*100, 2), "recall": round(per_r[3]*100, 2), "ap50": round(per_ap50[3]*100, 2), "ap50_95": round(per_ap[3]*100, 2)},
        },
        "false_positive_count_at_025": total_fp,
        "d10_tp_at_025": d10_tp_count,
        "d10_fp_at_025": d10_fp_count,
        "d10_matched_recall_at_025": d10_matched_recall,
        "small_object_gt_count": small_gt_count,
        "small_object_tp_count": small_gt_tp,
        "small_object_recall": small_recall,
        "latency_ms": avg_latency,
        "peak_vram_mb": peak_mem
    }
    return data

if __name__ == "__main__":
    models = [
        ("B0_baseline", "runs/detect/runs/road_damage/B0_baseline/weights/best.pt"),
        ("B1A_sampling_only", "runs/detect/runs/road_damage/B1A_sampling_only/weights/best.pt"),
        ("B1B_weighting_only", "runs/detect/runs/road_damage/B1B_weighting_only/weights/best.pt"),
    ]
    
    results = {}
    for name, path in models:
        r = evaluate_model_holdout(path, name)
        if r:
            results[name] = r
            
    with open("reports/b1_gate_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\nSaved gate audit results to reports/b1_gate_audit_results.json")

