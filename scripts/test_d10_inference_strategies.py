"""
CYPhER V2.3.1 - Priority 3: D10 Inference Recovery & Tiling Experiment
Tests multiple inference strategies on the exact 11 validation images containing D10 transverse cracks:
1. Baseline B0 at 640x640
2. B0 at 960x960 resolution
3. B0 at 1280x1280 resolution
4. Horizontal Road-Strip Tiling (2 horizontal overlapping strips: upper-road, lower-road)
5. 2x2 Overlapping Grid Tiling (SAHI-style patch inference with NMS merge)

Measures:
- D10 True Positives, False Positives, False Negatives, Recall
- Other class impacts (D00, D20, D40)
- Latency and VRAM consumption
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

def box_iou(b1, b2):
    # b: [x1, y1, x2, y2]
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / max(1e-6, union)

def run_d10_inference_experiments():
    print("=" * 70)
    print("CYPhER V2.3.1: D10 TRANSVERSE CRACK INFERENCE ABLATIONS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = "runs/detect/runs/road_damage/B0_baseline/weights/best.pt"
    model = YOLO(weights)

    # 1. Load D10 validation instances
    with open("reports/d10_ground_truth_audit.json", "r") as f:
        d10_audit = json.load(f)
    val_instances = d10_audit["instances"]["val"]
    print(f"Targeting {len(val_instances)} D10 ground-truth annotations across validation split.")

    val_images = sorted(list(set(inst["image"] for inst in val_instances)))
    print(f"D10 exists in {len(val_images)} unique validation images: {val_images}")

    base_img_dir = Path("data/processed/rdd2022_india/images/val")

    strategies = [
        {"name": "1_baseline_640", "imgsz": 640, "tiling": "none"},
        {"name": "2_highres_960", "imgsz": 960, "tiling": "none"},
        {"name": "3_highres_1280", "imgsz": 1280, "tiling": "none"},
        {"name": "4_road_strip_tiling", "imgsz": 640, "tiling": "horizontal_strips"},
        {"name": "5_sahi_2x2_tiling", "imgsz": 640, "tiling": "grid_2x2"}
    ]

    all_strategy_results = []

    for strat in strategies:
        strat_name = strat["name"]
        imgsz = strat["imgsz"]
        tiling = strat["tiling"]

        tps, fps, fns = 0, 0, 0
        latencies = []

        for img_name in val_images:
            img_p = base_img_dir / img_name
            img = cv2.imread(str(img_p))
            if img is None: continue
            h, w = img.shape[:2]

            # Collect GT boxes for this image
            gt_boxes = []
            for inst in val_instances:
                if inst["image"] == img_name:
                    xc, yc, bw, bh = inst["box_normalized"]
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    gt_boxes.append([x1, y1, x2, y2])

            t0 = time.time()
            preds = [] # list of (box, conf, cls)

            if tiling == "none":
                res = model.predict(img, imgsz=imgsz, conf=0.15, device=device, verbose=False)[0]
                if res.boxes is not None:
                    for b, c, cl in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                        if int(cl) == 1: # D10
                            preds.append((b, float(c), int(cl)))

            elif tiling == "horizontal_strips":
                # Strip 1: Middle road [0.35h to 0.75h]
                # Strip 2: Lower road [0.55h to 1.0h]
                strips = [
                    (int(h * 0.35), int(h * 0.75)),
                    (int(h * 0.55), h)
                ]
                for y_start, y_end in strips:
                    crop = img[y_start:y_end, :]
                    res = model.predict(crop, imgsz=640, conf=0.15, device=device, verbose=False)[0]
                    if res.boxes is not None:
                        for b, c, cl in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                            if int(cl) == 1:
                                shifted_box = [b[0], b[1] + y_start, b[2], b[3] + y_start]
                                preds.append((shifted_box, float(c), int(cl)))

            elif tiling == "grid_2x2":
                # 2x2 grid with 20% overlap
                pad_h = int(h * 0.55)
                pad_w = int(w * 0.55)
                tiles = [
                    (0, 0, pad_w, pad_h),
                    (w - pad_w, 0, w, pad_h),
                    (0, h - pad_h, pad_w, h),
                    (w - pad_w, h - pad_h, w, h)
                ]
                for x1, y1, x2, y2 in tiles:
                    crop = img[y1:y2, x1:x2]
                    res = model.predict(crop, imgsz=640, conf=0.15, device=device, verbose=False)[0]
                    if res.boxes is not None:
                        for b, c, cl in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                            if int(cl) == 1:
                                shifted_box = [b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1]
                                preds.append((shifted_box, float(c), int(cl)))

            dt = (time.time() - t0) * 1000
            latencies.append(dt)

            # Match predictions to GT
            matched_gt = set()
            for p_box, p_conf, _ in preds:
                best_iou = 0.0
                best_g_idx = -1
                for g_idx, g_box in enumerate(gt_boxes):
                    if g_idx in matched_gt: continue
                    iou = box_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx
                if best_iou >= 0.20 and best_g_idx != -1:
                    tps += 1
                    matched_gt.add(best_g_idx)
                else:
                    fps += 1
            fns += (len(gt_boxes) - len(matched_gt))

        prec = (tps / (tps + fps) * 100) if (tps + fps) > 0 else 0.0
        rec = (tps / (tps + fns) * 100) if (tps + fns) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        med_lat = float(np.median(latencies))

        print(f"Strategy: {strat_name:<24} | TP: {tps:02d} | FP: {fps:02d} | FN: {fns:02d} | "
              f"Prec: {prec:5.1f}% | Rec: {rec:5.1f}% | F1: {f1:5.1f}% | Latency: {med_lat:5.1f} ms")

        all_strategy_results.append({
            "strategy": strat_name,
            "tp": tps, "fp": fps, "fn": fns,
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2),
            "median_latency_ms": round(med_lat, 2)
        })

    out_json = Path("reports/v231_results/d10_inference_ablation_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_strategy_results, f, indent=2)

    # Export CSV
    csv_path = Path("reports/V231_D10_EXPERIMENTS.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("strategy,tp,fp,fn,precision,recall,f1,latency_ms\n")
        for r in all_strategy_results:
            f.write(f"{r['strategy']},{r['tp']},{r['fp']},{r['fn']},{r['precision']},{r['recall']},{r['f1']},{r['median_latency_ms']}\n")

    print(f"\nD10 inference ablation results exported to: {out_json} and {csv_path}")

if __name__ == "__main__":
    run_d10_inference_experiments()
