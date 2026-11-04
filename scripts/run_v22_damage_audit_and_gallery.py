"""
CYPhER V2.2 Workstream A: Road Damage Candidate Audit & D10 Error Analysis Gallery
Evaluates B0, B1A, B1B, B1C on 1,172 validation images.
Extracts D10 failure gallery (Ground Truth vs Model Predictions).
Outputs reports/V22_DAMAGE_EXPERIMENTS.csv and reports/damage_error_gallery/.
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

VAL_YAML = "data/processed/rdd2022_india/data.yaml"
VAL_IMG_DIR = Path("data/processed/rdd2022_india/images/val")
VAL_LBL_DIR = Path("data/processed/rdd2022_india/labels/val")
GALLERY_DIR = Path("reports/damage_error_gallery")
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
COLORS = {
    0: (255, 100, 0),   # D00: Blue
    1: (0, 0, 255),     # D10: Red
    2: (0, 255, 255),   # D20: Yellow
    3: (0, 255, 0),     # D40: Green
}

CANDIDATES = [
    ("B0_baseline", "runs/detect/runs/road_damage/B0_baseline/weights/best.pt"),
    ("B1A_sampling_only", "runs/detect/runs/road_damage/B1A_sampling_only/weights/best.pt"),
    ("B1B_weighting_only", "runs/detect/runs/road_damage/B1B_weighting_only/weights/best.pt"),
    ("B1C_augmentation_only", "runs/detect/runs/road_damage/B1C_augmentation_only/weights/best.pt"),
]

def evaluate_candidate(name: str, ckpt_path: str):
    print(f"\n=======================================================")
    print(f"EVALUATING CANDIDATE: {name} ({ckpt_path})")
    print(f"=======================================================")
    
    if not Path(ckpt_path).exists():
        print(f"ERROR: {ckpt_path} missing!")
        return None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model = YOLO(ckpt_path)
    
    # Val on Ultralytics standard evaluator
    t0 = time.perf_counter()
    res = model.val(data=VAL_YAML, split="val", imgsz=640, device=0, verbose=False)
    val_time = time.perf_counter() - t0

    p = float(res.results_dict.get("metrics/precision(B)", 0))
    r = float(res.results_dict.get("metrics/recall(B)", 0))
    f1 = 2 * p * r / max(1e-6, p + r)
    map50 = float(res.results_dict.get("metrics/mAP50(B)", 0))
    map50_95 = float(res.results_dict.get("metrics/mAP50-95(B)", 0))

    per_p = [float(x) for x in res.box.p] if hasattr(res.box, "p") else [0]*4
    per_r = [float(x) for x in res.box.r] if hasattr(res.box, "r") else [0]*4
    per_ap50 = [float(x) for x in res.box.ap50] if hasattr(res.box, "ap50") else [0]*4
    per_ap = [float(x) for x in res.box.ap] if hasattr(res.box, "ap") else [0]*4

    # Granular box-level matching at conf=0.25
    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")))
    total_fp = 0
    d10_gt_total = 0
    d10_tp_total = 0
    d10_fp_total = 0
    small_gt_total = 0
    small_tp_total = 0

    for img_p in val_images:
        lbl_p = VAL_LBL_DIR / f"{img_p.stem}.txt"
        gts = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        gts.append({"cls": cid, "box": [cx, cy, bw, bh]})

        preds = model.predict(str(img_p), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        pred_boxes = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            boxes_norm = preds.boxes.xywhn.cpu().numpy()
            clss = preds.boxes.cls.cpu().numpy().astype(int)
            confs = preds.boxes.conf.cpu().numpy()
            for b_norm, c, cf in zip(boxes_norm, clss, confs):
                pred_boxes.append({"cls": int(c), "box": b_norm, "conf": float(cf), "matched": False})

        for gi, gt in enumerate(gts):
            gc = gt["cls"]
            is_small = (gt["box"][2] * 640 * gt["box"][3] * 640) < (32 * 32)
            if is_small:
                small_gt_total += 1
            if gc == 1:
                d10_gt_total += 1

            best_iou = 0.0
            best_pi = -1
            for pi, pb in enumerate(pred_boxes):
                if pb["cls"] == gc and not pb["matched"]:
                    gx1, gy1 = gt["box"][0] - gt["box"][2]/2, gt["box"][1] - gt["box"][3]/2
                    gx2, gy2 = gt["box"][0] + gt["box"][2]/2, gt["box"][1] + gt["box"][3]/2
                    px1, py1 = pb["box"][0] - pb["box"][2]/2, pb["box"][1] - pb["box"][3]/2
                    px2, py2 = pb["box"][0] + pb["box"][2]/2, pb["box"][1] + pb["box"][3]/2

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
                pred_boxes[best_pi]["matched"] = True
                if is_small:
                    small_tp_total += 1
                if gc == 1:
                    d10_tp_total += 1

        for pb in pred_boxes:
            if not pb["matched"]:
                total_fp += 1
                if pb["cls"] == 1:
                    d10_fp_total += 1

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

    return {
        "candidate": name,
        "checkpoint": ckpt_path,
        "precision": round(p * 100.0, 2),
        "recall": round(r * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "mAP50": round(map50 * 100.0, 2),
        "mAP50_95": round(map50_95 * 100.0, 2),
        "D00_AP50": round(per_ap50[0] * 100.0, 2),
        "D10_AP50": round(per_ap50[1] * 100.0, 2),
        "D20_AP50": round(per_ap50[2] * 100.0, 2),
        "D40_AP50": round(per_ap50[3] * 100.0, 2),
        "D10_GT_count": d10_gt_total,
        "D10_TP_conf025": d10_tp_total,
        "D10_FP_conf025": d10_fp_total,
        "D10_Recall_conf025": round((d10_tp_total / max(1, d10_gt_total)) * 100.0, 2),
        "Total_FP_conf025": total_fp,
        "Small_Recall_conf025": round((small_tp_total / max(1, small_gt_total)) * 100.0, 2),
        "Peak_VRAM_MB": peak_vram,
    }

def generate_d10_error_gallery(b0_model_path: str):
    print("\n--- Generating D10 Transverse Crack Error Gallery ---")
    model = YOLO(b0_model_path)
    
    # Identify images with D10 ground truth
    d10_images = []
    for lbl_p in sorted(VAL_LBL_DIR.glob("*.txt")):
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("1 "): # class 1 = D10
                    img_p = VAL_IMG_DIR / f"{lbl_p.stem}.jpg"
                    if img_p.exists():
                        d10_images.append(img_p)
                    break

    print(f"Found {len(d10_images)} validation images containing D10 transverse cracks.")
    
    for img_p in d10_images:
        orig = cv2.imread(str(img_p))
        if orig is None:
            continue
        h, w = orig.shape[:2]

        # Draw Ground Truth on panel A
        panel_gt = orig.copy()
        lbl_p = VAL_LBL_DIR / f"{img_p.stem}.txt"
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    color = COLORS.get(cid, (255, 255, 255))
                    cv2.rectangle(panel_gt, (x1, y1), (x2, y2), color, 2)
                    label = f"GT: {CLASS_NAMES[cid]}"
                    cv2.putText(panel_gt, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw Model Predictions on panel B (conf=0.10 to catch low confidence)
        panel_pred = orig.copy()
        preds = model.predict(str(img_p), imgsz=640, conf=0.10, device=0, verbose=False)[0]
        if preds.boxes is not None and len(preds.boxes) > 0:
            boxes = preds.boxes.xyxy.cpu().numpy().astype(int)
            clss = preds.boxes.cls.cpu().numpy().astype(int)
            confs = preds.boxes.conf.cpu().numpy()
            for box, c, cf in zip(boxes, clss, confs):
                color = COLORS.get(c, (0, 255, 255))
                cv2.rectangle(panel_pred, (box[0], box[1]), (box[2], box[3]), color, 2)
                label = f"{CLASS_NAMES[c]} {cf:.2f}"
                cv2.putText(panel_pred, label, (box[0], max(20, box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Header titles
        cv2.putText(panel_gt, f"GROUND TRUTH: {img_p.stem}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(panel_pred, f"B0 PREDICTION (conf>=0.10): {img_p.stem}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        combined = np.hstack([panel_gt, panel_pred])
        out_path = GALLERY_DIR / f"d10_gallery_{img_p.stem}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"  Saved gallery image: {out_path.name}")

if __name__ == "__main__":
    results = []
    for name, path in CANDIDATES:
        res = evaluate_candidate(name, path)
        if res:
            results.append(res)

    # Save to CSV
    csv_path = "reports/V22_DAMAGE_EXPERIMENTS.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved comparison CSV to {csv_path}")

    # Generate Gallery
    generate_d10_error_gallery("runs/detect/runs/road_damage/B0_baseline/weights/best.pt")
