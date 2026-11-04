"""
Comprehensive B0 Road Damage Model Error Analysis Script
Executes thorough error diagnosis on the validation dataset using runs/detect/runs/road_damage/B0_baseline/weights/best.pt.
Generates quantitative error breakdowns and visual comparison images in outputs/B0_error_analysis/.
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole"
}

CLASS_COLORS = {
    0: (255, 128, 0),   # Orange for D00
    1: (0, 0, 255),     # Red for D10
    2: (0, 255, 255),   # Yellow for D20
    3: (0, 255, 0)      # Green for D40
}

def box_iou(b1, b2):
    # b1, b2: [x1, y1, x2, y2]
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def run_error_analysis():
    print("=" * 70)
    print("B0 ROAD DAMAGE MODEL ERROR ANALYSIS & DIAGNOSTICS")
    print("=" * 70)

    weights_p = Path("runs/detect/runs/road_damage/B0_baseline/weights/best.pt")
    assert weights_p.exists(), f"Checkpoint not found at: {weights_p}"

    model = YOLO(weights_p.as_posix())

    val_img_dir = Path("data/processed/rdd2022_india/images/val")
    val_lbl_dir = Path("data/processed/rdd2022_india/labels/val")

    out_base = Path("outputs/B0_error_analysis")
    out_dirs = {
        "d10_missed": out_base / "d10_missed",
        "small_defects_missed": out_base / "small_defects_missed",
        "false_positives": out_base / "false_positives",
        "class_confusion": out_base / "class_confusion",
        "successful_detections": out_base / "successful_detections"
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    val_images = sorted(list(val_img_dir.glob("*.jpg")))
    print(f"Total validation images available: {len(val_images)}")

    # Tracking metrics
    stats = {
        "total_images_analyzed": len(val_images),
        "total_gt_boxes": 0,
        "total_pred_boxes": 0,
        "tp_count": 0,
        "fp_count": 0,
        "fn_count": 0,
        "class_confusion_count": 0,
        "localization_error_count": 0,
        "per_class": {c: {"gt": 0, "tp": 0, "fn": 0, "fp": 0, "confused_as": {c2: 0 for c2 in range(4) if c2 != c}} for c in range(4)},
        "area_breakdown": {
            "small": {"gt": 0, "tp": 0, "fn": 0},      # < 32x32 px
            "medium": {"gt": 0, "tp": 0, "fn": 0},     # 32x32 to 96x96 px
            "large": {"gt": 0, "tp": 0, "fn": 0}       # > 96x96 px
        },
        "d10_case_studies": []
    }

    # Run batched inference or per-image inference
    conf_thresh = 0.25
    iou_thresh = 0.50

    saved_samples_count = {k: 0 for k in out_dirs.keys()}
    max_save_per_cat = 12

    for idx, img_p in enumerate(val_images):
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load GT boxes
        gt_boxes = [] # [{cls, box: [x1, y1, x2, y2], matched: False, area: float}]
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(parts[0])
                        xc, yc, bw, bh = [float(v) for v in parts[1:5]]
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        box_w_px = max(0, x2 - x1)
                        box_h_px = max(0, y2 - y1)
                        area = box_w_px * box_h_px
                        gt_boxes.append({
                            "cls": c,
                            "box": [x1, y1, x2, y2],
                            "matched": False,
                            "area": area,
                            "w_px": box_w_px,
                            "h_px": box_h_px
                        })
                        stats["total_gt_boxes"] += 1
                        stats["per_class"][c]["gt"] += 1

                        if area < (32 * 32):
                            stats["area_breakdown"]["small"]["gt"] += 1
                        elif area < (96 * 96):
                            stats["area_breakdown"]["medium"]["gt"] += 1
                        else:
                            stats["area_breakdown"]["large"]["gt"] += 1

        # Predict with YOLO
        res = model.predict(source=str(img_p), imgsz=640, conf=conf_thresh, iou=0.5, verbose=False, device=0)[0]

        pred_boxes = [] # [{cls, box: [x1, y1, x2, y2], conf, matched: False}]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for b, c, score in zip(boxes_xyxy, clss, confs):
                pred_boxes.append({
                    "cls": int(c),
                    "box": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                    "conf": float(score),
                    "matched": False
                })
                stats["total_pred_boxes"] += 1

        # Match predictions to GT
        # 1. Exact match: same class & IoU >= iou_thresh
        for p in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for g_i, g in enumerate(gt_boxes):
                if not g["matched"] and g["cls"] == p["cls"]:
                    iou = box_iou(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_i

            if best_iou >= iou_thresh and best_gt_idx != -1:
                p["matched"] = True
                gt_boxes[best_gt_idx]["matched"] = True
                stats["tp_count"] += 1
                stats["per_class"][p["cls"]]["tp"] += 1
                gt_area = gt_boxes[best_gt_idx]["area"]
                if gt_area < (32 * 32):
                    stats["area_breakdown"]["small"]["tp"] += 1
                elif gt_area < (96 * 96):
                    stats["area_breakdown"]["medium"]["tp"] += 1
                else:
                    stats["area_breakdown"]["large"]["tp"] += 1

        # 2. Check for classification confusion (IoU >= 0.5 with GT of different class)
        for p in pred_boxes:
            if not p["matched"]:
                for g in gt_boxes:
                    if not g["matched"] and g["cls"] != p["cls"]:
                        iou = box_iou(p["box"], g["box"])
                        if iou >= iou_thresh:
                            p["matched"] = True
                            g["matched"] = True
                            stats["class_confusion_count"] += 1
                            stats["per_class"][g["cls"]]["confused_as"][p["cls"]] += 1
                            break

        # 3. Check for localization errors (IoU between 0.1 and 0.5 with GT of same class)
        for p in pred_boxes:
            if not p["matched"]:
                for g in gt_boxes:
                    if not g["matched"] and g["cls"] == p["cls"]:
                        iou = box_iou(p["box"], g["box"])
                        if 0.1 <= iou < iou_thresh:
                            p["matched"] = True
                            g["matched"] = True
                            stats["localization_error_count"] += 1
                            break

        # 4. Remaining unmatched predictions are False Positives
        for p in pred_boxes:
            if not p["matched"]:
                stats["fp_count"] += 1
                stats["per_class"][p["cls"]]["fp"] += 1

        # 5. Remaining unmatched GT boxes are False Negatives
        has_d10 = False
        has_small_fn = False
        for g in gt_boxes:
            if not g["matched"]:
                stats["fn_count"] += 1
                stats["per_class"][g["cls"]]["fn"] += 1
                if g["area"] < (32 * 32):
                    stats["area_breakdown"]["small"]["fn"] += 1
                    has_small_fn = True
                elif g["area"] < (96 * 96):
                    stats["area_breakdown"]["medium"]["fn"] += 1
                else:
                    stats["area_breakdown"]["large"]["fn"] += 1

                if g["cls"] == 1:
                    has_d10 = True

        # Special record for D10 case studies
        if any(g["cls"] == 1 for g in gt_boxes):
            stats["d10_case_studies"].append({
                "image": img_p.name,
                "gt_d10_count": sum(1 for g in gt_boxes if g["cls"] == 1),
                "pred_boxes": [{"cls": CLASS_NAMES[p["cls"]], "conf": round(p["conf"], 3), "box": p["box"]} for p in pred_boxes]
            })

        # Save Visual Comparison Images for Diagnostics
        # Build side-by-side visualization: Left = Ground Truth, Right = Predictions
        should_save = False
        cat_to_save = None

        if has_d10 and saved_samples_count["d10_missed"] < max_save_per_cat:
            should_save = True
            cat_to_save = "d10_missed"
        elif has_small_fn and saved_samples_count["small_defects_missed"] < max_save_per_cat:
            should_save = True
            cat_to_save = "small_defects_missed"
        elif any(not p["matched"] for p in pred_boxes) and len(gt_boxes) == 0 and saved_samples_count["false_positives"] < max_save_per_cat:
            should_save = True
            cat_to_save = "false_positives"
        elif any(p.get("matched") and g_matched for p in pred_boxes for g_matched in [True]) and saved_samples_count["successful_detections"] < max_save_per_cat:
            # Check if has at least 1 TP
            if any(p["matched"] for p in pred_boxes):
                should_save = True
                cat_to_save = "successful_detections"

        if should_save and cat_to_save:
            saved_samples_count[cat_to_save] += 1
            vis_gt = img.copy()
            vis_pred = img.copy()

            # Draw GT on left
            for g in gt_boxes:
                c = g["cls"]
                x1, y1, x2, y2 = g["box"]
                color = CLASS_COLORS.get(c, (0, 255, 0))
                cv2.rectangle(vis_gt, (x1, y1), (x2, y2), color, 2)
                lbl = f"GT: {CLASS_NAMES[c].split('_')[0]}"
                cv2.putText(vis_gt, lbl, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Draw Pred on right
            for p in pred_boxes:
                c = p["cls"]
                x1, y1, x2, y2 = p["box"]
                color = CLASS_COLORS.get(c, (0, 0, 255))
                cv2.rectangle(vis_pred, (x1, y1), (x2, y2), color, 2)
                lbl = f"PRED: {CLASS_NAMES[c].split('_')[0]} {p['conf']:.2f}"
                cv2.putText(vis_pred, lbl, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Add title headers
            cv2.rectangle(vis_gt, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(vis_gt, f"GROUND TRUTH: {img_p.name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.rectangle(vis_pred, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(vis_pred, f"B0 PREDICTIONS (conf >= 0.25)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            # Concat side by side
            comparison = np.hstack([vis_gt, vis_pred])
            save_dest = out_dirs[cat_to_save] / f"{cat_to_save}_{img_p.stem}.jpg"
            cv2.imwrite(str(save_dest), comparison)

        if (idx + 1) % 250 == 0 or (idx + 1) == len(val_images):
            print(f"Processed {idx + 1}/{len(val_images)} validation images...")

    # Write statistical summary JSON
    out_json = out_base / "b0_error_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("B0 ERROR ANALYSIS SUMMARY METRICS:")
    print("=" * 70)
    print(f"Total Validation Images:  {stats['total_images_analyzed']}")
    print(f"Total Ground Truth Boxes: {stats['total_gt_boxes']}")
    print(f"Total Predicted Boxes:    {stats['total_pred_boxes']}")
    print(f"True Positives (IoU>=0.5):{stats['tp_count']}")
    print(f"False Positives:          {stats['fp_count']}")
    print(f"False Negatives (Missed): {stats['fn_count']}")
    print(f"Class Confusions:         {stats['class_confusion_count']}")
    print(f"Localization Errors:      {stats['localization_error_count']}")
    print("\nArea Breakdown (Recall):")
    for area_cat, d in stats["area_breakdown"].items():
        recall = (d["tp"] / d["gt"] * 100) if d["gt"] > 0 else 0.0
        print(f"- {area_cat.capitalize()} Defect Recall: {recall:.2f}% (TP: {d['tp']}/{d['gt']}, Missed: {d['fn']})")

    print("\nPer-Class Detection Recall:")
    for c, d in stats["per_class"].items():
        rec = (d["tp"] / d["gt"] * 100) if d["gt"] > 0 else 0.0
        print(f"- {CLASS_NAMES[c]}: Recall {rec:.2f}% (TP: {d['tp']}/{d['gt']}, FN: {d['fn']}, FP: {d['fp']})")
        if any(d["confused_as"].values()):
            print(f"  Confusions: {dict(d['confused_as'])}")

    print(f"\nDiagnostic visual comparisons saved to: {out_base}/")
    return stats

if __name__ == "__main__":
    run_error_analysis()

