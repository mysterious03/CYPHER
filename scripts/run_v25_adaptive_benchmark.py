"""
CYPhER V2.4 Master Adaptive Perception Benchmark & Damage Model Evaluation
Evaluates:
1. Baseline B0 Detector
2. CYPhER V2.4 Retrained Detector (with D10 Copy-Paste Augmentation)
3. FAST Adaptive Profile (High Speed / Highway)
4. BALANCED Adaptive Profile (Default Operational)
5. HIGH ACCURACY Adaptive Profile (Research / Inspection)

Evaluates on the full 1,172 held-out validation images.
Computes ground-truth matched Precision, Recall, F1 per class (D00, D10, D20, D40),
defect size recall (small, medium, large), latency statistics, and GPU VRAM consumption.
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Root import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_cv.adaptive.pipeline import AdaptivePerceptionPipeline

VAL_IMG_DIR = Path("data/processed/rdd2022_india/images/val")
VAL_LBL_DIR = Path("data/processed/rdd2022_india/labels/val")
VAL_IMAGES = sorted(list(VAL_IMG_DIR.glob("*.jpg")))
CLASS_NAMES = ["D00", "D10", "D20", "D40"]

def load_ground_truth(stem: str, h: int, w: int):
    lbl_p = VAL_LBL_DIR / f"{stem}.txt"
    gts = []
    if lbl_p.exists():
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                    gx1 = int((cx - bw / 2.0) * w)
                    gy1 = int((cy - bh / 2.0) * h)
                    gx2 = int((cx + bw / 2.0) * w)
                    gy2 = int((cy + bh / 2.0) * h)
                    area = (gx2 - gx1) * (gy2 - gy1)
                    if area < (32 * 32):
                        size_cat = "small"
                    elif area <= (96 * 96):
                        size_cat = "medium"
                    else:
                        size_cat = "large"
                    gts.append({
                        "cls": cid,
                        "box": [gx1, gy1, gx2, gy2],
                        "size_cat": size_cat,
                        "matched": False
                    })
    return gts

def compute_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / max(1e-6, union)

def match_predictions(preds, gts, iou_thresh=0.5):
    # Sort predictions by confidence descending
    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)
    for gt in gts:
        gt["matched"] = False

    tps = {0: 0, 1: 0, 2: 0, 3: 0}
    fps = {0: 0, 1: 0, 2: 0, 3: 0}
    fns = {0: 0, 1: 0, 2: 0, 3: 0}
    size_matches = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]} # [matched, total]

    for p in preds:
        p_box = p["box"]
        p_cls = p["cls"]
        best_iou = 0.0
        best_gt = None
        for gt in gts:
            if gt["cls"] == p_cls and not gt["matched"]:
                iou = compute_iou(p_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

        if best_iou >= iou_thresh and best_gt is not None:
            best_gt["matched"] = True
            tps[p_cls] += 1
        else:
            fps[p_cls] += 1

    for gt in gts:
        cat = gt["size_cat"]
        size_matches[cat][1] += 1
        if gt["matched"]:
            size_matches[cat][0] += 1
        else:
            fns[gt["cls"]] += 1

    return tps, fps, fns, size_matches

def evaluate_detector(model, name: str, is_pipeline=False, profile="balanced"):
    print(f"\nEvaluating: {name} on {len(VAL_IMAGES)} held-out images...")
    total_tps = {0: 0, 1: 0, 2: 0, 3: 0}
    total_fps = {0: 0, 1: 0, 2: 0, 3: 0}
    total_fns = {0: 0, 1: 0, 2: 0, 3: 0}
    total_size = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    latencies = []

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        if is_pipeline:
            model.run_adaptive_inference(dummy, profile=profile)
        else:
            model(dummy, verbose=False)

    for i, img_path in enumerate(VAL_IMAGES):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(VAL_IMAGES)} frames...")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_ground_truth(img_path.stem, h, w)

        t0 = time.perf_counter()
        preds = []
        if is_pipeline:
            res = model.run_adaptive_inference(img, profile=profile)
            for det in res["detections"]:
                preds.append({
                    "box": [int(x) for x in det.get("box", det.get("bbox"))],
                    "conf": det["confidence"],
                    "cls": det["class_id"]
                })
        else:
            yres = model(img, imgsz=640, conf=0.25, verbose=False)[0]
            boxes = yres.boxes.xyxy.cpu().numpy()
            confs = yres.boxes.conf.cpu().numpy()
            clss = yres.boxes.cls.cpu().numpy().astype(int)
            for b, c, cl in zip(boxes, confs, clss):
                preds.append({
                    "box": [int(x) for x in b],
                    "conf": float(c),
                    "cls": int(cl)
                })

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        tps, fps, fns, size_m = match_predictions(preds, gts, iou_thresh=0.5)
        for c in range(4):
            total_tps[c] += tps[c]
            total_fps[c] += fps[c]
            total_fns[c] += fns[c]
        for sz in ["small", "medium", "large"]:
            total_size[sz][0] += size_m[sz][0]
            total_size[sz][1] += size_m[sz][1]

    # Metrics computation
    total_tp = sum(total_tps.values())
    total_fp = sum(total_fps.values())
    total_fn = sum(total_fns.values())

    overall_prec = total_tp / max(1, total_tp + total_fp)
    overall_rec = total_tp / max(1, total_tp + total_fn)
    overall_f1 = 2 * overall_prec * overall_rec / max(1e-6, overall_prec + overall_rec)

    per_class = {}
    for c in range(4):
        c_tp = total_tps[c]
        c_fp = total_fps[c]
        c_fn = total_fns[c]
        c_p = c_tp / max(1, c_tp + c_fp)
        c_r = c_tp / max(1, c_tp + c_fn)
        c_f1 = 2 * c_p * c_r / max(1e-6, c_p + c_r)
        per_class[CLASS_NAMES[c]] = {
            "tp": c_tp, "fp": c_fp, "fn": c_fn,
            "precision": round(c_p * 100, 2),
            "recall": round(c_r * 100, 2),
            "f1": round(c_f1 * 100, 2)
        }

    sm_rec = total_size["small"][0] / max(1, total_size["small"][1])
    md_rec = total_size["medium"][0] / max(1, total_size["medium"][1])
    lg_rec = total_size["large"][0] / max(1, total_size["large"][1])

    mean_lat = float(np.mean(latencies))
    med_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps_val = 1000.0 / max(1e-3, med_lat)

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    res_summary = {
        "configuration": name,
        "overall_precision": round(overall_prec * 100, 2),
        "overall_recall": round(overall_rec * 100, 2),
        "overall_f1": round(overall_f1 * 100, 2),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "per_class": per_class,
        "small_recall": round(sm_rec * 100, 2),
        "medium_recall": round(md_rec * 100, 2),
        "large_recall": round(lg_rec * 100, 2),
        "median_latency_ms": round(med_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps": round(fps_val, 1),
        "peak_vram_mb": round(peak_vram, 2)
    }

    print(f"Results for {name}:")
    print(f"  Precision: {res_summary['overall_precision']}% | Recall: {res_summary['overall_recall']}% | F1: {res_summary['overall_f1']}%")
    print(f"  D10: TP={per_class['D10']['tp']}, FP={per_class['D10']['fp']}, FN={per_class['D10']['fn']}, Recall={per_class['D10']['recall']}%")
    print(f"  Latency: {res_summary['median_latency_ms']} ms ({res_summary['fps']} FPS) | Peak VRAM: {res_summary['peak_vram_mb']} MB")
    return res_summary

def run_all_benchmarks():
    print("=" * 70)
    print("CYPhER V2.4 ADAPTIVE PERCEPTION & ROAD DAMAGE BENCHMARK SUITE")
    print("=" * 70)

    b0_weights = "runs/detect/road_damage_v22/weights/best.pt"
    v24_weights = "runs/detect/road_damage_v24/weights/best.pt"
    if not os.path.exists(v24_weights):
        # Fallback to internal save_dir if needed
        alt = "runs/detect/runs/detect/road_damage_v24/weights/best.pt"
        if os.path.exists(alt):
            v24_weights = alt

    b0_model = YOLO(b0_weights)
    v24_model = YOLO(v24_weights)

    all_results = []

    # 1. Baseline B0
    r_b0 = evaluate_detector(b0_model, "B0 Baseline (Single Scale 640x640)")
    all_results.append(r_b0)

    # 2. V2.4 Retrained Model Standalone
    r_v24 = evaluate_detector(v24_model, "V2.4 Retrained (D10 Copy-Paste)")
    all_results.append(r_v24)

    # Adaptive Pipeline with V2.4 weights
    pipe = AdaptivePerceptionPipeline(
        damage_model_path=v24_weights,
        scene_model_path="yolo26n-sem.pt",
        default_profile="balanced"
    )

    # 3. FAST Adaptive Profile
    r_fast = evaluate_detector(pipe, "Adaptive FAST Profile", is_pipeline=True, profile="fast")
    all_results.append(r_fast)

    # 4. BALANCED Adaptive Profile
    r_bal = evaluate_detector(pipe, "Adaptive BALANCED Profile", is_pipeline=True, profile="balanced")
    all_results.append(r_bal)

    # 5. HIGH ACCURACY Adaptive Profile
    r_high = evaluate_detector(pipe, "Adaptive HIGH_ACCURACY Profile", is_pipeline=True, profile="high_accuracy")
    all_results.append(r_high)

    # Save CSV
    csv_path = Path("reports/V24_DAMAGE_COMPARISON.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration", "Precision (%)", "Recall (%)", "F1 (%)", "TP", "FP", "FN",
            "D00 Recall (%)", "D10 Recall (%)", "D20 Recall (%)", "D40 Recall (%)",
            "Small Recall (%)", "Medium Recall (%)", "Large Recall (%)",
            "Median Latency (ms)", "FPS", "Peak VRAM (MB)"
        ])
        for r in all_results:
            writer.writerow([
                r["configuration"], r["overall_precision"], r["overall_recall"], r["overall_f1"],
                r["tp"], r["fp"], r["fn"],
                r["per_class"]["D00"]["recall"], r["per_class"]["D10"]["recall"],
                r["per_class"]["D20"]["recall"], r["per_class"]["D40"]["recall"],
                r["small_recall"], r["medium_recall"], r["large_recall"],
                r["median_latency_ms"], r["fps"], r["peak_vram_mb"]
            ])
    print(f"\nSaved CSV comparison to: {csv_path}")

    # Save JSON
    json_path = Path("reports/v24_results/adaptive_benchmark_summary.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved JSON results to: {json_path}")

    return all_results

if __name__ == "__main__":
    run_all_benchmarks()

