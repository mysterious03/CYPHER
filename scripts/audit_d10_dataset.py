"""
CYPhER V2.3.1 - Priority 3: Deep D10 Transverse Crack Dataset Audit
Audits all D10 training and validation images:
- Instance count
- Aspect ratios (w/h)
- Bounding box dimensions (width, height in pixels)
- Pixel areas
- Contrast and intensity statistics
- Verification of whether D10 is underrepresented or mislabelled
"""

import os
import glob
import json
import cv2
import numpy as np
from pathlib import Path

def audit_d10():
    print("=" * 65)
    print("AUDITING D10 TRANSVERSE CRACK GROUND TRUTH")
    print("=" * 65)

    base_dir = Path("data/processed/rdd2022_india")
    results = {"train": [], "val": []}

    for split in ["train", "val"]:
        img_dir = base_dir / "images" / split
        lbl_dir = base_dir / "labels" / split

        label_files = list(lbl_dir.glob("*.txt"))
        for lf in label_files:
            with open(lf, "r") as f:
                lines = f.readlines()

            img_p = img_dir / f"{lf.stem}.jpg"
            if not img_p.exists():
                continue

            img = None
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id == 1: # D10 Transverse Crack
                    if img is None:
                        img = cv2.imread(str(img_p))
                    if img is None:
                        continue
                    h_img, w_img = img.shape[:2]

                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    px_w = bw * w_img
                    px_h = bh * h_img
                    x1 = max(0, int((xc - bw / 2) * w_img))
                    y1 = max(0, int((yc - bh / 2) * h_img))
                    x2 = min(w_img, int((xc + bw / 2) * w_img))
                    y2 = min(h_img, int((yc + bh / 2) * h_img))

                    aspect_ratio = px_w / max(1e-5, px_h)
                    area_px = px_w * px_h

                    # Crop patch and analyze contrast
                    crop = img[y1:y2, x1:x2]
                    contrast = 0.0
                    mean_lum = 0.0
                    if crop.size > 0:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        contrast = float(np.std(gray))
                        mean_lum = float(np.mean(gray))

                    results[split].append({
                        "image": img_p.name,
                        "line_idx": idx,
                        "box_normalized": [xc, yc, bw, bh],
                        "width_px": round(px_w, 2),
                        "height_px": round(px_h, 2),
                        "aspect_ratio": round(aspect_ratio, 2),
                        "area_px": round(area_px, 2),
                        "contrast_std": round(contrast, 2),
                        "mean_luminance": round(mean_lum, 2)
                    })

    # Summary statistics
    summary = {}
    for split in ["train", "val"]:
        instances = results[split]
        count = len(instances)
        if count == 0:
            summary[split] = {"count": 0}
            continue

        ars = [inst["aspect_ratio"] for inst in instances]
        widths = [inst["width_px"] for inst in instances]
        heights = [inst["height_px"] for inst in instances]
        areas = [inst["area_px"] for inst in instances]
        contrasts = [inst["contrast_std"] for inst in instances]

        summary[split] = {
            "instance_count": count,
            "mean_aspect_ratio": round(float(np.mean(ars)), 2),
            "median_aspect_ratio": round(float(np.median(ars)), 2),
            "max_aspect_ratio": round(float(np.max(ars)), 2),
            "min_aspect_ratio": round(float(np.min(ars)), 2),
            "mean_width_px": round(float(np.mean(widths)), 2),
            "mean_height_px": round(float(np.mean(heights)), 2),
            "mean_area_px": round(float(np.mean(areas)), 2),
            "median_area_px": round(float(np.median(areas)), 2),
            "small_area_ratio": round(float(np.mean(np.array(areas) < 1024)) * 100, 2),
            "mean_contrast": round(float(np.mean(contrasts)), 2)
        }

    out_file = Path("reports/d10_ground_truth_audit.json")
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "instances": results}, f, indent=2)

    print("\nD10 AUDIT RESULTS:")
    print(f"Train Instances: {summary['train']['instance_count']}")
    print(f"  Mean Aspect Ratio (w/h): {summary['train']['mean_aspect_ratio']} (Median: {summary['train']['median_aspect_ratio']})")
    print(f"  Mean Width: {summary['train']['mean_width_px']} px | Mean Height: {summary['train']['mean_height_px']} px")
    print(f"  Small Area (<1024 px^2): {summary['train']['small_area_ratio']}%")
    print(f"\nVal Instances: {summary['val']['instance_count']}")
    print(f"  Mean Aspect Ratio (w/h): {summary['val']['mean_aspect_ratio']} (Median: {summary['val']['median_aspect_ratio']})")
    print(f"  Mean Width: {summary['val']['mean_width_px']} px | Mean Height: {summary['val']['mean_height_px']} px")
    print(f"  Small Area (<1024 px^2): {summary['val']['small_area_ratio']}%")
    print(f"\nDetailed audit saved to: {out_file}")

if __name__ == "__main__":
    audit_d10()
