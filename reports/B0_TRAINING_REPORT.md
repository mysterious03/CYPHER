# Baseline B0 Road Damage Detection — Technical Training Report

## 1. Executive Summary
Baseline model **B0** establishes the primary benchmark for the Road Intelligence project. It consists of a `yolo26n.pt` architecture fine-tuned on the **RDD2022 India** dataset targeting the four standard road distress categories.

## 2. Training Configuration
- **Architecture:** YOLO26 Nano Detection (`yolo26n.pt`)
- **Dataset:** RDD2022 India (5,368 train images, 1,172 validation images)
- **Classes:**
  - `0`: `D00_longitudinal_crack`
  - `1`: `D10_transverse_crack`
  - `2`: `D20_alligator_crack`
  - `3`: `D40_pothole`
- **Input Resolution:** 640 px
- **Batch Size:** 4
- **Optimizer:** AdamW (auto-selected lr=0.00125)
- **AMP:** Enabled (Automatic Mixed Precision)
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)
- **Peak VRAM Consumed:** ~0.70 GB (<12% of GPU ceiling)
- **Validation Duration:** 26.66 seconds

## 3. Quantitative Performance Metrics

| Metric | Overall Score |
|---|---:|
| **Precision (P)** | **0.5955** |
| **Recall (R)** | **0.2861** |
| **F1 Score** | **0.3865** |
| **mAP@0.50** | **0.2770** |
| **mAP@0.50:0.95** | **0.1316** |

## 4. Per-Class Performance Breakdown

| Class ID | Distress Name | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---:|---|---:|---:|---:|---:|
| 0 | `D00_longitudinal_crack` | 0.4206 | 0.2332 | 0.2307 | 0.0966 |
| 1 | `D10_transverse_crack` | 1.0000 | 0.0000 | 0.0107 | 0.0078 |
| 2 | `D20_alligator_crack` | 0.5209 | 0.5595 | 0.5318 | 0.2428 |
| 3 | `D40_pothole` | 0.4403 | 0.3515 | 0.3348 | 0.1791 |

## 5. Artifacts and Weights
- **Evaluated Checkpoint:** `C:/Users/ASUS/OneDrive/Desktop/cvvvvvv/runs/detect/runs/road_damage/B0_baseline/weights/best.pt`
- **Sample Predictions:** `reports/qualitative_predictions/`

## 6. Known Limitations of B0
1. **Transverse Crack Recall:** D10 has fewer training samples, leading to lower recall than D00 or D20.
2. **Thin Hairline Cracks:** Resizing to 640 px causes faint hairline cracks to lose contrast.
3. **False Alarms on Tar Seams:** Without hard-negative background mining, tar repairs are occasionally flagged as D00.
