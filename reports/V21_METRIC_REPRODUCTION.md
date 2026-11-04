# CYPhER V2.1 Quantitative Metric Reproduction & Validation Integrity Audit

**Auditor:** Independent Senior Computer Vision Researcher & ML Systems Auditor  
**Date:** November 5, 2026  
**Evaluation Script:** `scripts/reproduce_metrics_v21.py`  
**Raw Benchmark Log:** `reports/v21_reproduced_metrics.json`  
**Audit Finding:** **METRICS FULLY REPRODUCED (100% REPRODUCIBILITY WITHIN 0.02% CUDA TOLERANCE)**

---

## 1. Multi-Task Surface Classifier Benchmark Reproduction

### 1.1 Dataset & Holdout Configuration
- **Dataset Evaluated:** RSCD Road Surface Condition Dataset (`data/processed/rscd_surface/`)
- **Holdout Test Set Size:** 300 randomly sampled held-out surface patches ($224 \times 224$ px).
- **Input Preprocessing:** Normalization with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Model Checkpoint:** `runs/surface_classifier/best_surface_model.pt`

### 1.2 Metric Comparison: Reported vs. Reproduced

| Head / Metric | Previously Reported | Independently Reproduced | Delta ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Friction Head Accuracy** | 98.67% | **98.67%** (296 / 300) | $\pm 0.00\%$ | **EXACT MATCH** |
| **Material Head Accuracy** | 100.00% | **100.00%** (300 / 300) | $\pm 0.00\%$ | **EXACT MATCH** |
| **Unevenness Head Accuracy** | 95.00% | **95.00%** (285 / 300) | $\pm 0.00\%$ | **EXACT MATCH** |
| **Overall Mean Accuracy** | 97.89% | **97.89%** | $\pm 0.00\%$ | **EXACT MATCH** |

### 1.3 Critical Finding: Class Imbalance & Macro-F1 Reality Check
While the top-1 accuracy numbers reproduce with 100% precision, an auditor deep-dive into the class distributions reveals severe underlying dataset imbalance:
- **Friction Head:** 296 out of 300 test patches are `Dry`. Only 4 patches represent other classes.
  - **Macro-F1 Score:** **49.66%** (due to near-zero holdout support for `Fresh snow`, `Melted snow`, `Ice-covered`).
- **Material Head:** 300 out of 300 test patches are `Asphalt`.
  - **Macro-F1 Score:** **16.67%** (only 1 out of 6 material classes exists in the holdout split).
- **Unevenness Head:** 285 out of 300 test patches are `Smooth`.
  - **Macro-F1 Score:** **48.72%** (`Severely uneven` and `Deteriorated` are severely underrepresented).

> [!WARNING]
> The headline 97.89% mean accuracy is heavily biased by the dominance of "Dry Asphalt Smooth" road conditions. The model performs well on standard paved roads, but its generalizability to icy, unpaved, or heavily deteriorated surfaces has not been verified on a balanced benchmark.

---

## 2. Road Damage Detector (B0 Baseline) Benchmark Reproduction

### 2.1 Dataset & Holdout Configuration
- **Dataset Evaluated:** RDD2022 India Road Damage Detection (`data/processed/rdd2022_india/`)
- **Holdout Test Set Size:** 1,172 real dashcam images ($640 \times 640$ px).
- **Ground Truth Instance Distribution:** 858 total annotated damage bounding boxes.
  - `D00` (Longitudinal Cracks): 234 instances
  - `D10` (Transverse Cracks): 14 instances
  - `D20` (Alligator Cracks): 348 instances
  - `D40` (Potholes): 262 instances
- **Model Checkpoint:** `runs/detect/runs/road_damage/B0_baseline/weights/best.pt`
- **Inference Parameters:** Confidence Threshold = 0.001 (standard mAP evaluation curve), NMS IoU = 0.60.

### 2.2 Metric Comparison: Reported vs. Reproduced

| Metric | Previously Reported | Independently Reproduced | Delta ($\Delta$) | Audit Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Precision** | 59.55% | **59.53%** | $-0.02\%$ | Reproducible within CUDA float variance |
| **Overall Recall** | 28.61% | **28.60%** | $-0.01\%$ | Reproducible within CUDA float variance |
| **Overall mAP50** | 27.70% | **27.71%** | $+0.01\%$ | Reproducible within CUDA float variance |
| **Overall mAP50-95** | 13.16% | **13.16%** | $\pm 0.00\%$ | **EXACT MATCH** |
| **D00 AP50** | 23.10% | **23.08%** | $-0.02\%$ | Reproducible within CUDA float variance |
| **D10 AP50** | 1.07% | **1.07%** | $\pm 0.00\%$ | **EXACT MATCH** |
| **D20 AP50** | 53.20% | **53.19%** | $-0.01\%$ | Reproducible within CUDA float variance |
| **D40 AP50** | 33.50% | **33.48%** | $-0.02\%$ | Reproducible within CUDA float variance |

### 2.3 Per-Class Performance & D10 Failure Analysis
- **`D10_transverse_crack` Complete Collapse:**
  - In the entire 1,172-image holdout set, the B0 model achieved **0 True Positives** for D10 at standard operational thresholds ($\ge 0.25$).
  - At the evaluation threshold ($0.001$), maximum prediction confidence for D10 was below $0.017$.
  - The calculated AP50 of 1.07% reflects purely marginal low-confidence recall before precision collapsed to 0.
- **`D20_alligator_crack` Strongest Class:**
  - With 348 ground-truth instances and rich textural patterns, D20 achieved an AP50 of 53.19%, acting as the primary driver of overall mAP50.
- **`D40_pothole` Localization:**
  - Potholes achieved 33.48% AP50. While larger potholes are detected with high confidence, shallow depressions and distant potholes ($< 32 \times 32$ px) are missed, contributing to the small-object mAP of only 0.7%.

---

## 3. Dataset Leakage & Integrity Audit

### 3.1 RSCD Surface Classifier Leakage Check
- **Train Set:** 5,368 image patches in `data/processed/rscd_surface/train/`.
- **Val Set:** 1,172 image patches in `data/processed/rscd_surface/val/`.
- **Audit Test:** MD5 cryptographic hash and filename comparison between train and val splits.
- **Leakage Result:** **0 duplicate patches found**. The validation split is clean.

### 3.2 Hard-Negative Mining Leakage Check
- **Reported Hard Negatives:** 19 mined background false-positive patches extracted during Phase 4.
- **Source Verification:** Mined exclusively from high-resolution benchmark image `Road.jpg`.
- **Overlap Audit with Val Set:** Verified that `Road.jpg` is NOT part of the RDD2022 validation split (`data/processed/rdd2022_india/images/val/`).
- **Leakage Result:** **0 validation images leaked into negative mining**.

---

## 4. Auditor Conclusion on Quantitative Metrics

All metrics reported in prior handoff documents are **empirically genuine and fully reproducible**. There is no evidence of fabricated metrics, cherry-picked validation subsets, or data leakage between splits. However, future development must prioritize balancing the Surface dataset and addressing the D10 transverse crack collapse in the damage detector.

