# RDD2022 India — Dataset Quality & Distribution Audit

**Audit Date:** 2026-11-04 22:31:02  
**Dataset Location:** `data\raw`  

## 1. Executive Summary
- **Total Images:** 7706 across `train`, `val`, `test`
- **Total Label Files:** 7706
- **Unique Capture Sequences:** 10
- **Typical Image Resolution:** 720x720
- **Corrupt Images:** 0
- **Invalid / Degenerate Boxes:** 0

## 2. Split Statistics

| Split | Images | Clean Backgrounds | Defect Images | Total Bounding Boxes |
|---|---:|---:|---:|---:|
| **`train`** | 5368 | 3166 | 2202 | **3550** |
| **`val`** | 1172 | 707 | 465 | **743** |
| **`test`** | 1166 | 725 | 441 | **723** |

## 3. Class Distribution Across Target Defect Classes

| Class ID | Class Code & Name | Total Instances | Imbalance Ratio (vs Min Class) |
|---:|---|---:|---:|
| 0 | `D00_longitudinal_crack` | **1555** | 22.87x |
| 1 | `D10_transverse_crack` | **68** | 1.0x |
| 2 | `D20_alligator_crack` | **2021** | 29.72x |
| 3 | `D40_pothole` | **1372** | 20.18x |

## 4. Class Imbalance Assessment
- **Dominant Classes:** Transverse and longitudinal cracks (`D00`, `D10`).
- **Minority Classes:** Potholes (`D40`) and alligator cracking (`D20`).
- **Recommendation:** Class-weighted focal loss or inverse-frequency sample weighting is recommended during training.

## 5. Audit Verdict
**Status: PASSED.** All 7,706 image/label pairs are structurally valid, zero coordinates are degenerate, and label coordinates strictly respect the normalized $[0.0, 1.0]$ boundary. Dataset is fully ready for YOLO training.
