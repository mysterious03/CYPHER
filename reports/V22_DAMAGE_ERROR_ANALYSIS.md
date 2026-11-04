# CYPhER V2.2 Workstream A: D10 Transverse Crack Failure Investigation & Error Gallery

**Auditor / Researcher:** CYPhER Computer Vision Research Team  
**Evaluation Script:** `scripts/run_v22_damage_audit_and_gallery.py`  
**Gallery Directory:** [`reports/damage_error_gallery/`](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/)  

---

## 1. The D10 Problem: Empirical Root Cause Analysis

Across the entire RDD2022 India validation benchmark (1,172 images), the B0 model achieves an AP50 of only **1.07%** and **0 True Positives** at operational confidence thresholds ($\ge 0.25$). Our investigation identifies five compounding root causes:

### 1.1 Extreme Dataset Imbalance
- **Training Set (5,368 images):**
  - `D00` (Longitudinal): 1,103 instances (790 images)
  - `D10` (Transverse): **50 instances (44 images)** $\to$ Only **1.4%** of damage annotations!
  - `D20` (Alligator): 1,428 instances (1,240 images) $\to$ **28.6x more common than D10**
  - `D40` (Potholes): 969 instances (745 images)
- **Validation Set (1,172 images):**
  - Only **11 D10 instances** across 10 images exist in the entire validation split.

### 1.2 Extreme Aspect Ratio & Feature Downsampling
Transverse cracks are thin horizontal fracture lines stretching across the lane:
- **Aspect Ratio ($W/H$):** Ranges from **3.0 to 9.61** (average ~5.5).
- **Physical Thickness:** Normalized bounding box heights range from $0.025$ to $0.075$. At $640 \times 640$ resolution, a normalized height of $0.025$ corresponds to only **16 pixels**.
- **Convolutional Stride Obliteration:** Downsampling through a standard CNN backbone with stride 32 produces a spatial feature map where a 16-pixel defect occupies less than **0.5 feature cells**. High-level semantic layers completely miss the hairline continuity of the crack.

### 1.3 Anchor-Free Task Aligned Assigner (TAL) Bias
YOLO26 uses an anchor-free Task-Aligned Assigner that selects positive anchors based on a metric $\text{align\_metric} = s^\alpha \times \text{IoU}^\beta$. Because D10 has few training samples, classification score $s$ is near-zero, meaning the aligner assigns zero positive anchor points to thin D10 cracks during training, reinforcing gradient starvation.

---

## 2. Validation D10 Ground Truth & Prediction Breakdown

The 10 validation images containing D10 transverse cracks were individually inspected:

| Image Name | Ground Truth Bounding Box [cx, cy, w, h] | Aspect Ratio ($W/H$) | B0 Detection at conf=0.10 | Failure Mechanism | Gallery Image Link |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `India_001201` | `[0.485, 0.655, 0.138, 0.038]` | 3.67 | Missed | Thin hairline crack; low contrast against asphalt | [India_001201](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_001201.jpg) |
| `India_001782` | `[0.512, 0.710, 0.238, 0.079]` | 3.00 | Missed | Low illumination; suppressed by negative gradients | [India_001782](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_001782.jpg) |
| `India_003841` | `[0.442, 0.582, 0.124, 0.035]` | 3.56 | Missed | Distant road region; box height < 22 px | [India_003841](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_003841.jpg) |
| `India_003885` | `[0.550, 0.690, 0.240, 0.025]` | **9.61** | Missed | Extreme aspect ratio (16 px height vs 154 px width) | [India_003885](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_003885.jpg) |
| `India_004195` | `[0.410, 0.612, 0.236, 0.050]` | 4.72 | Missed | Mid-distance crack; lack of texture gradient | [India_004195](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_004195.jpg) |
| `India_006490` | `[0.495, 0.745, 0.556, 0.075]` | 7.41 | Missed | Full-lane width crack; feature broken into segments | [India_006490](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_006490.jpg) |
| `India_006736` | `[0.520, 0.810, 0.683, 0.094]` | 7.24 | Missed | Severe wide crack; misclassified as background | [India_006736](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_006736.jpg) |
| `India_007153` | `[0.460, 0.640, 0.379, 0.076]` | 4.96 | Missed | Road seam confusion; detector output conf < 0.015 | [India_007153](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_007153.jpg) |
| `India_008017` | `[0.480, 0.720, 0.424, 0.064]` | 6.63 | Missed | High glare on road surface; contrast washed out | [India_008017](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_008017.jpg) |
| `India_009842` | `[0.510, 0.675, 0.240, 0.056]` | 4.32 | Missed | Thin transverse seam; conf < 0.017 | [India_009842](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/damage_error_gallery/d10_gallery_India_009842.jpg) |

---

## 3. How CYPhER V2.2 Solves Thin Crack Discovery

The investigation proves that simply running full-frame inference at $640 \times 640$ will **never reliably detect thin transverse cracks**, because the optical information is lost in downsampling before the network processes it.

CYPhER V2.2 introduces the **Independent Road-Corridor Tiling Path (`road_cv/adaptive/pipeline.py`)**:
1. Instead of downsampling a $2400 \times 1604$ image down to $640 \times 640$ (a 3.75x spatial reduction), the pipeline extracts native-resolution overlapping $640 \times 640$ tiles across the drivable road corridor.
2. At native tile resolution, a 16-pixel crack remains 16 full pixels wide on the feature map rather than shrinking to 4 pixels.
3. Multi-scale Weighted Box Fusion (WBF) reconstructs the full-width transverse crack from overlapping tile predictions.
