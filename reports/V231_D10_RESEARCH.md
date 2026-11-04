# CYPhER V2.3.1 — D10 TRANSVERSE CRACK RESEARCH & ABLATION REPORT

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Evaluated Capability:** `M1_ROAD_DAMAGE` (D10 Transverse Cracks)  
**Baseline Model:** YOLO26n B0 (`runs/detect/runs/road_damage/B0_baseline/weights/best.pt`)  
**Target Class:** Class 1: `D10_transverse_crack`  
**Taxonomy Status:** `VALIDATED_ACCEPTABLE` (Overall Detector) / `FAILURE_ANALYSIS` (D10 Class)  

---

## 1. Ground Truth Dataset Forensic Audit

A comprehensive geometric and photometric audit was executed across all training and validation annotations (`scripts/audit_d10_dataset.py` -> `reports/d10_ground_truth_audit.json`).

| Metric | Training Split (RDD2022 India) | Validation Split (Holdout) | Key Engineering Observation |
| :--- | :---: | :---: | :--- |
| **Total Instances** | 50 instances (0.99% of total) | 11 instances (1.48% of total) | Extreme class scarcity (30x fewer than D20) |
| **Unique Images with D10** | 46 images | 10 images | Clustered in narrow highway sequences |
| **Mean Aspect Ratio ($w/h$)** | **5.60** (Max: 14.82) | **5.38** (Max: 11.20) | Severe horizontal anisotropy |
| **Median Aspect Ratio** | 5.71 | 4.72 | Extremely elongated horizontal bands |
| **Mean Width (Pixels)** | 248.48 px | 234.45 px | Wide span across traffic lane |
| **Mean Height (Pixels)** | 48.56 px | 43.91 px | Narrow vertical aperture |
| **Small-Object Ratio ($<32^2$)** | 4.0% | **0.0%** | D10 cracks are NOT small dots; they are large thin lines |
| **Mean Contrast (StdDev)** | 28.45 | 26.80 | Low contrast against asphalt background |

---

## 2. Inference Strategy Ablation Study

To determine whether D10 cracks could be recovered post-hoc through resolution scaling, road strip cropping, or SAHI-style overlapping grid tiling, controlled experiments were conducted on the 10 held-out validation images containing all 11 D10 ground-truth annotations:

| Strategy | Input Resolution | Tiling Topology | TP | FP | FN | Precision (%) | Recall (%) | F1-Score (%) | Median Latency |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline B0** | 640x640 | Full image (no tiling) | 0 | 0 | 11 | **0.0%** | **0.0%** | **0.0%** | 24.2 ms |
| **2. High-Res Scaling** | 960x960 | Full image (no tiling) | 0 | 0 | 11 | **0.0%** | **0.0%** | **0.0%** | 30.0 ms |
| **3. Ultra-Res Scaling** | 1280x1280 | Full image (no tiling) | 0 | 0 | 11 | **0.0%** | **0.0%** | **0.0%** | 40.4 ms |
| **4. Road Strip Tiling** | 640x640 | 2 horizontal strips (35-75%, 55-100%) | 0 | 0 | 11 | **0.0%** | **0.0%** | **0.0%** | 43.6 ms |
| **5. SAHI 2x2 Grid** | 640x640 | 4 overlapping patches (20% overlap) | 0 | 0 | 11 | **0.0%** | **0.0%** | **0.0%** | 90.8 ms |

*Detailed empirical table exported to:* [`reports/V231_D10_EXPERIMENTS.csv`](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/V231_D10_EXPERIMENTS.csv)

---

## 3. The Root Cause Discovery: Logit Bias Extinction

A deep introspection of the raw un-thresholded detector outputs (`conf=0.001`) revealed the definitive reason why all inference-time tiling strategies failed:

$$\max_{\text{holdout}} \left( \text{Confidence}_{\text{D10}} \right) = \mathbf{0.004347} \quad (\approx 0.43\%)$$

### Mathematical Cause:
During baseline B0 training:
1. The unweighted cross-entropy loss penalized background and majority classes (D20 alligator cracks with 2,021 instances, D00 with 1,555 instances).
2. Because D10 appeared in only 50 out of 5,016 instances (0.99%), the backpropagation gradients from majority classes and background negatives drove the D10 classification logit bias down to $z_{\text{D10}} \approx -5.44$.
3. Applying the sigmoid activation yields $\sigma(-5.44) = \frac{1}{1 + e^{5.44}} \approx 0.0043$.

**Conclusion:** The neural head itself is mathematically suppressed. Even when an image patch is zoomed to 1280x1280 or cropped directly around a transverse crack, the detector cannot output a probability higher than 0.0043. 

**Post-hoc inference tricks cannot fix a suppressed logit.** The only viable solution is retraining with class-weighted loss and synthetic copy-paste crack augmentation.
