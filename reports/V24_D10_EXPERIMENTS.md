# CYPhER V2.4 — D10 TRANSVERSE CRACK RESEARCH & RETRAINING
**Forensic Root Cause Analysis, Synthetic Inference Limitations, and Copy-Paste Augmentation**

---

## 1. Executive Summary & Root Cause Analysis

In all prior CYPhER versions (B0 Baseline, V2.2, V2.3), the model consistently scored **0.00% recall on D10 transverse cracks** across the 1,172 held-out validation images (0 true positives out of 11 ground-truth instances).

### 1.1 Forensic Geometry Audit
Our deep ground-truth audit of all 61 D10 instances in RDD2022 India (`reports/d10_ground_truth_audit.json`) uncovered extreme physical characteristics:
- **Aspect Ratio ($w/h$):** Mean aspect ratio of **5.60** (median 5.71, maximum 10.18).
- **Physical Dimensions:** Mean width of **248.5 pixels**, mean height of only **48.6 pixels**.
- **Orientation:** Orthogonal to the driving path, spanning lane widths horizontally across the bottom camera perspective ($y \in [0.65h, 0.95h]$).
- **Severe Class Imbalance:** In the authentic training split:
  - D00 (Longitudinal): 1,024 instances
  - D20 (Alligator): 663 instances
  - D40 (Pothole): 1,146 instances
  - **D10 (Transverse): only 50 instances** ($1.73\%$ of all defects).

### 1.2 Mathematical Logit Suppression Discovery
We inspected the raw pre-sigmoid logits output by the B0 detector on the 11 validation instances. 
Because 98.27% of positive training targets and billions of background grid cells in RDD2022 lacked D10, the classification head weights for D10 were driven strongly negative during baseline training.
- **Maximum validation confidence score for D10:** $\le 0.0043$ ($0.43\%$).
- **Implication:** The detector was mathematically suppressed. At any standard confidence threshold ($\tau \ge 0.10$), D10 was completely invisible.

---

## 2. Exhaustive Audit of Inference-Only Strategies

In V2.3.1, we tested five post-hoc inference and tiling strategies on the 11 validation instances without retraining (`scripts/test_d10_inference_strategies.py`):

| Inference Strategy | Input Resolution | Tiling / Slicing Grid | True Positives (TP) | False Negatives (FN) | D10 Recall (%) |
|---|---|---|---|---|---|
| Standard Fixed 640 | $640 \times 640$ | None (Full frame) | 0 | 11 | **0.00%** |
| High-Resolution 960 | $960 \times 960$ | None (Bilinear upsample) | 0 | 11 | **0.00%** |
| Ultra-Resolution 1280 | $1280 \times 1280$ | None (Full sensor scale) | 0 | 11 | **0.00%** |
| Road Strip Crop | $640 \times 640$ | Lower trapezoid ($y \in [0.5h, 1.0h]$) | 0 | 11 | **0.00%** |
| SAHI Sliced Inference | $640 \times 640$ | $2 \times 2$ grid with 20% overlap | 0 | 11 | **0.00%** |

### Definitive Conclusion on Inference-Time Workarounds
> [!CAUTION]
> Slicing, tiling, and resolution scaling cannot recover a class whose classification head outputs $\le 0.0043$ confidence. The model *must* be retrained with gradient updates that actively un-suppress the D10 classification weights.

---

## 3. CYPhER V2.4 Copy-Paste Augmentation Architecture

To resolve the class starvation without collecting external synthetic datasets, CYPhER V2.4 designed an **Authentic Copy-Paste Augmentation Engine** (`scripts/prepare_d10_copypaste_dataset.py`):

### 3.1 Data Preparation Rules
1. **Source Isolation:** Extracted all 50 authentic D10 transverse cracks exclusively from `data/processed/rdd2022_india/images/train/`.
2. **Photometric LAB Alignment:** Transferred mean luminance ($L$) from target clean road regions to source crack patches to prevent unrealistic contrast boundaries:
   $$L_{\text{matched}} = \text{clip}(L_{\text{src}} + 0.70 \cdot (\bar{L}_{\text{dst}} - \bar{L}_{\text{src}}), 0, 255)$$
3. **Edge Feathering:** Applied a continuous 4-to-6 pixel soft feathered alpha mask to eliminate rectangular boundary seams.
4. **Perspective-Aware Placement:** Pasted patches into lower drivable road regions ($y \in [0.65h, 0.90h]$) with random horizontal flip and scale jitter ($\pm 20\%$).
5. **Strict Validation Segregation:**
   - Augmented images: 250 generated with 337 D10 instances.
   - Total training set: 5,618 images (5,368 original + 250 augmented).
   - **Validation set:** Exactly 1,172 original authentic images. **Zero synthetic or copy-paste data in validation.**

---

## 4. Retraining & Evaluation Protocol

- **Architecture:** YOLO26n.
- **Initialization:** Warm-start from verified B0 checkpoint (`runs/detect/road_damage_v22/weights/best.pt`).
- **Optimization:** AdamW optimizer, batch size 16, AMP enabled, fine-tuning for 10 epochs on NVIDIA RTX 4050 GPU.
- **Output Artifacts:** `runs/detect/road_damage_v24/weights/best.pt` and `reports/V24_DAMAGE_COMPARISON.csv`.

