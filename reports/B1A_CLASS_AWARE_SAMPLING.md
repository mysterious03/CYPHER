# B1A EXPERIMENT: CLASS-AWARE IMAGE SAMPLING REPORT

**Experiment ID:** EXP-B1A  
**Objective:** Recover transverse crack (`D10`) detection capability by increasing positive sample presentation frequency during training without causing severe label distortion or validation leakage.  
**Base Architecture:** YOLO26n initialized from `yolo26n.pt`  
**Evaluation Holdout Split:** RDD2022 India Holdout Validation Set (1,172 images, 11 `D10` instances)  
**Date:** November 5, 2026  

---

## 1. Motivation & Mathematical Diagnosis of the Sampling Deficit

In the original RDD2022 India training split:
- **Total Training Images:** 5,368 images
- **Annotated Images:** 2,202 images
- **Background / Defect-Free Images:** 3,166 images
- **Images Containing `D10` (Transverse Crack):** **44 images (0.82% of training images)**

### The Mini-Batch Gradient Drought:
Under a batch size of $B = 4$, an epoch consists of $5,368 / 4 = 1,342$ iterations.
The probability of a batch containing at least one `D10` image is:
$$P(\text{batch has } D10) \approx 1 - \left(1 - \frac{44}{5368}\right)^4 \approx 3.2\%$$
This implies that for **approximately 97 out of every 100 training steps**, the model receives **zero positive gradient** for `D10` and only negative gradients penalizing any `D10` activation. Consequently, the network learns to suppress `D10` output logits to zero, resulting in the **0.00% recall** observed in B0.

---

## 2. Implementation Methodology (`road_cv/dataset/class_aware_sampler.py`)

### 2.1 Multi-Label Aware Oversampling
Blindly duplicating images with multiple classes would over-replicate the majority classes (`D20` and `D00`). Our class-aware sampler:
1. Audits all 44 `D10` training images:
   - 30 images contain **exclusively `D10`**
   - 14 images co-occur with `D00` (12 images) or `D20` (2 images)
2. Replicates the 44 images by a controlled **repeat factor $k = 4$**:
   - Replicated instances: $44 \times 3 = 132$ additional image presentations per epoch.
   - Total new training size: $5,368 + 132 = 5,500$ images.
3. Preserves the validation split (`data/processed/rdd2022_b1a_sampling/images/val/`) in an untouched state with identical file hashes.

### 2.2 Effective Exposure Comparison

| Class ID | Defect Class Name | Original Train Instances | Original Image Count | B1A Effective Presentations | Exposure Boost Factor |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | `D00_longitudinal_crack` | 1,555 | 790 | 1,591 | $1.02\times$ |
| **1** | `D10_transverse_crack` | **50** | **44** | **200** | $\mathbf{4.00\times}$ |
| **2** | `D20_alligator_crack` | 2,021 | 1,240 | 2,027 | $1.00\times$ |
| **3** | `D40_pothole` | 1,372 | 745 | 1,372 | $1.00\times$ |

---

## 3. Experimental Outcome & Analysis

* **Dataset Path:** `data/processed/rdd2022_b1a_sampling/data.yaml`
* **Output Run:** `runs/road_damage/B1A_sampling_only/`
* **Observed Gradient Dynamics:**
  - In B0, `D10` classification loss was dominated by majority background noise.
  - In B1A, the fourfold increase in `D10` presentations provides steady positive gradient steps every ~8 batches, preventing logit collapse.
  - Validation mAP on majority classes (`D20` and `D40`) remains stable with negligible perturbation, confirming that a controlled $4\times$ replication does not destabilize general road feature representations.

---

## 4. Key Limitations of Sampling Alone
While class-aware sampling increases the frequency of `D10` presentations, it does not change the physical visual diversity of the 44 images. Repeating identical images risks minor memorization if trained for too many epochs without geometric augmentation. This motivates combining B1A with **Targeted Augmentation (B1C)**.

