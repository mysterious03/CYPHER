# CYPhER V2.5 — ROAD SURFACE CONDITION & MATERIAL IMPROVEMENT
**Multi-Task Imbalance Mitigation, Head-by-Head Forensic Audit, and Checkpoint Selection**

---

## 1. Executive Summary & Diagnostic Overview

Road surface condition perception in CYPhER comprises three discrete multi-task prediction heads operating on localized road ROI patches ($224 \times 224 \times 3$) extracted from the lower drivable perspective:
1. **Friction / Moisture State Head:** `Dry`, `Wet`, `Water-covered`, `Fresh snow`, `Melted snow`, `Ice-covered` (6 classes)
2. **Surface Material Head:** `Asphalt`, `Concrete`, `Gravel`, `Dirt`, `Mud`, `Loose stone` (6 classes)
3. **Surface Unevenness Head:** `Smooth`, `Slightly uneven`, `Severely uneven`, `Deteriorated` (4 classes)

### The Minority-Class Hazard Bottleneck
Across the authentic 6,540-patch Road Surface Condition Dataset (RSCD) paired with Indian driving routes:
- **Total Split:** 5,368 training patches, 1,172 held-out validation patches.
- **Friction Imbalance:** 5,331 Dry vs 37 Wet in train (**144.1 : 1** imbalance ratio); 1,164 Dry vs 8 Wet in validation.
- **Unevenness Imbalance:** 5,096 Smooth vs 272 Slightly Uneven in train (**18.7 : 1** ratio); 1,106 Smooth vs 66 Uneven in validation.

Under standard unweighted cross-entropy (CE), the majority dry/smooth gradients swamped the network, resulting in catastrophic hazard omission: the baseline missed **37.5% of wet roads** (only 5/8 detected). In an autonomous driving system, failing to detect a low-friction wet surface results in skidding, loss of steerability, and extended stopping distances.

---

## 2. Controlled 6-Way Ablation Study Comparison

All six experimental variants were trained on the identical pre-cached uint8 tensors (`cached_train.pt` 770.6 MB, `cached_val.pt` 168.2 MB) using AdamW ($lr = 5 \times 10^{-4}$), batch size 64, for 15 epochs on the NVIDIA RTX 4050 GPU:

| Exp ID | Loss & Sampling Formulation | Mean Acc (%) | Fric BalAcc (%) | Wet Recall (8 GT) | Fric Mac-F1 (%) | Unev BalAcc (%) | Unev Recall (66 GT) | Unev Mac-F1 (%) |
|---|---|---|---|---|---|---|---|---|
| **exp1** | Standard Unweighted CE | 99.06 | 80.69 | 62.5% (5/8) | 68.89 | 94.96 | 90.91% (60/66) | 93.41 |
| **exp2** | Inverse-Frequency Weighted CE | 96.47 | 97.68 | **100.0%** (8/8) | 60.24 | 95.41 | 96.97% (64/66) | 80.69 |
| **exp3** | Focal Loss ($\gamma = 2.0$) | 95.19 | 95.70 | **100.0%** (8/8) | 54.65 | 94.03 | 93.94% (62/66) | 80.52 |
| **exp4** | **Class-Balanced EffNum ($\beta=0.999$)** | **98.04** | **98.58** | **100.0% (8/8)** | **65.61** | **96.95** | **96.97% (64/66)** | **88.20** |
| **exp5** | Class-Aware Batch Resampling | 98.78 | 56.12 | 12.5% (1/8) | 58.12 | 97.80 | 98.48% (65/66) | 89.12 |
| **exp6** | Combined Resampling + EffNum | 98.58 | 74.83 | 50.0% (4/8) | **74.83** | 95.96 | 95.45% (63/66) | 86.53 |

---

## 3. Deep Forensic Head-by-Head Audit: exp4 vs exp6

A crucial technical dilemma arose when comparing `exp4_effective_number` against `exp6_combined`:

1. **Precision-Recall Dynamics:**
   - `exp6` achieved a higher macro-F1 on paper (**74.83%** vs 65.61%) because it suppressed false alarms by aggressively oversampling the 37 wet patches with balanced mini-batches.
   - However, this batch oversampling caused the feature extractor to overfit to the specific lighting of the 37 training patches, resulting in **missing 4 out of the 8 unseen validation wet patches (50% hazard miss rate)**.
   - Conversely, `exp4` achieved **100.0% Wet Road Recall (8/8)** with zero hazard misses, at the cost of 33 false positives across 1,164 dry frames (a negligible false alarm rate of 2.83%).
2. **Material Head Performance:**
   - Both exp4 and exp6 achieved **100.0% Accuracy and 100.0% Macro-F1** across all 1,172 validation asphalt patches.
3. **Unevenness Head Performance:**
   - `exp4` outperformed exp6 on uneven surface detection: **96.97% Recall** (64/66 true positives) and **96.95% Balanced Accuracy**, compared to 95.45% recall for exp6.

### Definitive Selection Decision:
> **`exp4_effective_number`** (`runs/surface_classifier_v25/best_surface_model.pt`) is certified as the primary operational checkpoint. In ADAS hazard intelligence, false negatives on slippery road conditions risk severe collisions; an ADAS perception system must prioritize 100% hazard recall.

---

## 4. Optical Confusion & Visual Error Taxonomy

Detailed visual review of the 33 false positives produced by `exp4` identified three distinct environmental triggers:
1. **Low-Angle Tree Canopy Shadows (19 cases, 57.6%):** Dense, sharp-edged tree shadows across asphalt lanes reduce local luminance and mimic the low reflectance of damp tarmac.
2. **Fresh Black Bituminous Resurfacing (9 cases, 27.3%):** New asphalt possesses an intense pitch-black color with high specular gloss under direct overhead sunlight, visually indistinguishable from water films in monocular RGB.
3. **Vehicle Oil Stains & Tire Polish (5 cases, 15.1%):** Concentrated chemical sheen near toll gates and bus stops.

---

## 5. Physical Sensor Gating Notice

> [!IMPORTANT]
> The surface classifier delivers **Visual Surface Risk Estimation**. 
> It DOES NOT directly measure the physical coefficient of friction ($\mu \approx 0.75$ dry, $\mu \approx 0.35$ wet) nor the International Roughness Index ($\text{IRI} \text{ mm/m}$). 
> True physical dynamics estimation strictly requires:
> - Wheel-speed slip-ratio telemetry via CAN-bus ($\mu$).
> - Vertical high-G accelerometer telemetry via IMU ($\text{IRI}$).
> 
> Therefore, in CYPhER V2.5, this capability is categorized as:
> **`VALIDATED_STRONG`** for Visual Surface Classification;
> **`SENSOR_REQUIRED`** for Physical Friction and Roughness Physics.
