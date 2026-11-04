# CYPhER V2.2 Workstream A: Road Damage Candidate Audit & Decision Report

**Auditor / Researcher:** CYPhER Computer Vision Research Team  
**Evaluation Script:** `scripts/run_v22_damage_audit_and_gallery.py`  
**Dataset:** RDD2022 India (1,172 untouched holdout validation images, 858 GT damage boxes)  
**Output CSV Artifact:** [`reports/V22_DAMAGE_EXPERIMENTS.csv`](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/V22_DAMAGE_EXPERIMENTS.csv)  

---

## 1. Candidate Checkpoint Provenance Audit

Before comparing metrics, an exhaustive audit of training logs, training arguments (`args.yaml`), and loss histories (`results.csv`) was conducted to establish candidate provenance:

| Candidate | Checkpoint Path | Trained Epochs | Training Time | Loss Formulation | Augmentation Strategy | Sampling Strategy | Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **`B0_baseline`** | `runs/detect/runs/road_damage/B0_baseline/weights/best.pt` | **38** | 5,944 s (~1.65 h) | Standard YOLO26n Box/Cls/DFL | Standard Mosaic + Flips | Natural RDD2022 distribution | **Fully Converged** |
| **`B1A_sampling_only`** | `runs/detect/runs/road_damage/B1A_sampling_only/weights/best.pt` | **1** | 142 s | Standard Box/Cls/DFL | Standard Mosaic | 3x D10 duplicate oversampling | **Unconverged Pilot (1 Epoch)** |
| **`B1B_weighting_only`** | `runs/detect/runs/road_damage/B1B_weighting_only/weights/best.pt` | **1** | 137 s | Class-balanced loss weights (D10 7.14x) | Standard Mosaic | Natural distribution | **Unconverged Pilot (1 Epoch)** |
| **`B1C_aug_only`** | `runs/detect/runs/road_damage/B1C_augmentation_only/weights/best.pt` | **1** | 242 s | Standard Box/Cls/DFL | Perspective + HSV tuning | Natural distribution | **Unconverged Pilot (1 Epoch)** |

> [!CRITICAL]
> **Provenance Discovery:** Checkpoints B1A, B1B, and B1C were not full training experiments. They were **single-epoch initial exploration runs** halted after ~140–240 seconds. Comparing a 38-epoch converged model (B0) against 1-epoch models explains the drastic metric drop across B1 candidates.

---

## 2. Quantitative Performance Matrix (Identical 1,172 Validation Images)

Evaluated under identical image preprocessing, IoU matching threshold ($\text{IoU} \ge 0.25$), and confidence evaluation curves:

| Metric | B0 Baseline (Converged) | B1A (1-Epoch Sampling) | B1B (1-Epoch Weighting) | B1C (1-Epoch Augmentation) |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | **59.53%** | 7.44% | 0.28% | 2.17% |
| **Recall** | 28.60% | 12.13% | **57.12%** | 23.90% |
| **F1 Score** | **38.64%** | 9.22% | 0.55% | 3.97% |
| **mAP50** | **27.71%** | 3.31% | 3.25% | 3.36% |
| **mAP50-95** | **13.16%** | 0.88% | 0.87% | 0.86% |
| **D00 AP50** | **23.08%** | 2.15% | 2.74% | 1.61% |
| **D10 AP50** | **1.07%** | 0.00% | 0.00% | 0.00% |
| **D20 AP50** | **53.19%** | 7.55% | 7.49% | 9.25% |
| **D40 AP50** | **33.48%** | 3.53% | 2.76% | 2.60% |
| **D10 TP (conf=0.25)** | 0 / 11 | 0 / 11 | 0 / 11 | 0 / 11 |
| **Total False Positives (conf=0.25)** | 114 | 0 | 0 | 0 |
| **Peak VRAM** | 505.5 MB | 523.8 MB | 539.8 MB | 523.8 MB |

---

## 3. Deconstruction of the "57.12% Recall" in B1B

Earlier project notes cited a "57.12% recall improvement" in B1B. Our dynamic evaluation deconstructs this claim:
1. **Precision Collapsed to 0.28%:** At epoch 1, weighting the loss of rare classes by 7.14x without sufficient training epochs caused the classification logits to fire unconstrained predictions across the entire feature grid.
2. **True Positive Reality:** At standard operational confidence ($\ge 0.25$), B1B produced **0 True Positives** for D10. The 57.12% recall was computed at the floor evaluation threshold ($0.001$), where thousands of low-confidence bounding boxes flooded the image.
3. **mAP50 Collapsed to 3.25%:** Because precision was near-zero, overall mAP50 dropped by $88.3\%$ relative to B0 ($27.71\% \to 3.25\%$).

---

## 4. Final Selection Verdict & Preservation

Per Non-Negotiable Engineering Rule 1 and Prompt Section A4:
- **Decision:** **`B0_baseline` is unequivocally selected as the winning foundation detector.**
- **Preservation Action:** The verified weights from `runs/detect/runs/road_damage/B0_baseline/weights/best.pt` have been copied and preserved as the official V2.2 road damage checkpoint:
  `runs/detect/road_damage_v22/weights/best.pt`
- Neither B0 nor the exploratory B1 directories have been overwritten or modified.
