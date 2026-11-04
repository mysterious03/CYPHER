# CYPhER V2.3.1 — CANONICAL EVALUATION PROTOCOL

**Document Version:** 1.0 (Immutable)  
**Standardized By:** Senior Computer Vision Research & Evaluation Team  

---

## 1. System Execution Environment

- **Operating System:** Windows 11 Home (x64)
- **Host GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, Compute Capability 8.9)
- **PyTorch Version:** 2.15.0+cu132 (CUDA 12.2)
- **Precision:** Mixed Precision (FP16 / AMP enabled for inference and training)
- **Device Placement:** `cuda:0` default with CPU fallback

---

## 2. Standardized Task Protocols

### A. Road Damage Object Detection (`M1_ROAD_DAMAGE`)
- **Canonical Holdout Set:** Complete 1,172 validation images (`data/processed/rdd2022_india/images/val/`).
- **Ground Truth Format:** Bounding boxes $[x_1, y_1, x_2, y_2]$ normalized to $[0, 1]$.
- **Deployment Operating Point:** Confidence threshold $\tau_{\text{conf}} = 0.25$.
- **Bounding Box Matching Threshold:** $\text{IoU} \ge 0.25$ (standard IEEE Global Road Damage Detection benchmark standard).
- **Matching Algorithm:** Greedy IoU matching with one-to-one assignment (highest IoU first, duplicate predictions penalised as False Positives).
- **Scale Defect Stratification:**
  - **Small Defects:** $\text{Area} < 32^2 = 1,024 \text{ px}^2$
  - **Medium Defects:** $1,024 \le \text{Area} < 96^2 = 9,216 \text{ px}^2$
  - **Large Defects:** $\text{Area} \ge 9,216 \text{ px}^2$

### B. Multi-Task Surface Condition Classification (`M2_SURFACE`)
- **Validation Dataset:** 1,172 held-out road patches (`val_patches.json`).
- **Mandatory Metrics:**
  - Raw Accuracy
  - Balanced Accuracy: $\frac{1}{C} \sum_{c=1}^C \text{Recall}_c$
  - Macro F1-Score: $\frac{1}{C} \sum_{c=1}^C F_{1, c}$ (Must ALWAYS accompany raw accuracy)
  - Confusion Matrix: $C \times C$ array with ground-truth rows and predicted columns
  - Class Support: Exact count of true instances per class

### C. Pixel-Level Segmentation (`M3_WATER`, `M4_VEHICLE_SEG`, `M5_ROAD_SCENE`)
- **Metric Formulation:**
  $$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}, \quad \text{mIoU} = \frac{1}{C} \sum_{c=1}^C \text{IoU}_c$$
  $$\text{Dice}_c = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$$
- **Authenticity Gate:** Only human-verified polygon masks are admitted into the evaluation score. Pseudo-masks are disqualified.

### D. Sensor & Calibration Gate Protocol (`M13`, `M14`, `M15`)
- When physical hardware telemetry (CAN bus, IMU, wheel speed) is unavailable, the pipeline must return `SENSOR_REQUIRED` and report qualitative visual estimates only.
- When camera intrinsics ($f_x, f_y, c_x, c_y$) and extrinsics ($h_{\text{cam}}, \theta_{\text{pitch}}$) are missing, metric dimensions (cm) must return `CALIBRATION_REQUIRED` and report 2D pixel extents only.
