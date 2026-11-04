# CYPhER V2.3.1 — MONOCULAR DEPTH ESTIMATION VALIDATION REPORT

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Evaluated Checkpoint:** `yolo26n-depth.pt` (12.40 MB)  
**Architecture:** YOLO26n-depth Monocular Estimator  
**Output Representation:** Continuous Relative Inverse Depth $[0, 1]$  
**Taxonomy Status:** `PRETRAINED_NOT_VALIDATED`  

---

## 1. Scale Demarcation & Scientific Truth

A critical scientific correction instituted in CYPhER V2.3.1 is the formal repudiation of "metric claims" from uncalibrated monocular depth:
- Prior documentation described the model as having a "mean depth of 22.98 m."
- **Audit Finding:** The neural network outputs **strictly normalized relative inverse depth values** $[0, 1]$.
- A single RGB camera cannot mathematically determine physical metric distance without external scale references (such as LiDAR ground truth, stereo baselines, or known camera mounting geometry).
- **Mandatory Reporting Nomenclature:** All outputs from this model must be labeled **`Relative Depth`**, **`Normalized Depth`**, or **`Relative Depth Anomaly`**.

---

## 2. Empirical Spatial Continuity (100 Indian Road Scenes)

- **Monotonic Depth Gradient:** Depth values smoothly transition from the immediate bottom foreground ($y=640$, normalized relative depth $\approx 0.05$, pseudo-distance $\approx 3.7\text{ m}$) to the horizon vanishing line ($y=250$, normalized relative depth $\approx 0.95$, pseudo-distance $\approx 55\text{-}60\text{ m}$).
- **Object Boundary Sharpness:** Vehicles, pedestrians, and guardrails display sharp step-discontinuities in the depth gradient, enabling clean foreground obstacle isolation.
- **Inference Speed:** **11.2 ms** per frame on RTX 4050 GPU.

---

## 3. Metric Pothole Depth Limitation

- **Fundamental Constraint:** The relative depth map **cannot measure depression depth** (e.g., distinguishing a 2 cm shallow depression from an 8 cm tire-bursting pothole).
- **Validation Requirement:** Evaluating metric depth error (AbsRel, RMSE, MAE) requires pairing dashcam video with synchronized LiDAR point clouds (e.g., KITTI Depth Benchmark or Indian Driving Dataset LiDAR). Status: **`PRETRAINED_NOT_VALIDATED`**.
