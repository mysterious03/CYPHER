# CYPhER V2.3.1 — ADAPTIVE PERCEPTION FULL BENCHMARK & TRADE-OFF REPORT

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Evaluated Capability:** Adaptive Perception Pipeline Strategies  
**Dataset:** Full 1,172 Held-Out Validation Images (743 GT Defects)  
**Operating Threshold:** $\tau_{\text{conf}} = 0.25$, $\text{IoU}_{\text{match}} = 0.25$  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)  

---

## 1. Full 1,172-Image Strategic Comparison

The CYPhER adaptive perception framework orchestrates multiple dynamic vision policies to balance detection precision, recall, latency, and GPU footprint:

| Strategy / Profile | Precision (%) | Recall (%) | F1-Score (%) | True Positives | False Positives | False Negatives | Small Recall (<32²) | Med Recall (32-96²) | Large Recall (>96²) | ECE (Calib Error) | Median Latency | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`original_b0` (Standard Full-Frame)** | 67.71% | 32.17% | 43.62% | 239 | 114 | 504 | 0.0% | 12.72% | 41.34% | 0.2657 | **10.38 ms** | **56.34 MB** |
| **`adaptive_spatial_prior` (Recall Profile)** | 60.50% | **38.76%** | **47.25%** | **288** | 188 | **455** | 0.0% | **17.11%** | **49.02%** | **0.2425** | 10.80 ms | 56.33 MB |
| **`adaptive_crops` (Precision Profile)** | 76.18% | 32.71% | 45.77% | 243 | 76 | 500 | 0.0% | 14.04% | 41.54% | 0.3108 | 11.01 ms | 56.33 MB |
| **`full_adaptive` (Balanced Profile)** | **76.49%** | 32.84% | 45.95% | 244 | **75** | 499 | 0.0% | 14.47% | 41.54% | 0.3204 | 27.68 ms | 98.04 MB |

---

## 2. In-Depth Operational Trade-Off Analysis

### Trade-Off 1: The Precision Profile (`adaptive_crops`)
- **Mechanism:** Dynamically extracts a focused RoI around the lower drivable road surface.
- **Empirical Impact:** False Positives drop by **33.3%** (from 114 to 76), eliminating tree shadows, roof textures, and roadside clutter.
- **Operational Benefit:** Precision increases from 67.71% to **76.18%** (+8.47 percentage points) with negligible latency cost (+0.63 ms). Highly recommended for high-speed highway driving where false alarms trigger unwanted braking.

### Trade-Off 2: The Recall Profile (`adaptive_spatial_prior`)
- **Mechanism:** Applies perspective-aware dynamic confidence thresholds (e.g., lower threshold $\tau=0.18$ in high-likelihood near-wheel paths; higher threshold $\tau=0.35$ near horizon).
- **Empirical Impact:** True Positives increase by **+49 detections** (from 239 to 288), driving overall recall up to **38.76%** (+6.59 percentage points) and achieving the highest F1-score (**47.25%**).
- **Trade-Off:** False positives rise from 114 to 188. Recommended for municipal road maintenance survey vehicles where capturing every defect is prioritized over false alarms.

### Trade-Off 3: Uncertainty & Probability Calibration (ECE)
- Raw model confidence is **not** a calibrated probability. The Expected Calibration Error (ECE) reveals:
  - Standard B0 has an ECE of 0.2657 (significant overconfidence on easy alligator cracks).
  - Spatial Prior achieves the best calibration ($\text{ECE} = \mathbf{0.2425}$), because perspective gating aligns prediction confidence with physical spatial defect probability.

---

## 3. Computational Overhead and Model Invocations

| Policy | Invocations per Frame | Processed Pixels | Frame Rate (RTX 4050) | Recommended Deployment Scenario |
| :--- | :---: | :---: | :---: | :--- |
| **`original_b0`** | 1 | $640 \times 640 = 409,600$ | ~96 FPS | Resource-constrained edge compute |
| **`adaptive_crops`** | 1 | $640 \times 640 = 409,600$ | ~91 FPS | Production ADAS highway cruising |
| **`adaptive_spatial_prior`**| 1 | $640 \times 640 = 409,600$ | ~93 FPS | Asset management road survey vans |
| **`full_adaptive`** | 2 | $819,200$ | ~36 FPS | Stationary / slow municipal inspection |
