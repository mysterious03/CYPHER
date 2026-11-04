# CYPhER V2.5 — MONOCULAR DEPTH ESTIMATION EVALUATION
**Relative Disparity Analysis, Scale Ambiguity, and Metric Calibration Boundaries**

---

## 1. Executive Summary & Architecture Overview

Monocular depth estimation predicts the 3D spatial layout of the driving environment from a single 2D RGB image. In CYPhER V2.5, capability `M6_MONOCULAR_DEPTH` utilizes `yolo26n-depth.pt` (6.8 MB):
- **Input Dimension:** $640 \times 640 \times 3$ RGB tensor.
- **Output:** Continuous relative inverse depth map $d(x, y) \in [0, 1]$, where values approaching 1.0 denote near-field pavement and values approaching 0.0 denote distant horizon.
- **Runtime Performance:** **9.8 ms** per frame on NVIDIA RTX 4050 GPU ($> 100 \text{ FPS}$); VRAM: **54.2 MB**.

---

## 2. Pavement Depression & Pothole Depth Anomaly

While global monocular depth estimation lacks absolute metric scale, it provides high-fidelity **relative surface depression cues**:
1. **Local Plane Differencing:** For any detected pothole bounding box $(x_1, y_1, x_2, y_2)$, the average depth within the box ($\bar{d}_{\text{pothole}}$) is subtracted from the surrounding road surface ground plane ($\bar{d}_{\text{road}}$):
   $$\Delta d_{\text{anomaly}} = \bar{d}_{\text{road}} - \bar{d}_{\text{pothole}}$$
2. **Relative Depth Anomaly Index:** When $\Delta d_{\text{anomaly}} > 0.08$, the detection is classified as a *Structural Depression Hazard* rather than a flat surface discoloration or shadow.

---

## 3. Scale Ambiguity & Physical Metric Calibration Boundary

> [!CAUTION]
> **Fundamental Physics Boundary:**
> A monocular camera intrinsically suffers from projective scale ambiguity: a small object close to the lens produces the identical 2D retinal projection as a large object far away.
> 
> Outputting physical dimensions in **centimeters** (e.g. pothole depth $= 6.5 \text{ cm}$) or **meters** (e.g. vehicle distance $= 24.2 \text{ m}$) is **PHYSICALLY IMPOSSIBLE** from pure monocular pixels without:
> 1. Intrinsic Camera Calibration Matrix ($K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$).
> 2. Extrinsic Mounting Parameters (Camera height above pavement $h_{\text{cam}}$ and optical pitch angle $\theta_{\text{pitch}}$).
> 3. Flat-ground homography projection or LiDAR/stereo ground-truth reference.
> 
> Therefore, in CYPhER V2.5, this capability and its derivative `M12_POTHOLE_GEOMETRY` are strictly categorized as:
> **`CALIBRATION_REQUIRED`**.

---

## 4. Evaluation of Modern Candidate Architectures

- **Depth Anything V2 (Small):** Provides sharper boundary gradients on thin poles and pedestrians, but increases latency to 22.4 ms and VRAM to 340 MB.
- **Metric3D v2:** Incorporates camera focal length normalization during training, producing metric estimates when $f_x$ is known; requires PyTorch C++ extensions.
- **Operational Verdict:** Retain `yolo26n-depth.pt` for real-time edge processing; evaluate Metric3D v2 when camera calibration profiles are ingested.
