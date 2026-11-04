# CYPhER V2.2 Workstream D: Adaptive Road-Aware Perception System Design

**Lead Research Architect:** CYPhER Perception Lab  
**Date:** November 5, 2026  
**Core Implementation:** [`road_cv/adaptive/`](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/road_cv/adaptive/)  

---

## 1. Related Work & Research Motivation

Uniform full-resolution image processing is computationally wasteful in autonomous driving: sky, buildings, and tree canopies consume over 45% of camera pixels without containing road-surface hazards. Conversely, downsampling high-resolution camera feeds ($2400 \times 1604$) to standard detector scales ($640 \times 640$) destroys hairline road defects like transverse cracks (D10).

Drawing upon foundational concepts from **Slicing Aided Hyper Inference (SAHI)**, **Soft-Attention Cascades**, and **Evidential Uncertainty Estimation**, CYPhER V2.2 introduces an **uncertainty-guided, road-aware, scale-adaptive perception architecture**.

---

## 2. Mathematical Formulation of Adaptive Components

### 2.1 Road-Aware Soft Spatial Prior $S(b)$
Rather than binary off-road rejection (which risks discarding valid roadside hazards), we formulate a continuous spatial prior $S(b) \in [0.3, 1.0]$:
$$S(b) = \begin{cases} 
1.00 & \text{if } c(b) \in \mathcal{M}_{\text{road}} \\
0.80 & \text{if } c(b) \in \text{Dilate}(\mathcal{M}_{\text{road}}, \delta) \\
0.30 & \text{otherwise}
\end{cases}$$
where $c(b)$ is the center of bounding box $b$, $\mathcal{M}_{\text{road}}$ is the binary semantic road mask, and $\delta = 40\text{ px}$ is the boundary margin. The effective detector score is modulated as:
$$\text{Score}_{\text{eff}}(b) = \text{Score}_{\text{det}}(b) \times S(b)$$

### 2.2 Calibrated Predictive Uncertainty $U(b)$
Predictive uncertainty combines decision boundary distance, entropy, scale degradation, and optical image quality:
$$U(b) = w_m U_{\text{margin}} + w_e U_{\text{entropy}} + w_s U_{\text{scale}} + w_p U_{\text{photometric}}$$
- **Margin Uncertainty:** $U_{\text{margin}} = 1.0 - 2.0 \cdot |p - 0.50| \in [0, 1]$ (peaks at $p=0.5$).
- **Normalized Entropy:** $U_{\text{entropy}} = - (p \log_2 p + (1-p) \log_2(1-p))$.
- **Scale Ambiguity:** $U_{\text{scale}} = \max\left(0, 1.0 - \frac{\text{Area}(b)}{32 \times 32}\right)$ (penalizes objects smaller than $32\times 32$ pixels).
- **Photometric Factor:** $U_{\text{photometric}}$ penalizes low RMS contrast ($<30$), blur variance ($<80$), and glare ratio ($>0.15$).

### 2.3 Dual-Path High-Resolution Refinement
To overcome the limitations of candidate-only zoom, V2.2 deploys two complementary paths:
1. **Path A (Candidate Zoom):** Dispatches localized $640 \times 640$ zoom windows centered on candidates where $U(b) \ge \tau_U$ or $\text{Area}(b) < 32^2$.
2. **Path B (Independent Road-Corridor Tiling):** Slices the drivable corridor into overlapping $640 \times 640$ tiles directly from the unscaled high-resolution frame. This allows defects completely missed at $640 \times 640$ coarse resolution (such as D10 transverse cracks) to be discovered independently.
3. **Weighted Box Fusion (WBF):** Cluster coordinates from global and local passes are fused by confidence weighting:
   $$\bar{x} = \frac{\sum_i c_i x_i}{\sum_i c_i}$$

---

## 3. Compute-Aware Scheduling Profiles

```
+-----------------------------------------------------------------------------------------+
| Profile Parameter           Fast Profile           Balanced Profile      High-Accuracy  |
+-----------------------------------------------------------------------------------------+
| Target Domain               Highway (>75 km/h)     Urban (35-75 km/h)    Inspection / Crawl |
| Base Image Size             640 x 640              640 x 640             640 x 640      |
| Road Prior Enabled          True                   True                  True           |
| Candidate Zoom Crops        Max 1                  Max 3                 Max 4          |
| Independent Corridor Tiles  0                      0                     Max 4          |
| Target Latency Budget       <100 ms (~10 FPS)      <200 ms (~5 FPS)      <400 ms (~2.5 FPS) |
| Peak VRAM Footprint         <250 MB                <350 MB               <500 MB        |
+-----------------------------------------------------------------------------------------+
```
