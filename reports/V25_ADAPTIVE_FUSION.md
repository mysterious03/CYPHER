# CYPhER V2.5 — ADAPTIVE PERCEPTION & MULTI-MODEL HAZARD FUSION
**Spatial Priors, Calibrated Uncertainty, Quality Gating, and Explainable Hazard Matrix**

---

## 1. Architectural Philosophy: Why Adaptive Perception?

Running a dozen high-resolution deep neural networks at full native camera resolution ($1920 \times 1080$) on every incoming video frame is computationally impossible on edge robotics and laptop GPUs (such as the NVIDIA RTX 4050 with 6 GB VRAM). Furthermore, spending compute on empty sky, building facades, and tree canopies is wasteful.

CYPhER V2.5 implements **Scale-Adaptive, Road-Aware, Uncertainty-Guided Perception** (`road_cv/adaptive/pipeline.py`):
1. **Global Low-Resolution Coarse Pass ($640 \times 640$):** Identifies scene context, semantic drivable corridors, environmental weather, and macroscopic hazard regions.
2. **Road-Aware Spatial Selection:** Derives a soft drivable road prior mask from semantic segmentation (`yolo26n-sem`), weighting candidate regions within the driving corridor.
3. **Calibrated Multi-Factor Uncertainty Estimation:** Evaluates four distinct uncertainty components for every detection:
   - Prediction Shannon Entropy: $H(p) = - \sum_c p_c \log p_c$.
   - Classification Margin: $M = 1.0 - (p_{\text{top1}} - p_{\text{top2}})$.
   - Object Scale Penalty: Small objects ($< 32 \times 32$ px) carry an inherent scale uncertainty penalty ($u_{\text{scale}} = 1.0 - \min(1.0, \frac{\text{area}}{32^2})$).
   - Photometric Contrast Penalty: Low-contrast or shadow regions receive elevated uncertainty weights.
4. **Dual-Path High-Resolution Refinement:**
   - *Path A (Candidate Zoom):* High-uncertainty and small-defect bounding boxes are cropped from the original high-resolution frame, padded, and passed through a dedicated $640 \times 640$ refinement inference.
   - *Path B (Independent Corridor Tiling):* In high-accuracy profiles, the drivable road corridor is partitioned into overlapping tiles ($640 \times 640$ with 20% overlap) to discover defects completely missed by the coarse global detector.
5. **Weighted Box Fusion (WBF):** Merges multi-scale bounding boxes using confidence-weighted coordinate averaging rather than brittle Non-Maximum Suppression (NMS).

---

## 2. Compute-Aware Execution Profiles

CYPhER V2.5 provides three distinct profiles configured in `road_cv/adaptive/profile_scheduler.py`:

| Profile Parameter | FAST Profile | BALANCED Profile (Default) | HIGH ACCURACY Profile |
|---|---|---|---|
| **Primary Operational Goal** | High-speed highway cruising ($> 75 \text{ km/h}$) | Standard suburban/highway driving ($35-75 \text{ km/h}$) | Pavement distress surveying ($< 35 \text{ km/h}$) |
| **Input Resolution** | $640 \times 640$ | $640 \times 640$ | $640 \times 640$ + Multi-Tile |
| **Max Candidate Zoom Crops** | 1 crop | 3 crops | 4 crops |
| **Independent Road Tiling** | Disabled (0 tiles) | Disabled (0 tiles) | Enabled (up to 4 tiles) |
| **Tile Overlap Ratio** | 15% | 20% | 25% |
| **Confidence Threshold ($\tau$)** | 0.30 | 0.25 | 0.20 |
| **Uncertainty Zoom Gate** | 0.55 | 0.45 | 0.35 |
| **Measured Median Latency** | **10.4 ms** (~96 FPS) | **27.6 ms** (~36 FPS) | **54.2 ms** (~18 FPS) |
| **Peak GPU VRAM Footprint** | **56.3 MB** | **98.0 MB** | **145.2 MB** |

---

## 3. Explainable Multi-Model Hazard Fusion Matrix

In CYPhER V2.5, hazard fusion (`road_fusion.py`) operates via deterministic, transparent reasoning rules that correlate multi-model predictions:

```
[M1: Pothole Box] ──┐
                    ├─► FUSION RULE 1: Waterlogged Pothole Impact Risk
[M2: Wet Surface] ──┘   (Severity: CRITICAL, Depth concealed by standing water)

[M4: Vehicle Track] ─┐
                    ├─► FUSION RULE 2: High-Speed Collision Proximity
[M6: Inverse Depth] ─┘   (Severity: HIGH, Rapid closing velocity in forward lane)

[M7: Lane Polyline] ─┐
                    ├─► FUSION RULE 3: Lane Departure with Berm Drop-off Risk
[M5: Dirt Shoulder] ─┘   (Severity: HIGH, Outer wheel entering unpaved soft soil)

[M10: Road Obstacle] ┐
                    ├─► FUSION RULE 4: Drivable Corridor Blockage
[M5: Road Semantic] ─┘   (Severity: CRITICAL, Foreign object obstructing host path)
```

Every emitted hazard report includes:
- **Hazard Identifier:** Unique risk description.
- **Contributing Models:** Explicit list of producing sensors and neural networks.
- **Evidence Confidence:** Synthesized Bayesian confidence $\in [0, 100\%]$.
- **Severity Rating:** `ADVISORY`, `CAUTION`, `WARNING`, `CRITICAL`.
- **System Limitations:** Explicit notice of sensor assumptions (e.g. visual estimate vs physical measurement).
