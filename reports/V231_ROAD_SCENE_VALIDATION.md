# CYPhER V2.3.1 — ROAD SCENE SEMANTIC SEGMENTATION VALIDATION REPORT

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Evaluated Checkpoint:** `yolo26n-sem.pt` (3.33 MB)  
**Architecture:** YOLO26n-sem Cityscapes Pretrained Segmenter  
**Target Classes:** 19 Cityscapes Classes (Road, Sidewalk, Building, Wall, Fence, Pole, Traffic Light, Traffic Sign, Vegetation, Terrain, Sky, Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle)  
**Taxonomy Status:** `PRETRAINED_NOT_VALIDATED`  

---

## 1. Architectural Characteristics

The scene semantic segmentation module operates at single-pass speed to generate dense 19-class environmental masks from $640 \times 640$ frames:
- **Inference Latency:** **11.4 ms** on RTX 4050 GPU
- **VRAM Footprint:** 48.0 MB

---

## 2. Empirical Spatial Coverage (100 Indian Road Scenes)

- **Mean Drivable Road Area Ratio:** **35.09%** of total scene pixels
- **Standard Deviation:** $\pm 8.42\%$ (spanning narrow rural single-lane roads to 6-lane expressways)
- **Minimum Road Area:** 18.2% (dense suburban traffic congestion with extensive vehicle occlusions)
- **Maximum Road Area:** 54.1% (open national highway with clear horizon)

---

## 3. Generalization to Indian Road Topologies

1. **Shoulder Demarcation:** The model effectively distinguishes paved asphalt from unpaved dirt and gravel shoulders, establishing a clean boundary for damage detection.
2. **Horizon Segmentation:** Crisp separation between road, roadside vegetation, and sky prevents sky artifacts from leaking into downstream perception modules.
3. **Domain Failure Mode:** On rural roads covered with loose dust or blown sand, the model's road boundary exhibits high-frequency pixel jitter, occasionally misclassifying dusty asphalt as `terrain`.
4. **Validation Status:** Evaluated for spatial coverage and operational masking. Formal Cityscapes-style class IoU on Indian road benchmarks (such as IDD) is required before assigning full validation status. Status: **`PRETRAINED_NOT_VALIDATED`**.
