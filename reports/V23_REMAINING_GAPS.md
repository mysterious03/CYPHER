# CYPhER V2.3 — SYSTEM GAP ANALYSIS & NEXT ENGINEERING PRIORITIES

**Audit Date:** November 5, 2026  
**Auditor:** Senior Computer Vision Evaluation Engineer & ML Researcher  
**System Version:** CYPhER V2.3 (`feature/cypher-v2.3`)  

---

## 1. Overview of Evaluated Perception Capabilities

The CYPhER V2.3 evaluation marks the first time that all 14 perception capabilities have been subjected to empirical holdout validation on the Indian RDD2022 dataset and processed multi-task patches.

| Capability Group | Operational Models | Gated / Quarantined Models |
| :--- | :---: | :---: |
| **Validated Computer Vision Models** | `M1_ROAD_DAMAGE`, `M11_TRAFFIC_SIGNALS` | None |
| **Pretrained / Trained Vision Models** | `M4_VEHICLES`, `M5_ROAD_SCENE`, `M6_DEPTH`, `M9_WEATHER`, `M10_LANES`, `M12_OBSTACLES` | None |
| **Weak / Quarantined Vision Models** | None | `M2_SURFACE` (Weak Macro-F1), `M3_WATER` (Quarantined) |
| **Untrained Neural Models** | None | `M8_ACCIDENT` (Kinematic Rule Fallback) |
| **Sensor & Calibration Gated Pipelines** | None | `M13_FRICTION`, `M14_ROUGHNESS`, `M15_POTHOLE_GEOMETRY` |

---

## 2. Weakest vs. Strongest Models

### The Four Strongest Performers:
1. **D20 Alligator Crack Detection (`M1_ROAD_DAMAGE` in Adaptive Mode):**
   - **87.03% Precision** and **52.44% Recall** across 307 ground-truth holdout instances. Highly reliable for pavement fatigue alerts.
2. **Vehicle Instance Segmentation & ByteTrack (`M4_VEHICLE_SEG`):**
   - 389 verified vehicle detections across 100 holdout frames in **12.4 ms**. Outstanding spatial mask boundary adherence.
3. **Road Scene Semantic Segmentation (`M5_SCENE_SEM`):**
   - Produces robust **35.09% mean drivable road coverage**; effectively eliminates out-of-road false alarms for defect detection.
4. **Traffic Signal Illumination Classifier (`M11_TRAFFIC_SIGNALS`):**
   - Hybrid CNN + HSV optical fallback achieves **100% precision** on active traffic light patches with fail-safe redundancy.

### The Four Weakest Performers:
1. **Multi-Task Surface Classifier (`M2_SURFACE_CLASSIFIER`):**
   - Despite 97.90% headline accuracy, Macro-F1 collapses to **49.83% (friction)** and **48.55% (unevenness)**. Fails to identify 100% of wet patches (0/8) and 100% of uneven patches (0/66).
2. **Road Waterlogging Segmenter (`M3_WATER_SEG`):**
   - Quarantined. Trained on heuristic pseudo-masks that confuse smooth dark bitumen and vehicle shadows with floodwater.
3. **Small-Defect Road Damage & D10 Cracks (`M1_ROAD_DAMAGE`):**
   - **0.0% Recall** on small defects ($< 32^2 \text{ px}^2$) and **0 True Positives** on D10 transverse cracks (11/11 missed).
4. **Temporal Accident 3D-CNN (`M8_ACCIDENT_INTELLIGENCE`):**
   - Architecture implemented but completely untrained; relies entirely on simple bounding-box kinematic rules.

---

## 3. Ranked Next Engineering Priorities

### Priority 1: Retrain Surface Classifier with Class-Balanced Loss & Resampling
- **Problem:** Model acts as a majority-class predictor (Dry/Asphalt/Smooth).
- **Solution:** Apply Class-Balanced Focal Loss ($\gamma = 2.0$, $\alpha_c \propto 1 / \sqrt{N_c}$) and oversample minority wet and gravel patches. Target Macro-F1 $\ge 75\%$.

### Priority 2: Ingest Authentic Water Dataset & Retrain Water Segmenter
- **Problem:** Pseudo-mask weights are quarantined and cannot be used in production.
- **Solution:** Ingest 1,500 authentic flood/puddle images from ACDC and RTK-Water. Retrain LRASPP MobileNetV3 with joint Cross-Entropy + Dice loss to graduate from `DATASET_REQUIRED`.

### Priority 3: Solve Small Damage Extinction via SAHI or High-Resolution Tiling
- **Problem:** 0.0% recall on defects $< 32^2 \text{ px}^2$ at 640x640 resolution.
- **Solution:** Integrate Slicing Aided Hyper Inference (SAHI) with $2 \times 2$ patch tiling (320x320 overlap) during focused highway ROI scans, or evaluate lightweight 960x960 inference.

### Priority 4: Re-weight & Augment D10 Transverse Cracks
- **Problem:** D10 achieves 0 TP / 11 FN due to 1.4% representation and 9.61 aspect ratio.
- **Solution:** Apply Copy-Paste synthetic augmentation for horizontal cracks and scale D10 classification loss by 7.14x (effective-number weighting) in YOLO training.

### Priority 5: Collect Crash Sequence Benchmark for 3D-CNN Training
- **Problem:** `M8_ACCIDENT` relies on crude bounding box aspect-ratio flips.
- **Solution:** Ingest DADA-2000 or DoTA dashcam crash datasets to train the 16-frame spatio-temporal 3D-CNN for genuine collision recognition.
