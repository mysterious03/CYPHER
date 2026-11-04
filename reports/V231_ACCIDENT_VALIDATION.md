# CYPhER V2.3.1 — TEMPORAL ACCIDENT INTELLIGENCE AUDIT & VALIDATION

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Module Architecture:** 16-Frame Spatio-Temporal 3D-CNN + ByteTrack Kinematics  
**Target Event Classes:** `normal_traffic`, `hard_braking_near_miss`, `vehicle_collision`, `vehicle_rollover_or_jackknife`  
**Current Operational Status:** Kinematic Rule Engine Active; 3D-CNN Untrained  
**Taxonomy Status:** `DATASET_REQUIRED` / `HEURISTIC_BASELINE`  

---

## 1. Executive Summary & Verification Findings

The CYPhER accident intelligence system (`road_cv/accident/temporal_accident_detector.py`) is designed as a hybrid two-tier architecture:
1. **Geometric Kinematic Tracker:** Tracks multi-vehicle bounding box centroids and aspect ratios over time to detect hard braking, trajectory convergence, and vehicle rollovers.
2. **Spatio-Temporal 3D-CNN (`SpatioTemporalAccident3DCNN`):** A neural video action recognition network taking `[B, 3, 16, 112, 112]` temporal video cubes to learn spatio-temporal representations of crash physics.

An inspection of the repository confirms that while the PyTorch network is fully instantiated, **no trained weights exist**. The active pipeline operates purely on trajectory-based kinematic rules.

---

## 2. Evaluation of the Active Kinematic Heuristic

The kinematic rule engine computes:
- **Deceleration Anomaly:** If a tracked vehicle exhibits $\frac{\Delta v}{\Delta t} < -6.5\text{ m/s}^2$, it triggers a `hard_braking_near_miss` alert.
- **Bounding Box Collision Conflict:** If two tracked vehicle bounding boxes have $\text{IoU} > 0.35$ and their velocity vectors converge with relative speed $> 15\text{ km/h}$, a `vehicle_collision` alert is emitted.
- **Orientation Inversion / Rollover:** If a vehicle's bounding box aspect ratio flips from horizontal ($w/h > 1.2$) to vertical ($w/h < 0.6$) within 5 consecutive frames, a `vehicle_rollover_or_jackknife` flag is raised.

### Empirical Strengths & Failure Modes:
- **Strengths:** Zero GPU computation overhead; operates in real time ($< 1.0\text{ ms}$) atop existing ByteTrack tracks.
- **Weaknesses:** Highly susceptible to tracking false alarms during occlusions (e.g., when a vehicle passes behind a tree or lamppost, the sudden centroid displacement mimics a high-G collision). Cannot recognize pedestrian impacts or sideswipes without visual texture features.

---

## 3. Dataset Roadblock & Roadmap to Neural 3D-CNN Training

To graduate from `DATASET_REQUIRED` to `VALIDATED_ACCEPTABLE`, the 3D-CNN model requires ingestion of public dashcam collision video sequences:
- **DADA-2000:** 2,000 video sequences with 54 driver anomaly and accident categories.
- **DoTA (Detection of Traffic Anomaly):** 4,677 videos with spatio-temporal accident bounding box labels.
- **Car Crash Dataset (CCD):** Real dashcam crash footage covering adverse lighting and weather.

**Auditor Status:** Labeled `DATASET_REQUIRED` / `HEURISTIC_BASELINE`. The untrained 3D-CNN must never be represented as a trained model.
