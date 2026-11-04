# CYPhER V2.3 — ACCIDENT & ANOMALY INTELLIGENCE EVALUATION REPORT

**Capability ID:** `M8_ACCIDENT_INTELLIGENCE`  
**Model Architecture:** 16-Frame Spatio-Temporal 3D-CNN + Trajectory Tracking Engine  
**Target Classes:** `normal_traffic`, `hard_braking_near_miss`, `vehicle_collision`, `vehicle_rollover_or_jackknife`  
**Current Operational Status:** Trajectory-based Heuristic Active; 3D-CNN Untrained  
**Taxonomy Status:** `DATASET_REQUIRED`  

---

## 1. Executive Summary & Verification Findings

The CYPhER accident intelligence system is architected as a two-tier pipeline:
1. **Geometric Trajectory Engine:** Tracks vehicle bounding box velocities, sudden deceleration vectors, and bounding box aspect-ratio shifts (e.g., rollover inversion) across ByteTrack histories.
2. **Spatio-Temporal 3D-CNN:** A 16-frame 3D convolutional network designed to process temporal video cubes for end-to-end collision recognition.

An audit of the repository confirms that while the 3D-CNN network architecture is fully defined and instantiated in PyTorch, **no trained weights exist**. The active pipeline operates purely on trajectory-based kinematic rules.

---

## 2. Kinematic Heuristic Verification

The active heuristic engine computes:
- **Deceleration Anomaly:** $\Delta v / \Delta t < -6.5\text{ m/s}^2$ triggers a hard-braking alert.
- **Overlap Conflict:** Two tracked vehicle bounding boxes with IoU $> 0.35$ and convergent velocity vectors trigger a collision warning.
- **Orientation Inversion:** Bounding box aspect ratio changing from $w/h \approx 1.8$ to $w/h < 0.6$ within 5 frames triggers a rollover flag.

While functional for high-speed crashes in test video snippets, kinematic rules lack the visual nuance to detect pedestrian impacts, sideswipes, or secondary collisions.

---

## 3. Dataset Requirements for Neural 3D-CNN Training

To train the 3D-CNN model and graduate beyond `DATASET_REQUIRED`, the following datasets must be ingested:
- **DADA-2000:** 2,000 video sequences with 54 categories of driver anomaly and traffic accidents.
- **DoTA (Detection of Traffic Anomaly):** 4,677 videos with spatio-temporal accident bounding box labels.
- **Car Crash Dataset (CCD):** Real dashcam crash footage under adverse weather and lighting.
