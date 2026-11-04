# CYPhER V2.3.1 — POLYLINE LANE DETECTOR VALIDATION REPORT

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Evaluated Checkpoint:** `runs/lane_detector_v22/best_lane_model.pt`  
**Architecture:** MobileNetV3 Row-Wise Anchor Polyline Detector (3.86 MB)  
**Target Markings:** `solid_white`, `dashed_white`, `solid_yellow`, `double_yellow`, `unmarked_edge`  
**Taxonomy Status:** `TRAINED_NOT_VALIDATED`  

---

## 1. Architectural Formulation & Advantage

The CYPhER lane perception engine rejects computationally prohibitive full-frame pixel semantic segmentation (which requires post-hoc curve fitting and morphological thinning) in favor of a **row-wise anchor regression formulation**:
- An input frame (`288x800`) passes through a lightweight MobileNetV3 encoder.
- For 72 predefined vertical row anchors ($y_i$), the network predicts horizontal coordinate offsets ($x_{i, l}$) and classification logits for lane boundaries.
- **Inference Latency:** **4.3 ms** per frame
- **VRAM Consumption:** 42.1 MB

---

## 2. Holdout Road Marking Distribution (100 Indian Scenes)

- **Dashed White:** 78 scenes (78.0%) — Multi-lane national highway sectors
- **Double Yellow:** 12 scenes (12.0%) — Two-lane undivided intercity state corridors
- **Solid White:** 5 scenes (5.0%) — Outer carriageway shoulder markings
- **Solid Yellow:** 3 scenes (3.0%) — Median road dividers
- **Unmarked Edge:** 2 scenes (2.0%) — Unpaved rural corridors
- **False Lane Departure Triggers:** 0 false alarms on forward-facing centered highway paths

---

## 3. Indian Roadway Domain Observations

1. **Curvature Handling:** Continuous polyline regression tracks winding highway curves smoothly without polygon self-intersection.
2. **Unmarked Asphalt-to-Dirt Boundaries:** On rural Indian stretches where painted markings have eroded, the detector switches to tracking high-contrast asphalt edge lines (`unmarked_edge`), though edge jitter increases when dust obscures the border.
3. **Benchmarking Requirement:** While operational inference is verified, exact mathematical polyline metrics (CULane F1-score with 30-pixel IoU threshold) require evaluation against an authentic annotated lane benchmark (e.g., CULane or TuSimple). Hence, the capability is classified as **`TRAINED_NOT_VALIDATED`**.
