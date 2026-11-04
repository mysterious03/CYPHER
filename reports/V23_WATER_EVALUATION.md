# CYPhER V2.3 — WATERLOGGING SEGMENTATION AUDIT & STATUS REPORT

**Capability ID:** `M3_WATER_SEG`  
**Model Architecture:** LRASPP MobileNetV3-Large (18.70 MB)  
**Evaluated Checkpoint:** `runs/water_segmenter/best_water_model.pt`  
**Training Source:** Heuristic Pseudo-Mask Pipeline (Quarantined)  
**Target Classes:** `dry_road`, `standing_water_flood`, `water_puddle`, `specular_wet_reflection`  
**Taxonomy Status:** `DATASET_REQUIRED`  

---

## 1. Executive Summary & Quarantine Decision

In CYPhER V2.0/V2.1, the waterlogging segmentation model reported a synthetic validation mIoU of 71.4%. A thorough engineering audit in V2.2 and V2.3 determined that this model was trained entirely on **heuristic pseudo-masks** generated via HSV color slicing and specular reflection thresholding.

Because these pseudo-masks consistently confuse fresh dark asphalt, vehicle shadows, and polished tar patches with standing water, the checkpoint does not represent genuine neural perception of water physics. 

**Decision:** The checkpoint is officially **quarantined** and labeled **`DATASET_REQUIRED`**. It must NOT be used for autonomous vehicle safety decisions until retrained on authentic, human-annotated water masks.

---

## 2. Technical Weaknesses of Heuristic Pseudo-Mask Training

1. **Shadow Confusion:** Shadow boundaries from roadside trees or overhead bridges exhibit low saturation and dark intensity, directly matching the heuristic filter for wet puddles.
2. **Fresh Bitumen False Alarms:** Newly laid, smooth dark bitumen has low luminance and high specular reflectivity under sunlight, triggering false flood alarms on completely dry roads.
3. **Specular vs. Standing Water:** The heuristic cannot distinguish between specular road reflections (which present friction hazards) and standing flood water (>5 cm depth, presenting hydroplaning hazards).

---

## 3. Retraining Requirements for Production Readiness

To exit the `DATASET_REQUIRED` status, the following protocol is established:
1. **Authentic Datasets:** Ingest genuine road flood benchmarks:
   - ACDC (Adverse Conditions Dataset with Correspondence - Water subset)
   - Real-world puddle segmentation datasets (e.g., Water Puddle Dataset / RTK-Water)
   - Indian monsoon road flood capture
2. **Loss Function:** Train using joint Cross-Entropy + Soft-Dice Loss to capture fine puddle boundaries and large flood pools.
3. **Evaluation Protocol:** Validate against holdout human-annotated polygon masks with separate evaluation of puddles vs. deep floodwater.
