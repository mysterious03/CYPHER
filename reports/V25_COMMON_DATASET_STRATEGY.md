# CYPhER V2.5 — COMMON INDIAN ROAD DATASET STRATEGY
**Multi-Task Benchmark Architecture, Environment Taxonomy, and Zero-Leakage Splitting**

---

## 1. Executive Summary & Problem Statement

Prior autonomous perception benchmarks evaluate individual computer vision models on disconnected, non-overlapping datasets:
- Road damage on RDD2022 (India / Japan subsets).
- Vehicles and pedestrians on COCO or Cityscapes (Western, highly structured highway layouts).
- Surface wetness on RSCD.
- Lane markings on CULane or TuSimple.

Deploying on real Indian roads creates massive cross-modal failure because these datasets do not share environmental context. A pothole in India does not exist in isolation: it exists alongside wandering cows, unpainted road dividers, auto-rickshaws cutting across lanes, and monsoon puddles concealing surface depth.

**CYPhER V2.5 introduces the Common Indian Road Benchmark (CIRB) Strategy**: A unified multimodal evaluation framework where a single cohesive collection of real Indian driving video sequences supports independent, task-specific ground-truth annotation layers.

---

## 2. Target Environmental Taxonomy (8 Operational Operational Domains)

The benchmark spans eight distinct operational domains representing the true variability of the Indian road network:

| Environment ID | Operational Domain | Dominant Visual Characteristics | Primary Perception Challenges |
|---|---|---|---|
| **ENV-1** | Chennai Urban Arterial | High traffic density, mixed vehicle sizes, commercial signage | Severe occlusions, near-field pedestrians, lane indiscipline |
| **ENV-2** | NH-44 National Highway | Multi-lane divided highway, high vehicle speeds ($> 80 \text{ km/h}$) | Distant small defect detection, high-speed approach tracking |
| **ENV-3** | State Highway Undivided | Two-lane bidirectional corridors, tree-canopy shade striations | Harsh shadow-to-sunlight transitions mimicking cracks/wetness |
| **ENV-4** | Rural Village Paved | Narrow single-lane asphalt, eroded edges, unpaved shoulders | Missing painted lane markings, pedestrian and cattle crossings |
| **ENV-5** | Rural Unpaved / Dirt | Mud, gravel, loose stones, non-asphalt drivable corridors | Absence of pavement edges, high dust optical scattering |
| **ENV-6** | Crowded Junction / Market | Complex intersections, auto-rickshaws, vendors, stalls | Multi-directional movement, ambiguous right-of-way |
| **ENV-7** | Monsoon Waterlogged | Turbid brown standing water, flooded gutters, wet asphalt | Water-covered potholes, specular headlight and solar glare |
| **ENV-8** | Nighttime / Low-Light | Poor street lighting, high-beam glare, unlit two-wheelers | Extreme dynamic range, loss of color saturation and texture |

---

## 3. Multi-Layer Task-Specific Annotation Architecture

A single raw RGB dashcam frame $(1920 \times 1080)$ supports six independent ground-truth annotation layers stored under `data/cypher_indian_road/annotations/`:

```
data/cypher_indian_road/
├── raw_frames/
├── annotations/
│   ├── damage_bboxes/         # Layer 1: YOLO format (D00, D10, D20, D40)
│   ├── vehicle_participants/  # Layer 2: 11 Indian vehicle & VRU classes
│   ├── surface_conditions/    # Layer 3: JSON multi-task (Friction, Material, Unevenness)
│   ├── lane_polylines/        # Layer 4: Row-anchor coordinates and line styles
│   ├── traffic_signals/       # Layer 5: Housing boxes and illuminated state labels
│   └── obstacles/             # Layer 6: Text-prompted hazard bounding boxes
└── splits/
    ├── train.txt
    ├── val.txt
    ├── test_heldout.txt
    └── test_adverse_conditions.txt
```

### Layer Specification
1. **Layer 1 (Pavement Distress):** Bounding boxes for longitudinal cracks (`D00`), transverse cracks (`D10`), alligator fatigue cracks (`D20`), and potholes (`D40`).
2. **Layer 2 (Road Participants):** 11 classes: `car`, `motorcycle`, `auto_rickshaw`, `e_rickshaw`, `bus`, `truck`, `tractor`, `bicycle`, `person`, `rider`, `animal`.
3. **Layer 3 (Surface Macro-Condition):** Discrete labels for friction state (`Dry`, `Wet`, `Water-covered`), surface material (`Asphalt`, `Concrete`, `Gravel`, `Dirt`), and unevenness (`Smooth`, `Uneven`).
4. **Layer 4 (Lanes & Boundaries):** Discrete polylines regressed at 72 vertical anchors: `solid_white`, `dashed_white`, `solid_yellow`, `double_yellow`, `unmarked_edge`.
5. **Layer 5 (Traffic Signals):** YOLO bounding box for signal housing + active illuminated color (`Red`, `Yellow`, `Green`, `Off`).
6. **Layer 6 (Obstacles):** Real-world road blockages: `fallen tree`, `rock`, `boulder`, `road debris`, `broken concrete`, `spilled cargo`, `tyre on road`, `construction material`.

---

## 4. Strict Zero-Leakage Dataset Splitting Protocol

A critical flaw discovered in legacy research was random image partitioning from continuous video streams, resulting in sequential near-duplicate frames appearing across train and validation splits.

### CIRB Disjoint Splitting Rules:
1. **Sequence-Disjoint:** All frames from a continuous drive sequence ($Seq\_k$) must reside exclusively in exactly one partition.
2. **Location-Disjoint:** Road sectors separated by less than 5 kilometers must not be split across train and test sets.
3. **Dedicated Adverse Test Partition:** `test_adverse_conditions.txt` contains 100% unseen severe weather, nighttime, and intense monsoon driving scenes reserved strictly for out-of-distribution generalization testing.
