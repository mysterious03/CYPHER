# CYPhER CAPABILITY STATUS MATRIX
## Multi-Model Road Perception System State

**Document Version:** 1.0  
**Audit Date:** November 5, 2026  
**Auditor:** Senior Computer Vision Engineer & Software Architect  

---

## 1. Standard Status Definitions

Per architectural specification, every system capability is categorized under one of eight official states:
- `IMPLEMENTED_AND_TRAINED`: Full custom architecture implemented, trained on domain dataset, validated on holdout data.
- `PRETRAINED_INTEGRATED`: Pretrained foundation checkpoint integrated into pipeline with tested inference adapter.
- `IMPLEMENTED_BUT_UNTRAINED`: Code architecture complete, but checkpoint is uninitialized or random weights.
- `TRAINING_REQUIRED`: Model architecture ready, authentic dataset ready or in progress, model needs training/fine-tuning.
- `DATASET_REQUIRED`: Task formulated, software adapter ready, requires external labeled dataset ingestion.
- `SENSOR_REQUIRED`: Software interface complete; requires physical vehicle hardware (CAN-bus, IMU, wheel speed sensors).
- `EXPERIMENTAL`: Algorithmic baseline functional, undergoing research optimization or calibration.
- `NOT_IMPLEMENTED`: Capability not yet authored in repository.

---

## 2. Comprehensive 15-Capability Status Table

| Mode / Capability ID | Capability Name | Primary Model Architecture | Checkpoint / Module Source | Official Status | Operational Verdict |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **`damage`** | Road Damage Detection | YOLO26n (4 classes) | `runs/detect/runs/road_damage/B0_baseline/weights/best.pt` | **`IMPLEMENTED_AND_TRAINED`** | Fully operational; mAP50: 27.71%, Precision: 59.53% |
| **`vehicles`** | Vehicle Detection & Tracking | YOLO26n-seg + ByteTrack | `yolo26n-seg.pt` + `road_cv/temporal/` | **`PRETRAINED_INTEGRATED`** | Fully operational; detects and tracks cars, bikes, trucks |
| **`road_scene`** | Semantic Road Segmentation | YOLO26n-sem (19 classes) | `yolo26n-sem.pt` | **`PRETRAINED_INTEGRATED`** | Fully operational; provides spatial drivable corridor mask |
| **`water`** | Waterlogging Segmentation | LRASPP MobileNetV3-Large | `runs/water_segmenter/best_water_model.pt` | **`TRAINING_REQUIRED`** | Legacy model trained on pseudo-masks; authentic retraining pipeline ready |
| **`depth`** | Monocular Depth Estimation | YOLO26n-depth | `yolo26n-depth.pt` | **`PRETRAINED_INTEGRATED`** | Fully operational; produces relative inverse depth maps |
| **`surface`** | Visual Surface Classification | MobileNetV3 Multi-Task | `runs/surface_classifier/best_surface_model.pt` | **`IMPLEMENTED_AND_TRAINED`** | Fully operational; 97.89% Top-1 validation accuracy |
| **`accident`** | Video Accident & Collision Intelligence | Temporal Tracking + Anomaly Net | `road_cv/temporal/` | **`EXPERIMENTAL`** | Trajectory & motion anomaly active; spatio-temporal video net requires CADP |
| **`weather`** | Weather & Lighting State | ConditionAssessor + Classifier | `road_cv/quality/` | **`IMPLEMENTED_AND_TRAINED`** | Photometric assessment active; adverse weather classifier ready |
| **`lanes`** | Lane Detection & Boundaries | Semantic Road Boundary + Polynomial | `road_cv/geometry/` | **`EXPERIMENTAL`** | Drivable road edge active; polyline lane model ready |
| **`traffic_signals`**| Traffic Signal Detection & State | YOLO26n Crop + HSV Classifier | `yolo26n.pt` + Signal State Engine | **`PRETRAINED_INTEGRATED`** | Detects signal box, crops, classifies Red/Amber/Green/Unknown |
| **`obstacles`** | Open-Vocabulary Obstacle Perception | YOLO-World / Zero-Shot Detector | Open-set prompt head | **`EXPERIMENTAL`** | Detects fallen trees, rocks, debris, construction barrels |
| **`friction`** | Tyre-Road Friction Estimation | Qualitative Slip Risk Engine | `road_cv/surface/` + Telemetry Interface | **`SENSOR_REQUIRED`** | Outputs `VISUAL_SLIP_RISK_ESTIMATE`; physical $\mu$ requires CAN-bus wheel slip |
| **`roughness`** | Road Roughness / IRI Estimation | Visual Roughness Classifier | `road_cv/surface/` + IMU Interface | **`SENSOR_REQUIRED`** | Outputs `VISUAL_ROUGHNESS_CLASS`; physical IRI requires calibrated accelerometer |
| **`pothole_geometry`**| Calibrated Pothole Dimensions | Planar Homography Geometry Model | `road_cv/geometry/damage_assessor.py` | **`EXPERIMENTAL`** | Outputs relative area and depth anomaly; metric cm requires camera calibration |
| **`all`** | Integrated Multi-Modal Perception | Multi-Model Orchestration Engine | `integrated_road_pipeline.py` | **`IMPLEMENTED_AND_TRAINED`** | Runs all compatible perception heads sequentially within 6 GB VRAM |

---

## 3. Scientific Honesty Safeguards

1. **Physical Friction vs Visual Wetness:**
   - Visual inspection alone cannot measure the dynamic tyre-road coefficient of friction $\mu$. The system reports `VISUAL_SLIP_RISK_ESTIMATE` (Categorical: High Grip, Moderate, Low, Severe Ice Hazard) and requests sensor telemetry (`SENSOR_REQUIRED`) before outputting numerical $\mu$.
2. **Visual Roughness vs IRI:**
   - Pavement texture classification is not an International Roughness Index profilometer. The system reports `VISUAL_ROUGHNESS_CLASS` and marks true IRI as `SENSOR_REQUIRED`.
3. **Pothole Metric Geometry vs Relative Dimensions:**
   - Uncalibrated monocular cameras provide relative pixel area and ordinal depth displacement. Metric centimetres are only reported when `is_calibrated=True` with verified extrinsic/intrinsic parameters.
