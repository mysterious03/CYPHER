# CYPhER V2.3.1 — MODEL AND DATASET INVENTORY

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**System Branch:** `feature/cypher-v2.3.1-accuracy`  

---

## 1. Complete Model Inventory

| Model ID | Perception Capability | Architecture | Checkpoint File | File Size | Input Shape | Output Shape / Classes | Pretrained vs Fine-Tuned | Task Adapter |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **M1** | Road Damage Detection | YOLO26n (Custom Anchorless) | `runs/detect/road_damage_v22/weights/best.pt` | 24.89 MB | `[3, 640, 640]` | 4 Classes: D00, D10, D20, D40 | Fine-Tuned (RDD2022 India, 38 ep) | `road_cv/adapters/yolo_adapter.py` |
| **M2** | Surface Condition Classifier | MobileNetV3-Large Multi-Task | `runs/surface_classifier/best_surface_model.pt` | 13.90 MB | `[3, 224, 224]` | 3 Heads: Friction (6), Material (6), Unevenness (4) | Fine-Tuned (RSCD subset, 5 ep) | `road_cv/adapters/surface_adapter.py` |
| **M3** | Waterlogging Segmentation | LRASPP MobileNetV3-Large | `runs/water_segmenter/best_water_model.pt` | 12.48 MB | `[3, 512, 512]` | 4 Classes: Dry, Flood, Puddle, Reflection | Fine-Tuned (Quarantined Pseudo-Masks) | `road_cv/adapters/water_adapter.py` |
| **M4** | Vehicle Instance Segmentation | YOLO26n-seg | `yolo26n-seg.pt` | 6.41 MB | `[3, 640, 640]` | 80 COCO Classes + Polygon Masks | Pretrained (COCO) | `road_cv/adapters/vehicle_adapter.py` |
| **M5** | Road Scene Segmentation | YOLO26n-sem | `yolo26n-sem.pt` | 3.33 MB | `[3, 640, 640]` | 19 Cityscapes Classes | Pretrained (Cityscapes) | `road_cv/adapters/scene_adapter.py` |
| **M6** | Monocular Depth Estimation | YOLO26n-depth | `yolo26n-depth.pt` | 12.40 MB | `[3, 640, 640]` | `[1, 640, 640]` Relative Inverse Depth | Pretrained | `road_cv/adapters/depth_adapter.py` |
| **M8** | Accident & Anomaly Engine | 16-Frame 3D-CNN + Kinematics | Code (Architecture Instantiated) | N/A | `[3, 16, 112, 112]` | 4 Classes: Normal, Near Miss, Collision, Rollover | Untrained Architecture (Rule Fallback) | `road_cv/adapters/accident_adapter.py` |
| **M9** | Weather & Lighting Classifier | MobileNetV3-Small Dual-Head | `runs/weather_classifier_v22/best_weather_model.pt` | 4.52 MB | `[3, 224, 224]` | 2 Heads: Weather (5), Lighting (3) | Fine-Tuned | `road_cv/adapters/weather_adapter.py` |
| **M10** | Polyline Lane Detector | MobileNetV3 Row-wise Anchor | `runs/lane_detector_v22/best_lane_model.pt` | 3.86 MB | `[3, 288, 800]` | 5 Marking Types, Row-wise Offsets | Fine-Tuned | `road_cv/adapters/lane_adapter.py` |
| **M11** | Traffic Signal Intelligence | YOLO26n Box + Patch CNN + HSV | `runs/traffic_signals_v22/best_signal_model.pt` | Code + HSV | `[3, 32, 64]` | 4 States: Red, Amber, Green, Unknown | Hybrid (Optical Fallback Active) | `road_cv/adapters/signal_adapter.py` |
| **M12** | Open-Vocabulary Obstacles | YOLO-World (yolov8s-world) | `yolov8s-world.pt` | 25.91 MB | `[3, 640, 640]` | 10 Dynamic Hazard Text Prompts | Pretrained (Zero-Shot) | `road_cv/adapters/obstacle_adapter.py` |
| **M13** | Tyre-Road Friction ($\mu$) | Magic Formula / Dugoff | Code Rules | N/A | CAN Telemetry | Friction Coefficient $\mu$ / Slip Risk | Analytical / Fusion Gateway | `road_cv/adapters/friction_adapter.py` |
| **M14** | Road Roughness (IRI) | Quarter-Car ISO 8608 | Code Rules | N/A | IMU / Accelerometer | IRI (m/km) / Visual Roughness | Analytical / Fusion Gateway | `road_cv/adapters/iri_adapter.py` |
| **M15** | Pothole Geometry | Inverse Perspective Mapping | Code Geometry | N/A | Camera Calibration | Metric Length, Width, Depth (cm) | Geometric / Calibration Gateway | `road_cv/adapters/geometry_adapter.py` |

---

## 2. Dataset Inventory

### A. RDD2022 India (Road Damage Detection)
- **Source:** Crowdsensing-based Road Damage Detection Challenge 2022 (IEEE BigData).
- **License:** Academic & Research Use Only.
- **Images:** 6,540 total processed images (Train: 5,368 images, Val: 1,172 images).
- **Format:** YOLO normalized bounding boxes (`class x_center y_center width height`) and Pascal VOC XML source.
- **Class Distribution:**
  - Train: D00 (1,555), D10 (68), D20 (2,021), D40 (1,372). Total = 5,016 instances.
  - Val: D00 (223), D10 (11), D20 (307), D40 (202). Total = 743 instances.
- **Integrity:** Strict separation between `train/` and `val/` directories. Zero duplicate leakages detected across splits.

### B. RSCD Surface Condition Patches
- **Source:** Extracted lower-road patches from Indian RDD highway imagery.
- **Format:** JSON metadata catalogs (`train_patches.json`, `val_patches.json`).
- **Samples:** Train: 5,368 patches; Val: 1,172 patches.
- **Severe Imbalance Discovered:**
  - Train Friction: 5,341 Dry, 27 Wet.
  - Train Material: 5,368 Asphalt.
  - Train Unevenness: 5,132 Smooth, 236 Slightly Uneven.
  - Val Friction: 1,164 Dry, 8 Wet.
  - Val Material: 1,172 Asphalt.
  - Val Unevenness: 1,106 Smooth, 66 Slightly Uneven.

### C. Authentic Waterlogging Segmentation Dataset
- **Directory:** `data/processed/water_authentic/`
- **Target Specification:** R²S100K / ADE20K wet road polygon extracts.
- **Classes:** 0: Dry pavement, 1: Standing water / flood, 2: Water puddle, 3: Specular reflection.
- **Loader:** `road_cv/water/authentic_water_dataset.py` enforcing strict pair-existence checks.
