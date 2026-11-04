# CYPhER V2.4: Robust Multi-Model Road Perception & Hazard Intelligence

**CYPhER** (Comprehensive Yield & Pavement Hazard Estimation for Roads) is an end-to-end, scale-adaptive computer vision application that unifies **15 road perception and hazard intelligence capabilities** into a single software suite.

Version 2.4 introduces **Empirical Retraining & Scientific Verification**, resolving chronic minority-class failures, eliminating synthetic pseudo-mask hallucinations, implementing authentic copy-paste data augmentation, and benchmarked adaptive multi-profile perception.

Engineered specifically for local execution on edge and laptop hardware with **6 GB VRAM** (such as the NVIDIA GeForce RTX 4050 Laptop GPU), CYPhER coordinates deep learning models, explainable multi-model hazard fusion, and physical sensor telemetry interfaces.

---

## 1. What CYPhER Does

CYPhER transforms raw dashcam images, video streams, or live camera feeds into structured, actionable road intelligence:
- **Structural Pavement Damage:** Detects longitudinal cracks (`D00`), transverse cracks (`D10`), alligator fatigue cracks (`D20`), and potholes (`D40`) with scale-adaptive tiling and zoom.
- **Adaptive Road-Aware Perception:** Utilizes semantic road masks and calibrated predictive uncertainty to direct high-resolution compute exclusively to the drivable corridor.
- **Surface Condition & Grip:** Multi-task classification of visual pavement wetness, road construction material, and surface macro-roughness with **100% wet hazard recall**.
- **Open-Vocabulary Obstacle Perception:** Detects unconstrained natural language obstacles (fallen tree, boulder, road debris, spilled cargo, tire on road) using YOLO-World.
- **Dedicated Lane Perception:** Predicts left/right boundary polylines, corridor width, lane marking types (dashed, solid, double), and departure warnings.
- **Waterlogging & Flood Hazards:** Rigorous intake protocol for authentic pixel-level benchmarks (RTK-Water / ACDC), quarantining synthetic pseudo-masks.
- **Traffic & Signal Awareness:** Tracks vehicles and pedestrians with ByteTrack and classifies traffic lights via dedicated neural CNN + optical fallback.
- **Explainable Multi-Model Fusion:** Synthesizes evidence across all active modules to output clear hazard ratings (e.g. *Waterlogged Pothole Impact Risk*, *Hydroplaning Adhesion Loss*, *Foreign Object Corridor Obstruction*).

---

## 2. All 15 Selectable Modes & Model Registry (V2.4)

| Mode Argument | Capability Task | Architecture | Checkpoint Path | Type | Verified Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **`all`** | Complete Multi-Modal Pipeline | Master Orchestration Engine | All active models | Integrated | **`VALIDATED_ACCEPTABLE`** |
| **`damage`** | Cracks & Potholes | Custom YOLO26n | `runs/detect/road_damage_v24/weights/best.pt` | Retrained with D10 Augmentation | **`VALIDATED_ACCEPTABLE`** |
| **`surface`** | Visual Surface Classification | MobileNetV3 Multi-Task | `runs/surface_classifier_v24/best_surface_model.pt` | Effective-Number Retrained | **`VALIDATED_ACCEPTABLE`** *(Visual)* <br> **`SENSOR_REQUIRED`** *(Physical $\mu$)* |
| **`water`** | Waterlogging Segmentation | LRASPP MobileNetV3-Large | *Quarantined* | Authentic Intake Ready | **`DATASET_REQUIRED`** |
| **`vehicles`** | Vehicle Detection & Tracking | YOLO26n-seg + ByteTrack | `yolo26n-seg.pt` | Pretrained COCO | **`PRETRAINED_NOT_VALIDATED`** |
| **`road_scene`** | Semantic Road Segmentation | YOLO26n-sem | `yolo26n-sem.pt` | Pretrained Cityscapes | **`PRETRAINED_NOT_VALIDATED`** |
| **`depth`** | Monocular Depth Estimation | YOLO26n-depth | `yolo26n-depth.pt` | Pretrained Relative | **`CALIBRATION_REQUIRED`** |
| **`accident`** | Video Collision Recognition | 16-Frame 3D-CNN + Trajectory | `road_cv/accident/` | Video Architecture | **`PRETRAINED_NOT_VALIDATED`** *(Heuristic)* <br> **`DATASET_REQUIRED`** *(3D ST-GCN)* |
| **`weather`** | Weather & Lighting State | MobileNetV3-Small Dual-Head | `runs/weather_classifier_v22/best_weather_model.pt` | Custom Neural | **`TRAINED_NOT_VALIDATED`** |
| **`lanes`** | Lane Detection & Boundaries | Row-wise Anchor Polyline Model | `runs/lane_detector_v22/best_lane_model.pt` | Custom Neural | **`TRAINED_NOT_VALIDATED`** |
| **`traffic_signals`**| Traffic Light State | YOLO26n + Dedicated State CNN | Built-in HSV Vertical Thirds | Hybrid Neural + Optical | **`VALIDATED_ACCEPTABLE`** *(Optical)* <br> **`DATASET_REQUIRED`** *(Patch CNN)* |
| **`obstacles`** | Open-Vocabulary Obstacles | YOLO-World Zero-Shot Model | `yolov8s-world.pt` | Open-Vocabulary Prompted | **`PRETRAINED_NOT_VALIDATED`** |
| **`pothole_geometry`**| Calibrated Pothole Dimensions | Planar Homography Assessor | `road_cv/geometry/damage_assessor.py` | Geometric Model | **`CALIBRATION_REQUIRED`** |
| **`friction`** | Tyre-Road Friction Estimation | Qualitative Slip Risk Engine | `road_cv/surface/` + Telemetry | Sensor Interface | **`SENSOR_REQUIRED`** |
| **`roughness`** | Road Roughness / IRI | Visual Roughness Classifier | `road_cv/surface/` + IMU | Sensor Interface | **`SENSOR_REQUIRED`** |

---

## 3. How to Run CYPhER

### 3.1 Interactive Mode Selector
Execute `main.py` without arguments:
```bash
python main.py
```
CYPhER displays an interactive menu listing all 15 modes, allowing selection of single, multiple, or `all` modes.

### 3.2 Running Single or Unified Modes
```bash
# Complete multi-modal hazard intelligence pipeline
python main.py --mode all --source Road.jpg

# High-precision road damage detection
python main.py --mode damage --source Road.jpg --profile balanced

# Real-time road surface condition classification
python main.py --mode surface --source Road.jpg
```

### 3.3 Adaptive Compute Profiles
- **`--profile fast`**: Optimized for high-speed highway cruising ($> 75 \text{ km/h}$). Latency $\le 12 \text{ ms}$ ($> 80 \text{ FPS}$).
- **`--profile balanced`**: Default operational profile for suburban and highway driving ($35-75 \text{ km/h}$). Latency $\approx 27 \text{ ms}$ (~36 FPS).
- **`--profile high_accuracy`**: Specialized for pavement distress inspection and low-speed crawling ($< 35 \text{ km/h}$). Multi-tile refinement.
