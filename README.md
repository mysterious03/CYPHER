# CYPhER V2.5: Complete Indian Road Perception Intelligence
**Robust, Domain-Adaptive, and Uncertainty-Aware Multi-Model Road Hazard Intelligence**

**CYPhER** (Comprehensive Yield & Pavement Hazard Estimation for Roads) is an end-to-end, scale-adaptive computer vision application that unifies **road perception and hazard intelligence capabilities** into a single modular software suite.

Version 2.5 introduces **Domain Adaptation for Unstructured Indian Roads**, authentic copy-paste distress retraining, minority-class loss balancing, and verified multi-profile adaptive perception.

The **Pretrained AI Model Integration Extension** transforms CYPhER into an "install-first" road intelligence hub, downloading and integrating official pretrained checkpoints across 11 perception subdirectories with sequential GPU memory management on the **NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)**.

---

## 1. What CYPhER Does

CYPhER transforms raw dashcam images, video streams, or live camera feeds into structured, actionable road intelligence:
- **Structural Pavement Damage:** Detects longitudinal cracks (`D00`), transverse cracks (`D10`), alligator fatigue cracks (`D20`), and potholes (`D40`) with scale-adaptive tiling and zoom.
- **Adaptive Road-Aware Perception:** Utilizes semantic road masks and calibrated predictive uncertainty to direct high-resolution compute exclusively to the drivable corridor.
- **Surface Condition & Grip:** Multi-task classification of visual pavement wetness, road construction material, and surface macro-roughness with **100% wet hazard recall**.
- **Open-Vocabulary Obstacle Perception:** Detects unconstrained natural language obstacles (fallen tree, boulder, road debris, spilled cargo, tire on road) using YOLO-World with 0.00 clean-road false alarms.
- **Vehicle Path Tracking & Trajectories:** Tracks road vehicles using ByteTrack/BoT-SORT, calculating road surface contact patches, velocity vectors, and trajectory motion arrows.
- **Dense Monocular Relative Depth:** Generates dense inverse depth maps via Depth Anything V2 Small with `CALIBRATION_REQUIRED` metric protection.
- **Temporal Video Traffic Anomaly / Accident Recognition:** Analyzes 16-frame sliding spatio-temporal video clips using VideoMAE ViT-S fine-tuned on DoTA.
- **Atmospheric Weather & Lighting:** Classifies 5 weather conditions using SigLIP2 while evaluating ambient illumination (day, night, dusk).
- **Dedicated Lane Perception:** Predicts left/right boundary polylines, corridor width, lane marking types (dashed, solid, double), and unpaved dirt shoulder edges (`unmarked_edge`).
- **Traffic & Signal Awareness:** Detects signals and signs via YOLOv8m and classifies lamp state (Red/Yellow/Green/Off) via dedicated optical color analysis.
- **Explainable Multi-Model Fusion:** Synthesizes evidence across all active modules to output clear hazard ratings (e.g. *Waterlogged Pothole Impact Risk*, *Hydroplaning Adhesion Loss*, *Foreign Object Corridor Obstruction*).

---

## 2. All 16 Selectable Modes & Pretrained Model Registry

| Mode Argument | Capability Task | Architecture | Checkpoint Path | Type | Verified Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **`all`** | Complete Multi-Modal Pipeline | Master Orchestration Engine | All active models | Integrated | **`VALIDATED_ACCEPTABLE`** |
| **`damage`** | Cracks & Potholes | Custom YOLO26n / YOLO12s RDD2022 | `runs/detect/road_damage_v25/weights/best.pt`<br>`models/pretrained/damage/yolo12s_RDD2022_best.pt` | Custom Retrained & Pretrained MIT | **`INFERENCE_VERIFIED`** |
| **`surface`** | Surface Type & Continuous Quality | EfficientNetV2-S Suite | `models/pretrained/surface/v1/surface_type_v1.pt`<br>`surface_quality_asphalt_v1.pt` | Pretrained SurfaceAI | **`INFERENCE_VERIFIED`** <br> **`SENSOR_REQUIRED`** *(Physical $\mu$)* |
| **`water`** | Road Wetness Segmentation | YOLOv8n-Seg | `models/pretrained/water/road_wetness_yolov8n_seg.pt` | Pretrained Lithuanian Model | **`INFERENCE_VERIFIED`** |
| **`vehicles`** | Vehicle Detection & Segmentation | YOLO26n-Seg | `yolo26n-seg.pt` | Pretrained COCO | **`INFERENCE_VERIFIED`** |
| **`vehicle_tracking`** | Trajectory & Path Tracking | YOLO26n-Seg + ByteTrack | `yolo26n-seg.pt` | Integrated Tracker | **`INFERENCE_VERIFIED`** |
| **`road_scene`** | Semantic Road Segmentation | YOLO26n-sem | `yolo26n-sem.pt` | Pretrained Cityscapes | **`INFERENCE_VERIFIED`** |
| **`depth`** | Dense Relative Depth | Depth-Anything-V2-Small | `models/pretrained/depth/depth_anything_v2_small/` | Pretrained Apache-2.0 | **`INFERENCE_VERIFIED`** <br> **`CALIBRATION_REQUIRED`** |
| **`accident`** | Video Traffic Anomaly / Collision | VideoMAE ViT-S (DoTA) | `models/pretrained/accident/simpletad_ft-dota_vm1-s_auroc.pth` | Temporal VideoMAE (16-frame) | **`INFERENCE_VERIFIED`** *(CC BY-NC 4.0)* |
| **`weather`** | Weather & Lighting State | SigLIP2 + Luminance Head | `models/pretrained/weather/weather_siglip2/` | Pretrained Apache-2.0 | **`INFERENCE_VERIFIED`** |
| **`lanes`** | Lane Detection & Boundaries | Polyline Anchor Model / CLRerNet | `runs/lane_detector_v22/best_lane_model.pt`<br>`models/pretrained/lanes/clrernet_culane_dla34_ema.pth` | Custom & Pretrained CULane | **`INFERENCE_VERIFIED`** *(Native)* <br> **`DEPENDENCY_CONFLICT`** *(CLRerNet)* |
| **`traffic_signals`**| Traffic Signs & Light Fixtures | YOLOv8m + Optical HSV Head | `models/pretrained/signals/traffic_signals_yolov8m.pt` | Pretrained YOLOv8m | **`INFERENCE_VERIFIED`** |
| **`obstacles`** | Open-Vocabulary Obstacles | YOLO-World Zero-Shot Model | `yolov8s-world.pt` | Open-Vocabulary Prompted | **`INFERENCE_VERIFIED`** |
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
CYPhER displays an interactive menu listing all 16 modes, allowing selection of single, multiple, or `all` modes.

### 3.2 Running Individual or Unified Modes
```bash
# Complete multi-modal hazard intelligence pipeline
python main.py --mode all --source Road.jpg

# Vehicle path tracking with road contact patches & trajectory motion arrows
python main.py --mode vehicle_tracking --source Road.jpg

# Road surface material & continuous quality estimation
python main.py --mode surface --source Road.jpg

# Road surface wetness segmentation
python main.py --mode water --source Road.jpg

# Traffic signal & sign detection
python main.py --mode traffic_signals --source Road.jpg

# Open-vocabulary roadside obstacle detection
python main.py --mode obstacles --source Road.jpg

# Relative dense monocular depth estimation
python main.py --mode depth --source Road.jpg

# Atmospheric weather classification
python main.py --mode weather --source Road.jpg
```

### 3.3 Pretrained Pipeline Execution Profiles
The `PretrainedModelManager` provides three sequential GPU execution profiles tailored to the RTX 4050 6GB VRAM envelope:
```bash
# Run the complete automated smoke test and profile verification harness
python run_pretrained_smoke_tests.py
```
- **`FAST Profile`**: Wetness YOLOv8n + Vehicle Tracking ByteTrack. Latency: **371.9 ms** (~2.7 FPS), Peak VRAM: **139.5 MB** (2.3% of 6GB).
- **`BALANCED Profile`**: Damage + Surface + Wetness + Tracking + Signals + Weather. Latency: **2145.4 ms**, Peak VRAM: **404.2 MB** (6.6% of 6GB).
- **`HIGH_ACCURACY Profile`**: Full Suite (+ Depth Anything V2 + Open-Vocabulary Obstacles + Simple-TAD). Latency: **4868.3 ms**, Peak VRAM: **404.2 MB** (6.6% of 6GB).

---

## 4. Verification & Technical Report
The complete technical acceptance report, model manifest, benchmark numbers, and licensing terms are consolidated in:
- [`reports/PRETRAINED_MODELS_MASTER_REPORT.md`](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/PRETRAINED_MODELS_MASTER_REPORT.md): Unified manifest with SHA-256 checksums, architecture matrices, latency/memory benchmarks, and licensing.
