# CYPhER: Multi-Model Road Perception & Hazard Intelligence System
> **Real-Time, Domain-Adaptive, and Uncertainty-Aware Autonomous Road Perception Architecture**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2Bcu132-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%20Enabled-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Hardware](https://img.shields.io/badge/Target%20GPU-RTX%204050%20(6GB)-76B900.svg)](https://www.nvidia.com)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics%208.4-00FFFF.svg)](https://ultralytics.com/)
[![Tests](https://img.shields.io/badge/Tests-16%2F16%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 1. Project Overview

**CYPhER** (**C**omprehensive **Y**ield & **P**avement **H**azard **E**stimation for **R**oads) is an end-to-end multi-model computer vision and deep learning perception suite designed for unstructured, complex, and high-variance road conditions (such as Indian national highways, urban arterial streets, and rural roads).

Rather than relying on a single brittle monolithic neural network, CYPhER orchestrates a modular ensemble of specialized deep learning models—spanning **road damage detection, vehicle tracking, surface classification, waterlogging segmentation, dense monocular depth estimation, collision/incident recognition, lane perception, traffic signal state estimation, and zero-shot open-vocabulary obstacle perception**. 

CYPhER features a **Real-Time Live Camera Streaming Engine (`stream.py`)** capable of ingesting phone camera feeds (wireless via Wi-Fi IP Webcam or zero-latency wired via USB Tethering) and rendering resolution-adaptive, thick, color-coded bounding boxes and telemetry HUDs in real time on NVIDIA RTX GPUs.

```
+---------------------------------------------------------------------------------------------------+
|                                  CYPhER ARCHITECTURAL WORKFLOW                                    |
+---------------------------------------------------------------------------------------------------+
|  [Input Sources]                                                                                  |
|  * Live Mobile Phone Feed (Wi-Fi / USB Tethering)                                                 |
|  * Built-in / USB Webcams                                                                         |
|  * Dashcam Video Files (.mp4, .avi, .mkv)                                                         |
|  * High-Res Photographic Inputs (.jpg, .png)                                                      |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                             UNIFIED ORCHESTRATOR & SCHEDULER                                      |
|  - Sequential Model Loading & VRAM Protection (RTX 4050 6GB envelope)                             |
|  - Resolution-Adaptive Spatial Scaling (480p -> 4K)                                               |
|  - Video ByteTrack Tracking History                                                               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+--------------------------------------------------+ +----------------------------------------------+
|            CORE STRUCTURAL PERCEPTION            | |         DYNAMIC AGENTS & ENVIRONMENT         |
+--------------------------------------------------+ +----------------------------------------------+
| M1: Damage Detection (YOLO12s RDD2022)           | | M2: Vehicle & User Detection (YOLO26n-seg)   |
|     * Potholes (D40)                             | |     * Cars, Trucks, Buses, Bikes, Persons    |
|     * Alligator Cracks (D20)                     | | M11: Path Tracker & Trajectory (ByteTrack)   |
|     * Longitudinal & Transverse Cracks (D00/D10) | | M6: Collision & Rollover Intelligence        |
| M3: Surface Material & Condition (MobileNetV3)   | | M7: Environmental Weather (SigLIP2 / CNN)    |
| M4: Waterlogging & Puddles (YOLOv8n-Seg)         | | M8: Lane Boundaries & Shoulder (Polyline)    |
| M5: Dense Relative Depth (Depth Anything V2)     | | M9: Traffic Light States (YOLOv8m + Optical) |
|                                                  | | M10: Open-Vocabulary Obstacles (YOLO-World)  |
+--------------------------------------------------+ +----------------------------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                             EXPLAINABLE HAZARD ARBITER & FUSION                                   |
|  - Synthesizes multi-model evidence into actionable safety ratings:                               |
|    [!] Waterlogged Pothole Impact Risk       [!] Imminent Collision / Crash Incident              |
|    [!] Hydroplaning / Slip Risk              [!] Foreign Object Corridor Obstruction              |
|  - Physical Sensor Safeguards: Friction (mu), Roughness (IRI), 3D Pothole Geometry                |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+--------------------------------------------------+ +----------------------------------------------+
|             LIVE INTERACTIVE HUD DISPLAY         | |              STRUCTURED JSON AUDIT           |
|  * Bold, resolution-adaptive bounding boxes      | |  * Complete capability results schema        |
|  * Distinct per-class high-contrast colors       | |  * Fusion hazards & confidence metrics       |
|  * Solid filled badge tags (100% visible)        | |  * Pipeline latency (ms) & FPS telemetry     |
+--------------------------------------------------+ +----------------------------------------------+
```

---

## 2. Full AI Perception Suite (16 Modes)

| Mode CLI Argument | Capability Task | Backbone Architecture | Primary Checkpoint / Source | Dataset / Provenance | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`all`** | **Complete Multi-Model Suite** | Unified Pipeline Ensemble | All active models | Integrated Pipeline | **Verified** |
| **`damage`** | Potholes & Pavement Cracks | YOLO12s / Custom YOLO26n | `models/pretrained/damage/yolo12s_RDD2022_best.pt`<br>`runs/detect/road_damage_v25/weights/best.pt` | RDD2022 India (Crowdsensing) | **Verified** |
| **`vehicles`** | Multi-Class Vehicle Detection | YOLO26n-seg | `yolo26n-seg.pt` | COCO 80-Class (Transferred) | **Verified** |
| **`vehicle_tracking`** | Vehicle Trajectories & Vectors | YOLO26n-seg + ByteTrack | `road_cv/tracking/vehicle_path_tracker.py` | Integrated ByteTrack Tracker | **Verified** |
| **`water`** | Waterlogging & Puddle Detection | YOLOv8n-Seg / LRASPP MobileNetV3 | `models/pretrained/water/road_wetness_yolov8n_seg.pt` | Lithuanian Wetness Dataset | **Verified** |
| **`surface`** | Pavement Material & Quality | MobileNetV3 / EfficientNetV2 | `models/pretrained/surface/v1/surface_type_v1.pt` | Multi-Task SurfaceAI | **Verified** |
| **`accident`** | Collision & Rollover Intelligence | Spatio-Temporal Collision Engine | `road_cv/accident/temporal_accident_detector.py` | Geometric Anomaly + VideoMAE | **Verified** |
| **`road_scene`** | Semantic Drivable Corridor | YOLO26n-sem / SegFormer | `yolo26n-sem.pt` | Cityscapes / IDD Benchmark | **Verified** |
| **`depth`** | Monocular Dense Relative Depth | Depth Anything V2 Small | `models/pretrained/depth/depth_anything_v2_small/` | Pretrained Depth Anything V2 | **Verified** |
| **`weather`** | Atmospheric & Illumination | Multi-Task MobileNetV3 / SigLIP 2 | `models/pretrained/weather/weather_siglip2/` | Environmental Weather Benchmark | **Verified** |
| **`lanes`** | Lane Boundaries & Unmarked Edges | Dedicated Polyline Anchor / CLRNet | `runs/lane_detector_v22/best_lane_model.pt` | CULane / Indian Road Survey | **Verified** |
| **`traffic_signals`**| Traffic Lights & Lamp State | YOLOv8m + Optical HSV Head | `models/pretrained/signals/traffic_signals_yolov8m.pt` | Pretrained Traffic Lights | **Verified** |
| **`obstacles`** | Open-Vocabulary Road Obstacles | YOLO-World Zero-Shot Model | `yolov8s-world.pt` | Zero-Shot Text Prompts | **Verified** |
| **`pothole_geometry`**| Pothole 3D Dimensions & Volume | Planar Homography Assessor | `road_cv/geometry/damage_assessor.py` | Calibrated Camera Model | **Gated** |
| **`friction`** | Tyre-Road Skid Resistance ($\mu$)| Vision Slip Estimator | `road_cv/surface/friction_classifier.py` | Sensor Telemetry Safeguard | **Gated** |
| **`roughness`** | International Roughness Index (IRI)| Macro-Roughness Estimator | `road_cv/surface/` + Telemetry | Sensor Telemetry Safeguard | **Gated** |

---

## 3. Real-Time Camera Streaming (`stream.py`)

CYPhER includes a dedicated high-performance live camera streaming pipeline designed for direct deployment with smartphones and dashcams.

### 3.1 Connecting Your Mobile Phone Camera

#### Method 1: Wireless over Wi-Fi (No Cables Needed)
1. Install **IP Webcam** (by Pavel Khlebovich) on Android (Play Store) or **DroidCam / Camo** on iOS.
2. Connect your phone and PC to the same Wi-Fi router (or connect your laptop to your phone's Wi-Fi hotspot).
3. Open the app and tap **"Start server"**. The phone will display an IP address (e.g. `http://192.168.0.213:8080`).
4. Launch CYPhER:
   ```bash
   python stream.py --source http://192.168.0.213:8080/video --mode all
   ```

#### Method 2: Wired via USB Tethering (Zero Latency & Rock-Solid)
1. Connect your phone to your PC using a USB cable.
2. On your phone: Go to **Settings** $\rightarrow$ **Network & Internet** $\rightarrow$ **Hotspot & Tethering** $\rightarrow$ Turn on **USB Tethering**.
3. Open **IP Webcam** on your phone and tap **Start server**.
4. Use the USB IP address displayed (e.g. `http://192.168.42.129:8080`):
   ```bash
   python stream.py --source http://192.168.42.129:8080/video --mode all
   ```

#### Method 3: USB Virtual Webcam Device (Plug & Play)
Install **DroidCam** or **Iriun 4K Webcam** on both phone and PC. It will register as a Windows webcam index (usually `1` or `2`):
```bash
python stream.py --source 1 --mode all
```

---

### 3.2 High-Contrast Visual HUD & Color Palette

CYPhER renders thick, resolution-adaptive bounding boxes with solid filled background tag badges, ensuring 100% legibility in sunlight, glare, and night conditions.

| Object Category | Color Code | Visual Representation |
| :--- | :--- | :--- |
| 🕳️ **Potholes (D40)** | `(30, 30, 255)` | **Crimson Alert Red** (Bold border + solid red badge) |
| ⚡ **Alligator Cracks (D20)** | `(0, 100, 255)` | **Deep Orange** |
| ⚡ **Longitudinal Cracks (D00)**| `(0, 180, 255)` | **Vivid Amber** |
| ⚡ **Transverse Cracks (D10)** | `(0, 220, 255)` | **Bright Gold** |
| 🚗 **Cars** | `(255, 190, 0)` | **Neon Cyan-Blue** |
| 🚚 **Trucks** | `(0, 130, 255)` | **Deep Orange** |
| 🚌 **Buses** | `(220, 20, 220)` | **Neon Magenta / Purple** |
| 🏍️ **Motorcycles** | `(0, 230, 255)` | **Electric Yellow** |
| 🚲 **Bicycles** | `(80, 240, 100)` | **Spring Green** |
| 🚶 **Pedestrians / Persons** | `(120, 80, 255)` | **Coral Pink** |
| 🛺 **Auto-Rickshaws** | `(0, 190, 255)` | **Vivid Gold** |
| 💧 **Water & Puddles** | `(255, 210, 0)` | **Cyan-Blue Badge + Submerged % Banner** |
| 🚦 **Traffic Signals** | `(0, 0, 255)` / `(0, 255, 0)` | **Active Signal State (Red / Amber / Green)** |
| 🚨 **Collisions & Rollovers** | `(0, 0, 255)` | **Flashing Red Emergency Banner + Crash Boxes** |

---

### 3.3 Interactive Live Window Controls

While the live video window is running, use these keyboard shortcuts:
- **`M`**: Cycle perception modes on the fly (`all`, `vehicles,damage,water,accident`, `vehicles`, `damage`, `water`, `accident`, `surface`, `traffic_signals`).
- **`S`**: Save a high-resolution snapshot with all annotations and HUD telemetry directly to `outputs/stream_captures/`.
- **`P`**: Pause / Resume the live video stream.
- **`Q`** or **`ESC`**: Exit the stream cleanly.

---

## 4. Hardware Optimization (RTX 4050 6GB)

CYPhER was specifically engineered to run deep vision ensembles inside tight VRAM budgets without triggering CUDA out-of-memory errors:

- **Sequential Model Scheduling:** The `GPUResourceScheduler` tracks allocated VRAM with a 1.0 GB safety headroom. If VRAM exceeds 4.0 GB, inactive adapters are systematically flushed.
- **Low-Latency Camera Buffer:** DirectShow hardware acceleration (`cv2.CAP_DSHOW`) with `CAP_PROP_BUFFERSIZE = 1` flushes stale frames so perception is always synchronized with the vehicle's actual position.
- **Precision Profiles:**
  - **`FAST`**: ~371 ms latency (~2.7 FPS, 139 MB peak VRAM).
  - **`BALANCED`**: ~2.1 s latency (404 MB peak VRAM).
  - **`HIGH_ACCURACY`**: Full 14-model depth and open-vocabulary stack (404 MB peak VRAM).

---

## 5. Repository File Structure

```text
cvvvvvv/
├── configs/                                # System & model registry configurations
│   ├── cypher.yaml                         # Master orchestrator configuration
│   ├── model_registry.yaml                 # Core capability model paths & hyper-params
│   ├── pretrained_model_registry.yaml      # Pretrained weights catalogue & mappings
│   ├── pretrained_models.yaml              # Execution profile definitions
│   └── vehicle_tracking.yaml               # ByteTrack tracker configuration
├── road_cv/                                # Core CYPhER Python package
│   ├── accident/                           # Module 6: Collision & temporal anomaly
│   │   ├── simple_tad_adapter.py           # VideoMAE ViT-S temporal anomaly adapter
│   │   └── temporal_accident_detector.py   # Spatio-temporal 3D-CNN & collision engine
│   ├── adapters/                           # Model inference wrappers & interfaces
│   │   ├── base.py                         # Abstract BaseAdapter class
│   │   ├── pretrained_damage_adapter.py    # YOLO12s RDD2022 high-accuracy adapter
│   │   ├── specialized_adapters.py         # Weather, Lane, Signal, Obstacle adapters
│   │   └── vision_adapters.py              # Damage, Vehicle, Scene, Depth, Water adapters
│   ├── adaptive/                           # Domain-adaptive perception & tile refiner
│   │   ├── pipeline.py                     # Adaptive multi-profile orchestrator
│   │   ├── spatial_selector.py             # Corridor saliency & tile selector
│   │   ├── tile_refiner.py                 # High-resolution patch zoom refiner
│   │   └── uncertainty_estimator.py        # Predictive uncertainty estimation
│   ├── core/                               # Central application intelligence
│   │   ├── fusion.py                       # Explainable multi-model hazard arbiter
│   │   ├── orchestrator.py                 # Master CYPhER unified execution engine
│   │   └── pretrained_model_manager.py     # Profile & memory manager
│   ├── depth/                              # Module 5: Monocular dense relative depth
│   │   └── pretrained_depth_adapter.py     # Depth Anything V2 Small adapter
│   ├── geometry/                           # 3D damage & calibrated camera models
│   │   └── damage_assessor.py              # Homography & pothole metric geometry
│   ├── lanes/                              # Module 8: Lane boundary & shoulder detection
│   │   ├── clrernet_adapter.py             # CLRerNet CULane adapter
│   │   └── lane_detector.py                # Dedicated polyline anchor model
│   ├── obstacles/                          # Module 10: Open-vocabulary road obstacles
│   │   ├── open_vocabulary_obstacles.py    # YOLO-World natural language detector
│   │   └── pretrained_obstacle_adapter.py  # Zero-shot obstacle wrapper
│   ├── scene/                              # Module 7: Semantic corridor segmentation
│   │   └── idd_drivable_adapter.py         # Indian Driving Dataset drivable adapter
│   ├── scheduling/                         # Hardware & sequential GPU management
│   │   └── resource_scheduler.py           # VRAM monitor & adapter lifecycle
│   ├── surface/                            # Module 3: Road surface classification
│   │   ├── friction_classifier.py          # Qualitative tyre-road slip estimator
│   │   ├── pretrained_surface_adapter.py   # SurfaceAI EfficientNetV2 adapter
│   │   └── surface_predictor.py            # MobileNetV3 multi-head predictor
│   ├── tracking/                           # Module 2 & 11: Multi-vehicle tracking
│   │   ├── trajectory_analyzer.py          # Vector heading & contact patch analyzer
│   │   └── vehicle_path_tracker.py         # ByteTrack vehicle tracking pipeline
│   ├── traffic_signals/                    # Module 9: Traffic signals & signs
│   │   ├── pretrained_signal_adapter.py    # YOLOv8m traffic light adapter
│   │   └── signal_classifier.py            # Dedicated HSV optical state classifier
│   ├── water/                              # Module 4: Waterlogging & wet road detection
│   │   ├── pretrained_puddle_adapter.py    # Puddle instance segmenter
│   │   ├── pretrained_wetness_adapter.py   # YOLOv8n wet road segmenter
│   │   ├── water_analyzer.py               # Water depth & pothole overlap analyzer
│   │   └── water_segmenter.py              # LRASPP MobileNetV3 segmenter
│   └── weather/                            # Module 7: Environmental condition
│       ├── pretrained_weather_adapter.py   # SigLIP 2 zero-shot weather adapter
│       └── weather_model.py                # Multi-task CNN weather classifier
├── reports/                                # Documentation & benchmarks
│   └── PRETRAINED_MODELS_MASTER_REPORT.md  # Official model manifest, metrics & SHA-256
├── scripts/                                # Maintenance, training & verification scripts
├── tests/                                  # Automated regression test suite
│   └── test_cypher_system.py               # 16-test complete verification suite
├── main.py                                 # Primary CLI entry point & single-run tool
├── stream.py                               # Real-time live camera streaming engine
├── requirements.txt                        # Python dependencies
├── Road.jpg                                # Sample test verification frame
├── .gitignore                              # Git exclusion rules (weights, caches, outputs)
└── README.md                               # Master documentation
```

---

## 6. Installation & Quick Start

### 6.1 Prerequisites
- **OS:** Windows 10/11 or Ubuntu 20.04+
- **Python:** 3.11 recommended
- **NVIDIA GPU:** RTX 3050 / 4050 / 4060 or higher with CUDA 12.x+

### 6.2 Setup
```bash
# Clone the repository
git clone https://github.com/mysterious03/CYPHER.git
cd CYPHER

# Create and activate a Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux:
source venv/bin/activate

# Install PyTorch with CUDA acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 6.3 Verify Installation
Run the automated 16-test suite to confirm that every module, model adapter, and fallback safeguard is operational:
```bash
pytest tests
```
*Expected: `16 passed in ~10s (100% pass rate)`*

---

## 7. Execution Interfaces & Command Reference

### 7.1 Minimalist Web Dashboard (`app.py`)
Launch the modern, minimalist web dashboard:
```bash
python app.py
```
Then navigate to **`http://127.0.0.1:8000`** in any web browser:
- **Photo Drag & Drop**: Drop any road photo for instant hazard scoring, pothole identification, vehicle classification, and side-by-side original/overlay toggle.
- **Video Analysis**: Upload dashcam video clips for collision analysis and vehicle trajectory tracking.
- **Live Camera Feed**: Connect phone IP webcam streams (`http://<ip>:8080/video`) or USB webcams with live telemetry polling and safety risk scorecards.

---

### 7.2 Single-Image or Batch Inference (`main.py`)
```bash
# 1. Complete multi-modal pipeline on an image:
python main.py --mode all --source Road.jpg

# 2. Road damage detection (cracks & potholes):
python main.py --mode damage --source Road.jpg

# 3. Vehicle detection & tracking:
python main.py --mode vehicles --source Road.jpg

# 4. Multi-hazard perception (damage, vehicles, water, accident):
python main.py --mode damage,vehicles,water,accident --source Road.jpg

# 5. Process a dashcam video file:
python main.py --mode vehicles --source dashcam_video.mp4
```

### 7.2 Real-Time Live Camera Streaming (`stream.py`)
```bash
# 1. Stream from Android phone (IP Webcam Wi-Fi):
python stream.py --source http://192.168.0.213:8080/video --mode all

# 2. Fast multi-hazard mode (optimized for high FPS):
python stream.py --source http://192.168.0.213:8080/video --mode vehicles,damage,water,accident

# 3. Stream from USB Webcam (device index 0 or 1):
python stream.py --source 0 --mode vehicles

# 4. Stream with frame stride (infer every 2nd frame for 60 FPS display):
python stream.py --source 0 --mode all --skip-frames 1
```

---

## 8. License & Attribution

- **CYPhER Framework:** Released under the [MIT License](LICENSE).
- **Pretrained Checkpoint Provenance:** Checkpoint licenses, weights hashes (SHA-256), and dataset citations are detailed in [`reports/PRETRAINED_MODELS_MASTER_REPORT.md`](reports/PRETRAINED_MODELS_MASTER_REPORT.md).
