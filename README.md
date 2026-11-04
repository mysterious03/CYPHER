# CYPhER: Complete Multi-Model Road Perception & Hazard Intelligence System

**CYPhER** (Comprehensive Yield & Pavement Hazard Estimation for Roads) is an end-to-end, modular computer vision application that unifies **15 perception and hazard intelligence capabilities** into a single software suite.

Engineered specifically for local execution on edge and laptop hardware with **6 GB VRAM** (such as the NVIDIA GeForce RTX 4050 Laptop GPU), CYPhER coordinates deep learning models, explainable multi-model hazard fusion, and physical sensor telemetry interfaces.

---

## 1. What CYPhER Does

CYPhER transforms raw dashcam images, video streams, or live camera feeds into structured, actionable road intelligence:
- **Structural Pavement Damage:** Detects longitudinal cracks (`D00`), transverse cracks (`D10`), alligator fatigue cracks (`D20`), and potholes (`D40`).
- **Surface & Grip Intelligence:** Classifies visual pavement wetness, road construction material, and surface macro-roughness.
- **Waterlogging & Flood Hazards:** Identifies puddles, standing water, and concealed waterlogged potholes.
- **Traffic & Obstacle Awareness:** Tracks vehicles and pedestrians with ByteTrack, classifies traffic signals, and flags open-vocabulary road debris.
- **Road Geometry & Context:** Extracts drivable road masks, estimates lane boundaries, and projects planar depth anomalies.
- **Explainable Multi-Model Fusion:** Synthesizes evidence across all active modules to output clear hazard ratings (e.g. *Waterlogged Pothole Impact Risk*, *Hydroplaning Adhesion Loss*, *Vehicle Collision*).

---

## 2. All 15 Selectable Modes & Model Registry

| Mode Argument | Capability Task | Architecture | Checkpoint Path | Type | Official Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **`all`** | Complete Multi-Modal Pipeline | Master Orchestration Engine | All active models | Integrated | **`IMPLEMENTED_AND_TRAINED`** |
| **`damage`** | Cracks & Potholes | YOLO26n | `runs/detect/runs/road_damage/B0_baseline/weights/best.pt` | Custom Fine-Tuned | **`IMPLEMENTED_AND_TRAINED`** |
| **`vehicles`** | Vehicle Detection & Tracking | YOLO26n-seg + ByteTrack | `yolo26n-seg.pt` | Pretrained COCO | **`PRETRAINED_INTEGRATED`** |
| **`road_scene`** | Semantic Road Segmentation | YOLO26n-sem | `yolo26n-sem.pt` | Pretrained Cityscapes | **`PRETRAINED_INTEGRATED`** |
| **`water`** | Waterlogging Segmentation | LRASPP MobileNetV3 | `runs/water_segmenter/best_water_model.pt` | Custom Pilot | **`TRAINING_REQUIRED`** |
| **`depth`** | Monocular Depth Estimation | YOLO26n-depth | `yolo26n-depth.pt` | Pretrained Relative | **`PRETRAINED_INTEGRATED`** |
| **`surface`** | Visual Surface Classification | MobileNetV3 Multi-Task | `runs/surface_classifier/best_surface_model.pt` | Custom Fine-Tuned | **`IMPLEMENTED_AND_TRAINED`** |
| **`accident`** | Video Collision Recognition | Trajectory & Anomaly Net | `road_cv/temporal/` | Custom Algorithmic | **`EXPERIMENTAL`** |
| **`weather`** | Weather & Lighting State | ConditionAssessor Engine | `road_cv/quality/` | Custom Algorithmic | **`IMPLEMENTED_AND_TRAINED`** |
| **`lanes`** | Lane Detection & Boundaries | Road Edge + Polyline Fitter | `road_cv/geometry/` | Custom Algorithmic | **`EXPERIMENTAL`** |
| **`traffic_signals`**| Traffic Light State | YOLO26n + Crop HSV Classifier | `yolo26n.pt` + State Engine | Pretrained + Custom | **`PRETRAINED_INTEGRATED`** |
| **`obstacles`** | Open-Vocabulary Obstacles | Open-Set Prompt Detector | `yolo26n.pt` + Prompt Head | Pretrained + Custom | **`EXPERIMENTAL`** |
| **`friction`** | Tyre-Road Friction Estimation | Qualitative Slip Risk Engine | `road_cv/surface/` + Telemetry | Sensor Interface | **`SENSOR_REQUIRED`** |
| **`roughness`** | Road Roughness / IRI | Visual Roughness Classifier | `road_cv/surface/` + IMU | Sensor Interface | **`SENSOR_REQUIRED`** |
| **`pothole_geometry`**| Calibrated Pothole Dimensions | Planar Homography Assessor | `road_cv/geometry/damage_assessor.py` | Geometric Model | **`EXPERIMENTAL`** |

---

## 3. How to Select a Mode and Run CYPhER

### 3.1 Interactive Mode Selector
Simply execute `main.py` without arguments:
```bash
python main.py
```
CYPhER will display an interactive CLI menu listing all 15 modes, allowing you to select single, multiple (e.g. `damage,surface,water`), or `all` modes, and specify the input source.

### 3.2 Running on an Image
```bash
# Run complete multi-modal road perception
python main.py --mode all --source Road.jpg

# Run specific detection modes
python main.py --mode damage --source Road.jpg
python main.py --mode surface --source Road.jpg
python main.py --mode traffic_signals --source Road.jpg
```

### 3.3 Running on a Video
```bash
# Run vehicle tracking on dashcam video
python main.py --mode vehicles --source dashcam_video.mp4

# Run video collision and accident recognition
python main.py --mode accident --source dashcam_video.mp4

# Run complete perception on video
python main.py --mode all --source dashcam_video.mp4
```

### 3.4 Running on Live Camera
```bash
# Connect to default USB/webcam (index 0)
python main.py --mode all --source 0

# Connect to secondary camera (index 1)
python main.py --mode damage --source 1
```

---

## 4. Structured JSON Output Format

CYPhER saves structured JSON output to `outputs/cypher_<source>_result.json` following the Phase 6 standard schema:

```json
{
  "source": "Road.jpg",
  "timestamp": "2026-11-05T01:52:14",
  "image_size": { "width": 2400, "height": 1604 },
  "selected_mode": "all",
  "executed_modes_count": 14,
  "capabilities": {
    "damage": {
      "status": "success",
      "model_name": "Custom YOLO26n Road Damage Detector",
      "predictions": []
    },
    "vehicles": {
      "status": "success",
      "predictions": [
        { "track_id": 1, "class_name": "car", "confidence": 0.853, "box": [1981, 1107, 2351, 1372] }
      ]
    },
    "surface": {
      "status": "success",
      "predictions": {
        "visual_surface_condition": { "class_name": "Dry", "confidence": 0.71 },
        "road_material": { "class_name": "Asphalt", "confidence": 0.74 },
        "visual_unevenness": { "class_name": "Smooth", "confidence": 0.63 }
      }
    },
    "friction": {
      "status": "requires_sensor",
      "predictions": {
        "visual_slip_risk_estimate": "LOW_SLIP_RISK_HIGH_GRIP",
        "scientific_disclaimer": "RGB cameras estimate visual condition only. Direct numerical tyre-road friction coefficient (mu) requires wheel-slip telemetry."
      }
    }
  },
  "fusion": {
    "overall_road_safety_risk": "CRITICAL_HAZARD",
    "fusion_confidence": 0.86,
    "detected_hazards_count": 1,
    "hazards": [
      {
        "hazard_type": "ACTIVE_COLLISION_OR_ROLLOVER_INCIDENT",
        "risk_level": "CRITICAL_HAZARD",
        "reason": "Temporal video anomaly confirms vehicle trajectory overlap or abnormal vehicular rollover orientation."
      }
    ]
  },
  "performance": {
    "total_pipeline_latency_ms": 2482.09,
    "fps": 0.4,
    "peak_gpu_memory_mb": 324.61,
    "device": "cuda"
  },
  "visualization_path": "outputs/cypher_Road_overlay.jpg"
}
```

---

## 5. How to Train Individual Models

### 5.1 Training the Road Damage Detector (YOLO26n)
```bash
python train_b0_baseline.py --epochs 38 --batch 4 --device 0
```
- **Dataset:** `data/processed/rdd2022_india/data.yaml`
- **Output:** `runs/detect/runs/road_damage/B0_baseline/weights/best.pt`

### 5.2 Training the Multi-Task Surface Classifier (MobileNetV3)
```bash
python train_phase1_models.py
```
- **Dataset:** `data/processed/rscd_surface/`
- **Output:** `runs/surface_classifier/best_surface_model.pt`

### 5.3 Training the Authentic Waterlogging Segmenter (LRASPP)
```bash
python scripts/prepare_authentic_water_pipeline.py
python -c "from road_cv.water import train_water_model; print('Training harness ready')"
```
- **Dataset:** `data/processed/water_authentic/`

---

## 6. Hardware Support & Memory Management

- **Target Architecture:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM), Windows 11.
- **Memory Optimization:** Uses sequential model execution and dynamic GPU memory cleanup. Even during the complete 14-model `all` mode pipeline, peak VRAM remains under **350 MB** (leaving over 5.6 GB of safety headroom).
- **CPU Fallback:** Fully supported. If CUDA is unavailable or disabled, all models automatically switch to CPU execution.

---

## 7. Scientific Honesty & Known Limitations

1. **Physical Tyre-Road Friction ($\mu$):** Visual surface wetness is an estimate of pavement moisture state, not a direct measurement of the tyre-road friction coefficient $\mu$. The system reports `VISUAL_SLIP_RISK_ESTIMATE` and marks true friction as `SENSOR_REQUIRED`.
2. **Road Roughness (IRI):** Visual unevenness indicates surface texture degradation. Standard International Roughness Index (IRI in m/km) requires vertical accelerometer telemetry and is marked `SENSOR_REQUIRED`.
3. **Pothole Metric Geometry:** Monocular cameras provide relative pixel area and ordinal depth displacement. True metric centimetre dimensions require verified extrinsic/intrinsic camera calibration (`is_calibrated=True`).
4. **Water Segmentation Labels:** The legacy water segmenter was trained on synthetic HSV pseudo-masks. It is currently queued for retraining on genuine pixel-level water masks from R²S100K.
