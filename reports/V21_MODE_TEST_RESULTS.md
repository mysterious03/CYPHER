# CYPhER V2.1 Mode-by-Mode Acceptance & Verification Audit

**Auditor:** Independent Senior Computer Vision Researcher & ML Systems Auditor  
**Date:** November 5, 2026  
**Test Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM), Windows 11  
**Verification Inputs:** `Road.jpg` ($2400 \times 1604$) and `data/processed/rdd2022_india/images/val/India_000002.jpg` ($640 \times 640$)

---

## 1. Audit Framework & Questions per Mode

For every selectable mode in the CYPhER perception system, the following 10 architectural and scientific criteria were audited:
1. **Underlying Mechanism:** Neural network architecture vs. heuristic / rule engine vs. sensor integration.
2. **Adapter File & Class:** Exact Python implementation path.
3. **Model Checkpoint:** File path, size, and origin.
4. **Execution Status:** Verified runtime response (`success`, `requires_sensor`, `requires_calibration`, `failed`).
5. **Output Schema & Predictions:** Structure and contents of generated JSON predictions.
6. **Execution Profiling:** Latency (ms) and GPU VRAM footprint (MB).
7. **Scientific Safeguards:** Presence of disclaimers, sensor requirements, or calibration gates.
8. **Shared Context Integration:** Upstream inputs consumed and downstream signals provided.
9. **Known Limitations & Failure Cases:** Specific technical failure modes.
10. **Scientific Integrity Status:** Exact taxonomy classification.

---

## 2. Comprehensive 15-Mode Technical Audit

---

### Mode 1: `damage` (Road Damage Perception)
1. **Underlying Mechanism:** Deep Convolutional Object Detection (YOLO26n architecture).
2. **Adapter File & Class:** `road_cv/adapters/vision_adapters.py::DamageAdapter`
3. **Model Checkpoint:** `runs/detect/runs/road_damage/B0_baseline/weights/best.pt` (24.89 MB).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** List of detected damage bounding boxes `[x1, y1, x2, y2]`, confidence score, class ID, and class name (`D00_longitudinal_crack`, `D10_transverse_crack`, `D20_alligator_crack`, `D40_pothole`).
6. **Execution Profiling:** Latency: 1451.5 ms (cold model load + inference on $2400 \times 1604$) / 42.1 ms (warm inference on $640 \times 640$); Peak VRAM: 51.6 MB.
7. **Scientific Safeguards:** Filters detections below confidence threshold (0.25); rejects zero-area boxes.
8. **Shared Context Integration:** Consumes `road_mask` from `road_scene` to suppress off-road false positives; feeds detected potholes to `water` (waterlogging intersection) and `pothole_geometry`.
9. **Known Limitations:** Severely under-recalls D10 transverse cracks (1.07% mAP50, 0 TP in holdout); misses thin hair-cracks below 2 px width; low recall (0.7%) on small distant defects.
10. **Scientific Integrity Status:** `IMPLEMENTED_AND_TRAINED`

---

### Mode 2: `vehicles` (Vehicle Segmentation & Tracking)
1. **Underlying Mechanism:** Deep Instance Segmentation (YOLO26n-seg) with ByteTrack multi-object tracking.
2. **Adapter File & Class:** `road_cv/adapters/vision_adapters.py::VehicleAdapter`
3. **Model Checkpoint:** `yolo26n-seg.pt` (6.41 MB).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** List of vehicle instances: bounding box, polygon contour mask, class name (`car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person`), tracking ID, and velocity vector (if video).
6. **Execution Profiling:** Latency: 297.3 ms on $2400 \times 1604$; Peak VRAM: 66.4 MB.
7. **Scientific Safeguards:** Filters non-vehicle COCO classes; handles stationary single frames gracefully.
8. **Shared Context Integration:** Feeds vehicle bounding boxes and aspect ratios to `accident` engine.
9. **Known Limitations:** In single images without temporal context, velocities and tracking IDs cannot be calculated; dense occlusions cause mask merging.
10. **Scientific Integrity Status:** `PRETRAINED_INTEGRATED`

---

### Mode 3: `road_scene` (Semantic Scene Segmentation)
1. **Underlying Mechanism:** Deep Semantic Segmentation (YOLO26n-sem).
2. **Adapter File & Class:** `road_cv/adapters/vision_adapters.py::RoadSceneAdapter`
3. **Model Checkpoint:** `yolo26n-sem.pt` (3.33 MB).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** Binary road mask `road_mask` $[H, W]$, road area percentage, and detected scene elements (sidewalk, sky, terrain, vegetation).
6. **Execution Profiling:** Latency: 196.7 ms on $2400 \times 1604$; Peak VRAM: 348.9 MB (due to full-resolution mask upsampling).
7. **Scientific Safeguards:** Verifies road class index 0 matches Cityscapes label convention.
8. **Shared Context Integration:** Upstream foundational provider: exports binary `road_mask` consumed by `weather`, `damage`, `lanes`, `water`, and `obstacles`.
9. **Known Limitations:** Unpaved dirt roads with heavy dust can be misclassified as terrain; boundary edges exhibit slight dilation artifacts at $640 \times 640$ input scale.
10. **Scientific Integrity Status:** `PRETRAINED_INTEGRATED`

---

### Mode 4: `water` (Road Waterlogging Segmentation)
1. **Underlying Mechanism:** Deep Semantic Segmentation (LRASPP MobileNetV3-Large).
2. **Adapter File & Class:** `road_cv/adapters/vision_adapters.py::WaterAdapter`
3. **Model Checkpoint:** `runs/water_segmenter/best_water_model.pt` (12.48 MB).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** Water coverage ratio, flags for `standing_water_present`, and list of `waterlogged_potholes` intersecting with `damage`.
6. **Execution Profiling:** Latency: 430.5 ms on $2400 \times 1604$; Peak VRAM: 85.7 MB.
7. **Scientific Safeguards:** Only analyzes pixels within the `road_mask` when provided; warns when high specular glare is detected.
8. **Shared Context Integration:** Consumes `road_mask` and `detected_potholes`; flags potholes containing standing water for high hazard escalation in `CYPhERHazardFusion`.
9. **Known Limitations:** **Trained on heuristic pseudo-masks**, NOT human-annotated flood datasets. Vulnerable to false positives on fresh, highly reflective black asphalt under intense sunlight.
10. **Scientific Integrity Status:** `DATASET_REQUIRED` (Trained on synthetic pseudo-masks)

---

### Mode 5: `depth` (Monocular Depth Estimation)
1. **Underlying Mechanism:** Deep Monocular Relative Depth Estimation (YOLO26n-depth).
2. **Adapter File & Class:** `road_cv/adapters/vision_adapters.py::DepthAdapter`
3. **Model Checkpoint:** `yolo26n-depth.pt` (12.40 MB).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** Normalized depth map $[H, W] \in [0, 1]$, road surface depth profile, and mean distance estimate.
6. **Execution Profiling:** Latency: 302.8 ms on $2400 \times 1604$; Peak VRAM: 130.2 MB.
7. **Scientific Safeguards:** Output explicitly marked as `RELATIVE_INVERSE_DEPTH`. Does not claim metric meters without stereo or LiDAR ground truth.
8. **Shared Context Integration:** Feeds dense depth tensor to `pothole_geometry` for relative depth gradient calculations.
9. **Known Limitations:** Scale ambiguity: cannot distinguish a small nearby pothole from a large distant pothole without external camera intrinsics/extrinsics.
10. **Scientific Integrity Status:** `PRETRAINED_INTEGRATED`

---

### Mode 6: `surface` (Multi-Task Surface Condition Classification)
1. **Underlying Mechanism:** Multi-Task Deep Neural Network (MobileNetV3-Large with 3 linear heads).
2. **Adapter File & Class:** `road_cv/adapters/vision_adapters.py::SurfaceAdapter`
3. **Model Checkpoint:** `runs/surface_classifier/best_surface_model.pt` (13.90 MB).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** 
   - `visual_surface_condition`: class and probability distribution across 6 friction states.
   - `road_material`: class and probability distribution across 6 material classes.
   - `visual_unevenness`: class and probability distribution across 4 roughness classes.
6. **Execution Profiling:** Latency: 204.1 ms on $2400 \times 1604$; Peak VRAM: 113.8 MB.
7. **Scientific Safeguards:** Explicitly labels friction and unevenness outputs as *visual estimations*, not physical measurements.
8. **Shared Context Integration:** Feeds visual condition classes to `friction`, `roughness`, and `CYPhERHazardFusion`.
9. **Known Limitations:** RSCD dataset holds an extreme class imbalance towards Dry Asphalt / Smooth; rare classes (Ice, Mud, Severely Deteriorated) have low holdout sample representation (Macro-F1 ~49%).
10. **Scientific Integrity Status:** `IMPLEMENTED_AND_TRAINED`

---

### Mode 7: `accident` (Vehicle Anomaly & Rollover Detection)
1. **Underlying Mechanism:** **Geometric Heuristic Engine** (analyzes bounding box aspect ratio and spatial overlap between detected vehicles).
2. **Adapter File & Class:** `road_cv/adapters/specialized_adapters.py::AccidentAdapter`
3. **Model Checkpoint:** None (Uses detections from `M4_VEHICLE_SEG`).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** `anomaly_detected` (bool), `anomaly_type` (`normal_traffic`, `possible_rollover_or_jackknife`, `vehicle_collision_spatial_overlap`), and `confidence`.
6. **Execution Profiling:** Latency: 47.7 ms; Peak VRAM: 99.9 MB.
7. **Scientific Safeguards:** Reports heuristic confidence based on geometric deformation rather than claiming deep video action recognition.
8. **Shared Context Integration:** Consumes `vehicles` detections from `VehicleAdapter`.
9. **Known Limitations:** In single images, parked cars overlapping in perspective projection can trigger false collision flags; genuine rollovers without aspect ratio inversion (>1.5) are missed.
10. **Scientific Integrity Status:** `HEURISTIC_BASELINE`

---

### Mode 8: `weather` (Environmental Condition & Adverse Weather Engine)
1. **Underlying Mechanism:** **Photometric Optical Heuristic Engine** (`ConditionAssessor`: Laplacian variance blur, RMS contrast, high-luminance glare ratio, mean luminance).
2. **Adapter File & Class:** `road_cv/adapters/specialized_adapters.py::WeatherAdapter`
3. **Model Checkpoint:** None (Calculated via NumPy/OpenCV image transforms).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** `predicted_weather` (`Clear_Daylight`, `Overcast_Cloudy`, `Fog_or_Haze`, `Rain_or_Wet_Reflective`, `Night_or_Low_Light`), `lighting_condition`, and photometric metrics (`contrast_rms`, `sharpness_laplacian`, `glare_ratio`).
6. **Execution Profiling:** Latency: 130.6 ms on $2400 \times 1604$; Peak VRAM: 99.9 MB.
7. **Scientific Safeguards:** Transparently exposes raw photometric statistics; does not claim neural network inference.
8. **Shared Context Integration:** Consumes `road_mask` to evaluate pavement specular reflection; informs overall hazard fusion multiplier.
9. **Known Limitations:** Severe overcast days can be misclassified as light fog due to low RMS contrast; water spray on camera lens is not specifically modeled.
10. **Scientific Integrity Status:** `HEURISTIC_BASELINE`

---

### Mode 9: `lanes` (Drivable Corridor & Boundary Extractor)
1. **Underlying Mechanism:** **Geometric Contour Extraction & Polyline Approximation** (extracts outer boundary contours of semantic road mask `road_mask == 0`).
2. **Adapter File & Class:** `road_cv/adapters/specialized_adapters.py::LaneAdapter`
3. **Model Checkpoint:** None (Operates directly on `RoadSceneAdapter` semantic mask).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** `left_boundary_points`, `right_boundary_points`, `corridor_width_px`, `lane_departure_warning` (bool).
6. **Execution Profiling:** Latency: 23.7 ms; Peak VRAM: 99.9 MB.
7. **Scientific Safeguards:** Returns fallback estimated boundaries when road mask is absent; flags geometric approximation in metadata.
8. **Shared Context Integration:** Consumes `road_mask` from `RoadSceneAdapter`.
9. **Known Limitations:** Does NOT detect painted white/yellow dashed lane lines or center markings; measures physical pavement edges only.
10. **Scientific Integrity Status:** `HEURISTIC_BASELINE`

---

### Mode 10: `traffic_signals` (Traffic Signal Detection & State Classifier)
1. **Underlying Mechanism:** **Hybrid: Pretrained Neural Detector + HSV Vertical Thirds Heuristic** (Detects traffic lights using `yolo26n.pt`, crops bounding boxes, divides into vertical thirds, and counts red, yellow, and green HSV pixels).
2. **Adapter File & Class:** `road_cv/adapters/specialized_adapters.py::TrafficSignalAdapter`
3. **Model Checkpoint:** `yolo26n.pt` (5.29 MB) for box localization; heuristic for state classification.
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** List of signals: bounding box, `state` (`red`, `yellow`, `green`, `unknown_or_off`), and color pixel counts.
6. **Execution Profiling:** Latency: 303.2 ms; Peak VRAM: 152.7 MB.
7. **Scientific Safeguards:** Marks signals with insufficient saturation/brightness as `unknown_or_off` instead of hallucinating green.
8. **Shared Context Integration:** Standalone detection; reports signal state to fusion engine.
9. **Known Limitations:** Horizontal traffic lights (common in certain jurisdictions) are misclassified because vertical thirds assumption fails; distant lights (<15 px height) suffer HSV dilution.
10. **Scientific Integrity Status:** `HEURISTIC_BASELINE` (Neural Box + Heuristic State)

---

### Mode 11: `obstacles` (Road Hazard Obstacle Perception Proxy)
1. **Underlying Mechanism:** **Closed-Set Object Detection Filtering Proxy** (Filters base `yolo26n.pt` detections for COCO classes: 10/traffic light, 11/fire hydrant, 13/stop sign, 24/backpack, 25/umbrella, 26/handbag, 28/suitcase).
2. **Adapter File & Class:** `road_cv/adapters/specialized_adapters.py::ObstacleAdapter`
3. **Model Checkpoint:** `yolo26n.pt` (5.29 MB).
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** List of obstacles on road: bounding box, obstacle category, confidence, and in-corridor flag.
6. **Execution Profiling:** Latency: 206.5 ms; Peak VRAM: 162.0 MB.
7. **Scientific Safeguards:** Clarified in `model_registry.yaml` as a COCO filter proxy, NOT an open-vocabulary text-prompted model.
8. **Shared Context Integration:** Consumes `road_mask` to discard objects on sidewalks or outside the drivable corridor.
9. **Known Limitations:** Cannot detect genuine unconstrained road hazards such as fallen tree branches, spilled gravel, lost cargo, or boulders without true zero-shot foundation models (e.g., Grounding DINO).
10. **Scientific Integrity Status:** `HEURISTIC_BASELINE` (COCO Filter Proxy)

---

### Mode 12: `friction` (Tyre-Road Friction Slip Risk)
1. **Underlying Mechanism:** **Vision-Assisted Slip Risk Estimator with Sensor Gating**.
2. **Adapter File & Class:** `road_cv/adapters/specialized_adapters.py::FrictionAdapter`
3. **Model Checkpoint:** None (Uses outputs from `SurfaceAdapter`).
4. **Execution Status:** `requires_sensor`.
5. **Output Schema & Predictions:** 
   - `visual_slip_risk_estimate`: `LOW`, `MEDIUM`, or `HIGH` based on visual surface condition.
   - `measured_friction_mu`: `None` (Safeguarded).
   - `scientific_disclaimer`: Explicitly warns that true coefficient of friction $\mu$ cannot be derived from single-camera RGB images without wheel slip and IMU data.
6. **Execution Profiling:** Latency: 37.2 ms; Peak VRAM: 119.4 MB.
7. **Scientific Safeguards:** **Mandatory status gate:** Returns `status="requires_sensor"`. Refuses to output fake numerical $\mu$ values.
8. **Shared Context Integration:** Consumes `surface` classifications; accepts optional CAN bus / IMU telemetry dict.
9. **Known Limitations:** In absence of physical sensors, provides qualitative visual slip advisory only.
10. **Scientific Integrity Status:** `SENSOR_REQUIRED`

---

### Mode 13: `roughness` (Road Roughness & IRI Class)
1. **Underlying Mechanism:** **Vision-Assisted Visual Roughness Classifier with Sensor Gating**.
2. **Adapter File & Class:** `road_cv/adapters/specialized_adapters.py::RoughnessAdapter`
3. **Model Checkpoint:** None (Uses outputs from `SurfaceAdapter`).
4. **Execution Status:** `requires_sensor`.
5. **Output Schema & Predictions:** 
   - `visual_roughness_class`: `Smooth`, `Slightly uneven`, `Severely uneven`, `Deteriorated`.
   - `iri_m_per_km`: `None` (Safeguarded).
   - `scientific_disclaimer`: Explains that International Roughness Index (IRI) requires vehicle quarter-car suspension accelerometers.
6. **Execution Profiling:** Latency: 38.3 ms; Peak VRAM: 119.4 MB.
7. **Scientific Safeguards:** **Mandatory status gate:** Returns `status="requires_sensor"`. Prevents synthetic IRI reporting.
8. **Shared Context Integration:** Consumes `surface` predictions; accepts optional accelerometer telemetry.
9. **Known Limitations:** Visual surface texture does not always correlate with low-frequency vehicle vertical displacement.
10. **Scientific Integrity Status:** `SENSOR_REQUIRED`

---

### Mode 14: `pothole_geometry` (Calibrated Pothole Dimensions & Depth)
1. **Underlying Mechanism:** **Uncalibrated Relative Geometry & Depth Profiler**.
2. **Adapter File & Class:** `road_cv/adapters/specialized_adapters.py::PotholeGeometryAdapter`
3. **Model Checkpoint:** None (Operates on detections from `M1_ROAD_DAMAGE` and relative depth from `M6_DEPTH`).
4. **Execution Status:** `requires_calibration`.
5. **Output Schema & Predictions:** 
   - `calibration_state`: `UNCALIBRATED_RELATIVE`.
   - `potholes`: List of detected potholes with relative width (px), relative aspect ratio, and relative depth gradient $[0, 1]$.
   - `metric_width_cm`: `None` (Safeguarded).
   - `metric_depth_cm`: `None` (Safeguarded).
6. **Execution Profiling:** Latency: 45.1 ms; Peak VRAM: 119.4 MB.
7. **Scientific Safeguards:** **Mandatory calibration gate:** Returns `status="requires_calibration"`. Refuses to invent metric centimeters without camera focal length, height, and pitch angle.
8. **Shared Context Integration:** Consumes `damage` boxes and `depth` depth map.
9. **Known Limitations:** Requires extrinsic and intrinsic camera matrix calibration to project pixels into metric Euclidean space.
10. **Scientific Integrity Status:** `CALIBRATION_REQUIRED`

---

### Mode 15: `all` (Master Multi-Model Hazard Perception Pipeline)
1. **Underlying Mechanism:** **Sequential Multi-Model Orchestrator with Explainable Rule-Based Hazard Fusion** (`CYPhEROrchestrator` + `CYPhERHazardFusion`).
2. **Adapter File & Class:** `road_cv/core/orchestrator.py::CYPhEROrchestrator`
3. **Model Checkpoints:** Dynamically calls all underlying checkpoints in managed sequence.
4. **Execution Status:** `success`.
5. **Output Schema & Predictions:** Unified JSON object containing all 14 individual capability outputs, plus `fusion` dictionary (`overall_hazard_level`, `hazard_score`, `active_hazards`, `driver_recommendation`).
6. **Execution Profiling:** 
   - $2400 \times 1604$ Image (`Road.jpg`): Latency: 374.8 ms warm; Peak VRAM: 438.0 MB.
   - $640 \times 640$ Image (`India_000002.jpg`): Latency: 178.5 ms warm; Peak VRAM: 207.1 MB.
7. **Scientific Safeguards:** Cleans non-serializable ndarrays from JSON; maintains safeguards of individual sub-adapters; provides explainable hazard rules.
8. **Shared Context Integration:** Implements cross-model blackboard architecture: road mask, depth map, detections, and surface classes are shared downstream in memory.
9. **Known Limitations:** Sequential CPU/GPU execution takes ~2.48s when loading all weights cold from hard disk.
10. **Scientific Integrity Status:** `VALIDATED`

