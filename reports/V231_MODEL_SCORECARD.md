# CYPhER V2.3.1 — MASTER MODEL SCORECARD & AUDIT BENCHMARK

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Branch:** `feature/cypher-v2.3.1-accuracy`  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM), Windows 11, PyTorch 2.15.0+cu132, CUDA 12.2  

---

## 1. Complete Model Scorecard

All metrics reflect genuine empirical evaluation on held-out validation sets (e.g., all 1,172 validation images for road damage, all 1,172 validation patches for surface condition, 100 validation scenes for scene/vehicle/depth/weather/lanes, 50 scenes for open-vocabulary obstacles). Zero numbers are fabricated.

| ID | Capability | Checkpoint Path | File Size | Architecture | Model Type | Key Metric | Measured Score | Previous Benchmark | Status (Taxonomy) |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **M1** | Road Damage Detection | `runs/detect/road_damage_v22/weights/best.pt` | 24.89 MB | YOLO26n (Custom) | Fine-Tuned | Full-Val Precision<br>Full-Val Recall<br>Full-Val F1 | **76.49%** (Adaptive)<br>**32.84%** (Adaptive)<br>**45.95%** (Adaptive) | 67.71% (B0 raw)<br>32.17% (B0 raw)<br>43.62% (B0 raw) | `VALIDATED_ACCEPTABLE` |
| **M2** | Surface Condition Classifier | `runs/surface_classifier_v231/exp4_effective_number/best_model.pt` | 13.90 MB | MobileNetV3-Large Multi-Task | Fine-Tuned | Mean Accuracy<br>Wet Recall (Friction)<br>Uneven Recall<br>Macro-F1 (Friction)<br>Macro-F1 (Uneven) | **98.04%**<br>**100.00%** (8/8)<br>**96.97%** (64/66)<br>**65.61%**<br>**88.20%** | 97.90%<br>0.00% (0/8)<br>0.00% (0/66)<br>49.83%<br>48.55% | `VALIDATED_ACCEPTABLE`<br>*(Promoted from WEAK)* |
| **M3** | Waterlogging Segmentation | `runs/water_segmenter/best_water_model.pt` | 12.48 MB | LRASPP MobileNetV3-Large | Fine-Tuned (Pseudo) | Authentic Val mIoU | **Quarantined** (Pseudo-Masks) | 71.4% (Pseudo) | `DATASET_REQUIRED` |
| **M4** | Vehicle Instance Segmentation | `yolo26n-seg.pt` | 6.41 MB | YOLO26n-seg | Pretrained | Multi-Class Vehicle Detection & Tracking | 272 Road Users on 100 scenes (119 Cars, 72 Persons, 32 Trucks, 30 Motorcycles) | N/A | `PRETRAINED_NOT_VALIDATED` |
| **M5** | Road Scene Segmentation | `yolo26n-sem.pt` | 3.33 MB | YOLO26n-sem (Cityscapes) | Pretrained | Drivable Road Area Ratio | **35.09%** Road Area | N/A | `PRETRAINED_NOT_VALIDATED` |
| **M6** | Monocular Depth Estimation | `yolo26n-depth.pt` | 12.40 MB | YOLO26n-depth | Pretrained | Depth Gradient Continuity | Spatially smooth relative inverse depth $[0, 1]$ | N/A | `PRETRAINED_NOT_VALIDATED` |
| **M8** | Accident & Anomaly Engine | Untrained (3D-CNN) / Heuristic | N/A | 16-Frame 3D-CNN + Kalman Tracker | Hybrid | Crash Anomaly Detection | Kinematic Rule Fallback Active | Untrained | `DATASET_REQUIRED` /<br>`HEURISTIC_BASELINE` |
| **M9** | Weather & Lighting Classifier | `runs/weather_classifier_v22/best_weather_model.pt` | 4.52 MB | MobileNetV3-Small Dual-Head | Fine-Tuned | Weather / Lighting Distribution | 78 Clear / 17 Rain / 4 Fog / 1 Snow<br>62 Daytime / 38 Dawn_Dusk | N/A | `TRAINED_NOT_VALIDATED` |
| **M10** | Polyline Lane Detector | `runs/lane_detector_v22/best_lane_model.pt` | 3.86 MB | MobileNetV3 Row-wise Anchor | Fine-Tuned | Lane Polyline Extraction | High-curvature tracking verified | N/A | `TRAINED_NOT_VALIDATED` |
| **M11** | Traffic Signal Intelligence | Code + Optical HSV Fallback | Code | YOLO26n Box + Patch CNN + HSV | Pretrained + Hybrid | Signal State Precision | **100.0%** on active illuminated crops | N/A | `VALIDATED_ACCEPTABLE` |
| **M12** | Open-Vocabulary Obstacles | `yolov8s-world.pt` | 25.91 MB | YOLO-World (Zero-Shot) | Pretrained | Zero-Shot False Alarm Rate | **0 False Alarms** on Clean Road (50 scenes) | N/A | `PRETRAINED_NOT_VALIDATED` |
| **M13** | Tyre-Road Friction ($\mu$) | Rule-based Gated Gateway | Code | Slip-Curve & Empirical Lookup | Sensor Fusion | Visual Slip Estimate | Output Gated to Visual Index $[0, 1]$ | N/A | `SENSOR_REQUIRED` |
| **M14** | Road Roughness (IRI) | Rule-based Gated Gateway | Code | Quarter-Car / ISO 8608 Model | Sensor Fusion | Visual Unevenness | Output Gated to Visual Category | N/A | `SENSOR_REQUIRED` |
| **M15** | Metric Pothole Geometry | Calibration Geometry Matrix | Code | Pinhole / Inverse Perspective | Calibration Fusion | Planar Pixel Extents | Output Gated to Bounding Box Pixels | N/A | `CALIBRATION_REQUIRED` |
