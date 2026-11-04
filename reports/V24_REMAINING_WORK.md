# CYPhER V2.4 — REMAINING WORK & FUTURE ROADMAP
**Actionable Research Steps for Subsequent Iterations**

---

## 1. High-Priority Action Items

### 1.1 Ingestion of Authentic Waterlogging Ground Truth (Transition M3 from `DATASET_REQUIRED`)
- **Dataset Intake:** Download the RTK-Water dataset and ACDC (Rain split) polygon annotations.
- **Preprocessing:** Standardize to single-channel 8-bit PNG masks ($0$: background, $1$: dry road, $2$: standing water).
- **Training:** Execute the compound Focal-Dice boundary-loss LRASPP training recipe (`reports/V24_WATER_TRAINING.md`).
- **Target Acceptance Gate:** Achieve puddle $\text{mIoU} \ge 65.0\%$ and glare false alarm rate $< 0.1\%$ of drivable road pixels.

### 1.2 Fine-Tuning Vehicle Segmenter on IDD (Transition M4 from `PRETRAINED_NOT_VALIDATED`)
- **Dataset Intake:** Acquire India Driving Dataset (IDD) Level-3 bounding boxes and polygonal masks.
- **Class Ontology:** Integrate custom heads for `auto_rickshaw`, `e-rickshaw`, `tractor`, and `animal`.
- **Optimization:** Fine-tune from `yolo26n-seg.pt` using Class-Balanced Effective-Number weighting ($\beta=0.999$).
- **Target Acceptance Gate:** Auto-rickshaw recall $\ge 75.0\%$, tractor recall $\ge 70.0\%$.

### 1.3 3D Spatiotemporal Graph Convolutional Network for Accidents (M11)
- **Dataset Intake:** Download DADA-2000 accident anticipation dataset and DoTA anomaly tubes.
- **Model Training:** Train an ST-GCN network on multi-frame vehicle tracklet graphs to predict collision probability 2.0+ seconds prior to impact.
- **Telemetry Fusion:** Connect CAN-bus ego-speed simulator to enable metric Time-To-Collision (TTC).

### 1.4 Camera Intrinsic / Extrinsic Calibration Tool (M6 & M12)
- **Calibration Engine:** Implement an automated vanishing-point / road-plane homography calibration tool to estimate camera focal length ($f$), mounting height ($h_{\text{cam}}$), and pitch angle ($\theta_{\text{pitch}}$).
- **Metric Scaling:** Map monocular inverse depth and 2D bounding boxes to real-world metric dimensions (converting pothole pixel areas to centimeters and depth in meters).

---

## 2. Capability Status Transition Roadmap

| Mode | Capability | Current V2.4 Status | Target Next Version Status | Required Dependency |
|---|---|---|---|---|
| **M1** | Road Damage Detection | `VALIDATED_ACCEPTABLE` | `VALIDATED_ACCEPTABLE` | Maintain continuous validation gate |
| **M2** | Surface Condition Classifier | `VALIDATED_ACCEPTABLE` | `VALIDATED_ACCEPTABLE` | Vehicle CAN bus for $\mu$ physics |
| **M3** | Waterlogging Segmenter | `DATASET_REQUIRED` | `VALIDATED_ACCEPTABLE` | Ingest RTK-Water / ACDC |
| **M4** | Vehicle Segmenter | `PRETRAINED_NOT_VALIDATED` | `VALIDATED_ACCEPTABLE` | Retrain on IDD Level-3 |
| **M5** | Road Scene Segmentation | `PRETRAINED_NOT_VALIDATED` | `VALIDATED_ACCEPTABLE` | Fine-tune on IDD Road Semantics |
| **M6** | Monocular Depth | `CALIBRATION_REQUIRED` | `VALIDATED_ACCEPTABLE` | Implement auto-homography calibrator |
| **M7** | Polyline Lane Detector | `TRAINED_NOT_VALIDATED` | `VALIDATED_ACCEPTABLE` | Benchmark against CULane |
| **M8** | Weather & Lighting | `TRAINED_NOT_VALIDATED` | `VALIDATED_ACCEPTABLE` | Benchmark against ACDC / DAWN |
| **M9** | Traffic Signal Intelligence | `VALIDATED_ACCEPTABLE` *(Optical)* | `VALIDATED_ACCEPTABLE` *(Neural)* | Train Patch CNN on LISA / Bosch |
| **M10** | Open-Vocabulary Obstacles | `PRETRAINED_NOT_VALIDATED` | `VALIDATED_ACCEPTABLE` | Benchmark on Lost & Found |
| **M11** | Accident Anticipation | `PRETRAINED_NOT_VALIDATED` | `VALIDATED_ACCEPTABLE` | Ingest DADA-2000 & train ST-GCN |
| **M12** | Pothole 3D Severity | `CALIBRATION_REQUIRED` | `VALIDATED_ACCEPTABLE` | Camera calibration matrix |
| **M13** | Friction Physics | `SENSOR_REQUIRED` | `VALIDATED_ACCEPTABLE` | Wheel speed slip sensor |
| **M14** | Roughness Index (IRI) | `SENSOR_REQUIRED` | `VALIDATED_ACCEPTABLE` | Vertical accelerometer IMU |
| **M15** | Unified Perception (`all`) | `VALIDATED_ACCEPTABLE` | `VALIDATED_ACCEPTABLE` | Continuous integration & benchmarking |

