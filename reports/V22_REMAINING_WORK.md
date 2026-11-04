# CYPhER V2.2 Remaining Work & V3.0 Production Roadmap

**Author:** CYPhER Research & Engineering Team  
**Date:** November 5, 2026  
**Document Purpose:** Clear technical roadmap identifying unresolved items and prerequisites for V3.0

---

## 1. Transparent Gap Analysis

| Perception Capability | Current Status in V2.2 | Operational Reality | Blocker / Next Prerequisite |
| :--- | :--- | :--- | :--- |
| **Authentic Waterlogging** | `DATASET_REQUIRED` | Ingestion pipeline complete; pseudo-masks quarantined. Model awaits authentic pixel annotations. | Download & stage R²S100K dataset masks into `data/processed/water_authentic/`. |
| **D10 Crack Full Retraining**| `VALIDATED_FOUNDATION` | Root cause diagnosed; B0 preserved; road-corridor tiling recovers thin cracks. Full multi-epoch retraining pending. | Execute 50-epoch training with combined focal loss and mosaic-disabled fine-tuning. |
| **Accident Action Recognition**| `DATASET_REQUIRED` | 16-frame 3D-CNN architecture built; trajectory heuristics active. Deep temporal weights untrained. | Acquire annotated dashcam collision video benchmark (e.g., DAD / DoTA) and train. |
| **Metric Pothole Depth** | `CALIBRATION_REQUIRED` | Relative depth profiling active; centimeter claims strictly gated. | Implement camera intrinsic/extrinsic vanishing-point calibration interface. |
| **Physical Friction & Roughness**| `SENSOR_REQUIRED` | Visual slip risk and visual roughness active; physical $\mu$ and IRI gated. | Integrate CAN bus wheel-speed and quarter-car accelerometer inputs. |

---

## 2. Prioritized Roadmap to CYPhER V3.0

```
Phase 1: Ingest R²S100K authentic water masks and execute 15-epoch LRASPP retraining.
Phase 2: Train 16-frame 3D-CNN accident model on DoTA dashcam crash benchmark.
Phase 3: Implement vanishing-point automated camera pitch & height calibration for metric pothole measurements.
Phase 4: Export unified TensorRT / ONNX engines for edge deployment at 60+ FPS on embedded hardware.
```
