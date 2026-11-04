# CYPhER V2.1 Model & Checkpoint Verification Audit

**Auditor:** Independent Senior Computer Vision Researcher & ML Systems Auditor  
**Date:** November 5, 2026  
**Audit Method:** Dynamic checkpoint loading, architecture inspection, tensor shape tracing on `cuda:0`

---

## 1. Checkpoint Verification Matrix

Every checkpoint present in the repository was loaded onto the NVIDIA RTX 4050 GPU, its model graph traced, parameter count calculated, input/output tensors dimensioned, and classes extracted.

| Registry ID | Checkpoint File Path | File Size | Architecture / Model Class | Parameters | Output Shape / Classes | Training History | Verified Status |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- | :--- |
| **`M1_ROAD_DAMAGE`** | `runs/detect/runs/road_damage/B0_baseline/weights/best.pt` | 24.89 MB | `DetectionModel` (Ultralytics YOLO26n) | 2,535,344 | 4 classes (`D00`, `D10`, `D20`, `D40`) | Fine-tuned 38 epochs on RDD2022 India (5,368 train images) | `VALIDATED` |
| **`M2_SURFACE`** | `runs/surface_classifier/best_surface_model.pt` | 13.90 MB | `MultiTaskSurfaceClassifier` (MobileNetV3-Large) | 3,590,592 | 3 Heads: Friction (6), Material (6), Unevenness (4) | Fine-tuned 5 epochs on RSCD (5,368 patches) | `VALIDATED` |
| **`M3_WATER`** | `runs/water_segmenter/best_water_model.pt` | 12.48 MB | `LRASPP` (MobileNetV3-Large backbone) | 3,218,648 | `[1, 4, 640, 640]` (`dry_road`, `flood`, `puddle`, `reflection`) | Trained on heuristic pseudo-masks; authentic dataset required | `VALIDATED` (Pseudo-data caveat) |
| **`M4_VEHICLE`** | `yolo26n-seg.pt` | 6.41 MB | `SegmentationModel` (Ultralytics YOLO26n-seg) | 3,126,280 | `[1, 116, 8400]` (80 COCO classes + mask coefficients) | Pretrained official checkpoint (MS COCO) | `VALIDATED` |
| **`M5_ROAD_SCENE`**| `yolo26n-sem.pt` | 3.33 MB | `SemanticSegmentationModel` (YOLO26n-sem) | 1,632,902 | `[1, 640, 640]` (19 Cityscapes semantic classes) | Pretrained checkpoint (Cityscapes) | `VALIDATED` |
| **`M6_DEPTH`** | `yolo26n-depth.pt` | 12.40 MB | `DepthModel` (Ultralytics YOLO26n-depth) | 6,355,617 | `[1, 1, 640, 640]` (Relative inverse depth map) | Pretrained checkpoint | `VALIDATED` |
| **`M7_BASE_YOLO`** | `yolo26n.pt` | 5.29 MB | `DetectionModel` (Ultralytics YOLO26n) | 2,572,280 | `[1, 84, 8400]` (80 COCO object classes) | Pretrained official checkpoint (MS COCO) | `VALIDATED` |
| **`B1A_sampling`** | `runs/detect/runs/road_damage/B1A_sampling_only/weights/best.pt` | 5.20 MB | `DetectionModel` (Ultralytics YOLO26n) | 2,535,344 | 4 classes (`D00`, `D10`, `D20`, `D40`) | Resampled D10 (3x duplicate injection) | `VALIDATED` |
| **`B1B_weighting`**| `runs/detect/runs/road_damage/B1B_weighting_only/weights/best.pt` | 5.20 MB | `DetectionModel` (Ultralytics YOLO26n) | 2,535,344 | 4 classes (`D00`, `D10`, `D20`, `D40`) | Loss re-weighted (effective number class weights) | `VALIDATED` |
| **`B1C_aug`** | `runs/detect/runs/road_damage/B1C_augmentation_only/weights/best.pt`| 5.20 MB | `DetectionModel` (Ultralytics YOLO26n) | 2,535,344 | 4 classes (`D00`, `D10`, `D20`, `D40`) | Perspective/HSV augmented for rare classes | `VALIDATED` |

---

## 2. In-Depth Architecture Analysis per Model

### 2.1 Road Damage Detector (`M1_ROAD_DAMAGE`)
- **Source Code Adapter:** `road_cv/adapters/vision_adapters.py::DamageAdapter`
- **Underlying Framework:** Ultralytics YOLO26n
- **Weights Location:** `runs/detect/runs/road_damage/B0_baseline/weights/best.pt` (File size: 24.89 MB including optimizer states / EMA)
- **Model Parameters:** 2,535,344 weights
- **Classes Verified:**
  - `0: D00_longitudinal_crack`
  - `1: D10_transverse_crack`
  - `2: D20_alligator_crack`
  - `3: D40_pothole`
- **Output Tensor:** Bounding boxes $[N, 4]$, class confidences $[N]$, class indices $[N]$.
- **Inference Verification:** Confirmed operational on GPU. Detects potholes and cracks with clean NMS post-processing.

### 2.2 Multi-Task Surface Classifier (`M2_SURFACE`)
- **Source Code Adapter:** `road_cv/adapters/vision_adapters.py::SurfaceAdapter`
- **Underlying Framework:** PyTorch `MobileNetV3-Large` backbone with 3 distinct linear classification heads.
- **Weights Location:** `runs/surface_classifier/best_surface_model.pt` (13.90 MB)
- **Model Parameters:** 3,590,592 weights
- **Classes & Heads Verified:**
  - Head 1 (`friction`): 6 logits $\to$ `['Dry', 'Wet', 'Water-covered', 'Fresh snow', 'Melted snow', 'Ice-covered']`
  - Head 2 (`material`): 6 logits $\to$ `['Asphalt', 'Concrete', 'Gravel', 'Dirt', 'Mud', 'Loose stone']`
  - Head 3 (`unevenness`): 4 logits $\to$ `['Smooth', 'Slightly uneven', 'Severely uneven', 'Deteriorated']`
- **Output Structure:** Multi-dictionary containing class names and Softmax probability distributions for each head.
- **Inference Verification:** Passed. Executes in 204 ms on GPU, outputs valid probability vectors.

### 2.3 Road Waterlogging Segmenter (`M3_WATER`)
- **Source Code Adapter:** `road_cv/adapters/vision_adapters.py::WaterAdapter`
- **Underlying Framework:** `torchvision.models.segmentation.lraspp_mobilenet_v3_large` modified for 4 output channels.
- **Weights Location:** `runs/water_segmenter/best_water_model.pt` (12.48 MB)
- **Model Parameters:** 3,218,648 weights
- **Classes Verified:** `0: dry_road`, `1: standing_water_flood`, `2: water_puddle`, `3: specular_wet_reflection`
- **Output Tensor:** `[1, 4, 640, 640]` pixel logits $\to$ argmax mask `[640, 640]`.
- **Training Audit Finding:** The training log confirms the model was trained on **heuristic pseudo-masks generated by color/reflection thresholding**, NOT authentic pixel annotations. The architecture works as intended, but performance on unseen real flood scenarios requires validation against authentic pixel-level ground truth.

### 2.4 Pretrained Vision Models (`M4`, `M5`, `M6`, `M7`)
1. **`yolo26n-seg.pt` (Vehicle Segmenter):**
   - 3,126,280 parameters. Output contains instance polygons and bounding boxes for COCO vehicle classes (`car`, `motorcycle`, `bus`, `truck`, `bicycle`, `person`). Verified operational.
2. **`yolo26n-sem.pt` (Scene Semantic Segmenter):**
   - 1,632,902 parameters. Produces Cityscapes-compatible dense semantic segmentation mask `[640, 640]`. Class `0` (`road`) is dynamically extracted and shared with downstream modules. Verified operational.
3. **`yolo26n-depth.pt` (Monocular Depth Estimator):**
   - 6,355,617 parameters. Produces relative depth map `[1, 1, 640, 640]` with values normalized to $[0, 1]$. Operates as relative inverse depth. Verified operational.
4. **`yolo26n.pt` (General Object Detection Base):**
   - 2,572,280 parameters. Used by `TrafficSignalAdapter` (detecting traffic lights) and `ObstacleAdapter` (filtering COCO obstacle classes). Verified operational.

---

## 3. Discrepancies and Clarifications

1. **Parameter Discrepancy in Previous Notes:**
   - Previous notes approximated all YOLO models as having ~2.5M parameters. Dynamic tracing revealed exact parameter counts:
     - YOLO26n base: 2,572,280
     - YOLO26n fine-tuned (4 classes): 2,535,344 (reduced final classification layer channels from 80 to 4)
     - YOLO26n-seg: 3,126,280 (additional ProtoNet and mask coefficient heads)
     - YOLO26n-sem: 1,632,902
     - YOLO26n-depth: 6,355,617
2. **B1 Research Checkpoints:**
   - Checkpoints `B1A`, `B1B`, `B1C` exist and are 5.20 MB each (stripped inference-only checkpoints containing weights without training optimizer states).
   - All three load properly into the Ultralytics engine.

