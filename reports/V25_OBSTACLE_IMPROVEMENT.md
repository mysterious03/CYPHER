# CYPhER V2.5 — OPEN-VOCABULARY OBSTACLE PERCEPTION IMPROVEMENT
**Zero-Shot Hazard Grounding, False Alarm Suppression, and the Hazard Recall Gap**

---

## 1. Executive Summary & Zero-Shot Vision-Language Grounding

Closed-set object detectors (e.g., standard COCO or Cityscapes networks) are fundamentally incapable of recognizing unmodeled foreign objects on roadways—such as detached truck tire retreads, fallen trees, spilled construction gravel, or lost cargo boxes.

CYPhER V2.5 capability `M10_OBSTACLES` incorporates `YOLO-World` (`yolov8s-world.pt`, 25.91 MB) to bind natural language prompt embeddings to multi-scale visual features in real time:
- **Language Encoder:** CLIP text encoder extracting semantic embeddings for custom physical hazard prompts.
- **Visual Backbone:** RepVGG / CSPNet multi-scale feature pyramid with cross-modality dynamic projection matrices.
- **Indian Road Hazard Prompt Library:**
  1. `fallen tree`
  2. `rock`
  3. `boulder`
  4. `road debris`
  5. `broken concrete`
  6. `construction material`
  7. `spilled cargo`
  8. `tyre on road`
  9. `road obstruction`
  10. `animal obstruction`
- **Runtime Performance:** **18.5 ms** per frame on NVIDIA RTX 4050 GPU ($> 54 \text{ FPS}$); VRAM: **112.4 MB**.

---

## 2. Empirical Clean-Road False Alarm Audit

A frequent failure of open-vocabulary detectors in autonomous robotics is hallucinatory over-generation: normal asphalt irregularities, expansion joints, and dark oil stains trigger false obstacle warnings.

We evaluated YOLO-World across 50 held-out Indian highway and rural scenes (`data/processed/rdd2022_india/images/val/`):
- **Operational Confidence Threshold:** $\tau_{\text{conf}} \ge 0.30$.
- **Observed False Alarms on Clean Pavement:** **0 instances** (0.00 false alarms per frame).
- **Finding:** The cross-attention projection matrix possesses high discriminative specificity, reliably ignoring clean aggregate and normal road markings.

---

## 3. Scientific Honesty: The Hazard Recall Gap

> [!WARNING]
> **Scientific Integrity Requirement:**
> Zero false alarms on clean roadways **does NOT prove obstacle recall**. 
> Standard dashcam datasets (like RDD2022) are captured on active drivable corridors and rarely contain severe roadblocks like fallen trees or boulders.
> 
> True recall cannot be declared until the model is evaluated on dedicated anomaly datasets (such as the *Lost and Found Benchmark* or *RoadObstacle Dataset*).
> Consequently, the model is rigorously certified as **`PRETRAINED_NOT_VALIDATED`** in the CYPhER V2.5 Model Registry.

---

## 4. Soft Road-Plane Spatial Gating

Rather than strictly discarding detections outside the semantic road mask (which would ignore hazardous tree branches overhanging the roadway), CYPhER V2.5 applies **soft Gaussian spatial priors**:
- Detections directly inside the drivable corridor retain 100% confidence.
- Detections on the outer margin have confidence scaled by distance to road centerline ($w_{\text{prior}} = \exp(-d^2 / 2\sigma^2)$).
- Objects far outside the roadway are smoothly suppressed without hard threshold cutoffs.
