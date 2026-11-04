# CYPhER V2.4 — OPEN-VOCABULARY OBSTACLE PERCEPTION EVALUATION
**Zero-Shot Hazard Grounding, False Alarm Suppression, and Generalization Limits**

---

## 1. System Architecture & Zero-Shot Vision-Language Grounding

Traditional autonomous driving obstacle detectors fail catastrophically when encountering unmodeled objects (e.g. fallen trees, spilled construction gravel, detached truck tires, discarded cargo).

CYPhER V2.4 integrates `YOLO-World` (`yolov8s-world.pt`, 25.91 MB) for real-time open-vocabulary zero-shot grounding:
- **Vision Backbone:** Efficient visual feature pyramid.
- **Language Encoder:** CLIP text encoder extracting semantic embeddings for custom hazard prompts.
- **Real-Time Cross-Modality Fusion:** Dynamic projection matrix binding text queries directly to multi-scale visual features.
- **Prompt Library (Indian Road Hazards):**
  1. `fallen tree`
  2. `large rock`
  3. `boulder on road`
  4. `road debris`
  5. `broken concrete`
  6. `spilled cargo`
  7. `tyre on road`
  8. `construction material`
  9. `traffic barrier`
  10. `road obstruction`
- **Inference Latency:** **18.5 ms** per frame on RTX 4050 GPU ($> 54 \text{ FPS}$).
- **VRAM Consumption:** **112.4 MB**.
- **Taxonomy Status:** **`PRETRAINED_NOT_VALIDATED`**.

---

## 2. Clean-Road False Alarm Verification

A primary vulnerability of open-vocabulary detectors in safety-critical robotics is the tendency to hallucinate nonexistent hazards on complex pavement textures (e.g., expansion joints, shadows, dark patches).

We conducted an evaluation across 50 held-out Indian highway and rural scenes (`data/processed/rdd2022_india/images/val/`):
- **Operational Confidence Threshold:** $\tau_{\text{conf}} \ge 0.30$.
- **Observed False Alarms on Clean Road Surfaces:** **0 instances** across 50 frames.
- **Result:** The vision-language cross-attention weights in YOLO-World exhibit high specificity, effectively rejecting normal pavement aggregate, oil spots, and road striping.

---

## 3. Scientific Verification Boundary: The Hazard Recall Gap

> [!WARNING]
> **Scientific Integrity Requirement:**
> Demonstrating a 0% false alarm rate on clean road surfaces **does NOT prove hazard recall**. 
> Standard dashcam datasets (like RDD2022) are captured on active traffic lanes and rarely depict catastrophic roadblocks like fallen trees or spilled cargo.
> 
> Validating true recall requires evaluation against specialized anomaly benchmarks (such as the *Lost and Found Benchmark* or *RoadObstacle Dataset*).
> Consequently, the open-vocabulary obstacle detector is categorized as **`PRETRAINED_NOT_VALIDATED`** in the CYPhER V2.4 Model Registry.

