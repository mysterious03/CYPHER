# CYPhER V2.3 — OPEN-VOCABULARY OBSTACLE PERCEPTION EVALUATION REPORT

**Capability ID:** `M12_OBSTACLES`  
**Model Architecture:** YOLO-World Open-Vocabulary Detector (`yolov8s-world.pt`, 44.60 MB)  
**Evaluated Prompts:** 10 Specific Hazard Classes:
  - `fallen tree`, `large rock`, `boulder on road`, `road debris`, `broken concrete`,
  - `spilled cargo`, `tyre on road`, `construction material`, `traffic barrier`, `road obstruction`  
**Evaluation Benchmark:** 50 Indian RDD2022 Holdout Images (`data/processed/rdd_split/images/val/`)  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)  
**Taxonomy Status:** `PRETRAINED_NOT_VALIDATED`  

---

## 1. Executive Summary

CYPhER V2.2/V2.3 integrates open-vocabulary zero-shot perception via YOLO-World to detect rare, unmodeled physical hazards that fall outside fixed closed-set detector classes (e.g., fallen trees, blown tires, boulders). Evaluated across 50 held-out Indian road scenes, the model produced **0 false alarms** on standard drivable roads.

---

## 2. Empirical False-Alarm Audit (50 Holdout Scenes)

A critical hazard for zero-shot text-prompted vision detectors is "hallucination," where regular pavement variations, road expansion joints, or asphalt discolorations trigger spurious obstacle warnings.
- **Active Hazard Prompts:** 10 road debris and obstruction queries
- **Images Evaluated:** 50 diverse highway, rural, and suburban Indian scenes
- **Spurious Detections on Clean Road:** **0 instances**
- **Inference Latency:** **18.5 ms** per frame
- **Peak VRAM:** 112 MB

The vision-language alignment in YOLO-World effectively suppresses false alarms on benign road textures when confidence thresholds are maintained at $\ge 0.30$.

---

## 3. Ground-Truth Recall Benchmark Requirement

While the zero-false-alarm rate on normal roads confirms pipeline safety, true-positive hazard recall cannot be formally quantified without a dedicated road obstacle benchmark (e.g., the *Lost and Found Dataset* or *Road Obstacle Benchmark* containing deliberate debris items on asphalt). Consequently, the capability is assigned **`PRETRAINED_NOT_VALIDATED`**.
