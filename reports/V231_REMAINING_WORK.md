# CYPhER V2.3.1 — REMAINING ENGINEERING WORK & SYSTEM ROADMAP

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Research Software Architect  
**Branch:** `feature/cypher-v2.3.1-accuracy`  

---

## 1. Concrete Engineering Backlog by Workstream

### Workstream A: Road Damage & D10 Thin-Crack Recovery
- **Task A.1:** Implement verified copy-paste crack augmentation in `road_cv/damage/copy_paste_augmentation.py`, extracting 50 authentic D10 cracks and pasting them into non-cracked highway road patches with Poisson blending.
- **Task A.2:** Retrain YOLO26n from B0 weights applying Effective-Number loss scaling ($w_{\text{D10}} = 7.14$) and mosaic augmentation, evaluating on the 1,172 validation holdout set to confirm D10 recall $> 40\%$ without degrading D20/D40.
- **Task A.3:** Integrate adaptive SAHI tiling for small-defect recovery ($<32^2 \text{ px}^2$).

### Workstream B: Surface Condition & Friction
- **Task B.1:** Promote `runs/surface_classifier_v231/exp4_effective_number/best_model.pt` as default checkpoint in `configs/model_registry.yaml`.
- **Task B.2:** Ingest multi-material road dataset (concrete, unpaved gravel, dirt) to expand the material head beyond 100% asphalt.
- **Task B.3:** Validate visual slip-risk against physical vehicle telemetry on test track data.

### Workstream C: Authentic Waterlogging Segmentation
- **Task C.1:** Ingest 1,500 authentic polygon masks from RTK-Water / ACDC into `data/processed/water_authentic/`.
- **Task C.2:** Train LRASPP MobileNetV3-Large using `road_cv/water/authentic_water_dataset.py` with Cross-Entropy + 1.5 $\times$ Soft-Dice loss.
- **Task C.3:** Evaluate against the 100-image glare rejection benchmark (sunlit asphalt, shadows, standing water).

### Workstream D: Traffic Participants & Indian Road Domain
- **Task D.1:** Download the India Driving Dataset (IDD) mini-benchmark containing labeled auto-rickshaws, e-rickshaws, and tractors.
- **Task D.2:** Fine-tune `yolo26n-seg` on IDD to natively predict 3-wheeler classes.
- **Task D.3:** Benchmark ByteTrack multi-camera tracking using MOTA, HOTA, and IDF1 on annotated Indian video sequences.

### Workstream E: Accident Intelligence 3D-CNN
- **Task E.1:** Download DADA-2000 / DoTA crash video sequences (16-frame temporal clips).
- **Task E.2:** Train `SpatioTemporalAccident3DCNN` on crash clips versus normal driving.
- **Task E.3:** Evaluate temporal collision recognition AUC and false alarm rate per hour.

---

## 2. Updated Single Source of Truth Alignment

The model configuration files `configs/model_registry.yaml` and `configs/cypher.yaml` are synchronized to version `2.3.1`, embedding the new surface classifier metrics, D10 findings, and strict taxonomy statuses.
