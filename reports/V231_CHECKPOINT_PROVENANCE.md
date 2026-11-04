# CYPhER V2.3.1 — CHECKPOINT PROVENANCE & WEIGHT VERIFICATION AUDIT

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Verification Method:** Byte-level Hash (MD5), PyTorch Tensor Introspection, Disk Audit  

---

## 1. Checkpoint Provenance Matrix

| Checkpoint Path | Architecture | Size | MD5 Hash | Provenance & Origin | Status |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `runs/detect/runs/road_damage/B0_baseline/weights/best.pt` | YOLO26n | 24.89 MB | `456338c179bbf1b1681fff8a30e203f2` | Trained on RDD2022 India (38 epochs) | Preserved Canonical Baseline |
| `runs/detect/road_damage_v22/weights/best.pt` | YOLO26n | 24.89 MB | `456338c179bbf1b1681fff8a30e203f2` | Exact replica of B0 baseline | Preserved Canonical Baseline |
| `runs/surface_classifier/best_surface_model.pt` | MobileNetV3-Large | 13.90 MB | Verified PyTorch StateDict | Trained on RSCD 1,200 subset (5 epochs) | Active (Weak on Minority Classes) |
| `runs/water_segmenter/best_water_model.pt` | LRASPP MobileNetV3 | 12.48 MB | Verified PyTorch StateDict | Trained on heuristic pseudo-masks | **Quarantined** (`DATASET_REQUIRED`) |
| `yolo26n-seg.pt` | YOLO26n-seg | 6.41 MB | Verified TorchScript/PT | Official Ultralytics COCO Pretrained | Pretrained (Indian 3W Gap) |
| `yolo26n-sem.pt` | YOLO26n-sem | 3.33 MB | Verified TorchScript/PT | Official Cityscapes Pretrained | Pretrained (Valid Road Extents) |
| `yolo26n-depth.pt` | YOLO26n-depth | 12.40 MB | Verified TorchScript/PT | Monocular Depth Pretrained | Pretrained (Relative Depth Only) |
| `runs/weather_classifier_v22/best_weather_model.pt` | MobileNetV3-Small | 4.52 MB | Verified PyTorch StateDict | Fine-tuned dual-head classifier | Trained (Unvalidated on ACDC) |
| `runs/lane_detector_v22/best_lane_model.pt` | MobileNetV3 Polyline | 3.86 MB | Verified PyTorch StateDict | Fine-tuned row-wise anchor detector | Trained (Unvalidated on CULane) |
| `runs/traffic_signals_v22/best_signal_model.pt` | Patch CNN | 0.00 MB | Missing on Disk | Hybrid fallback active via HSV thirds rule | **Code + Optical Fallback Active** |
| `yolov8s-world.pt` | YOLO-World | 25.91 MB | Verified TorchScript/PT | Official YOLO-World Pretrained | Pretrained (Zero-Shot Active) |

---

## 2. Integrity Commitments

1. **Baseline Preservation:** The canonical checkpoints `runs/detect/runs/road_damage/B0_baseline/weights/best.pt` and `runs/detect/road_damage_v22/weights/best.pt` are permanently locked and write-protected. No retraining will overwrite these weights.
2. **Pseudo-Mask Quarantine:** `runs/water_segmenter/best_water_model.pt` will remain in isolation and will not be reported as a validated water segmentation model.
3. **Dedicated Directory for New Experiments:** All new model checkpoints created in CYPhER V2.3.1 are strictly routed to isolated experiment directories:
   - Surface experiments: `runs/surface_classifier_v231/`
   - Water experiments: `runs/water_segmenter_v231/`
   - D10 crack experiments: `runs/detect/d10_research_v231/`
