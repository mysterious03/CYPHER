# NEW MODEL EXPERIMENT TRACKER

| Experiment ID | Task Description | Backbone Model | Dataset / Spec | Batch / Imgsz | Target Metric | Achieved Empirical Metric | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-B0-RDD** | Road Damage Detection | Custom YOLO26n | RDD2022 India | 4 / 640 | mAP50 $\ge 25\%$ | **mAP50: 27.70%, Precision: 59.55%, D20 mAP: 53.2%** | **VERIFIED & TESTED** |
| **EXP-S1-RSCD** | Multi-Task Surface Friction, Material, Unevenness | MobileNetV3-Large Multi-Task | RSCD Taxonomy | 32 / 224 | Top-1 Acc $\ge 80\%$ | **Friction: 98.67%, Material: 100.0%, Unevenness: 95.0% (Mean: 97.89%)** | **VERIFIED & TESTED** |
| **EXP-W1-WATER** | Waterlogging & Flood Segmentation | LRASPP MobileNetV3-Large | R²S100K Spec | 8 / 512 | Puddle IoU / Flood Area | **Trained & Saved (best_water_model.pt, Water IoU: 30.76%)** | **VERIFIED & TESTED** |
| **EXP-ENV-ACDC** | Adverse Weather Classifier | MobileNetV3 | ACDC Spec | 16 / 224 | Top-1 Acc $\ge 80\%$ | *Scheduled* | Phase 2 |
| **EXP-LANE-BDD** | Lane & Marking Boundary | YOLOP / Seg | BDD100K Spec | 8 / 640 | Lane F1 $\ge 75\%$ | *Scheduled* | Phase 2 |
