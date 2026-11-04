# Road Damage Perception — Research Experiment Tracker

Tracks progress across planned research experiments (B0 through B5) on RDD2022 India.

| Experiment ID | Description | Status | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall | F1 Score | Key Focus |
|---|---|---|---:|---:|---:|---:|---:|---|
| **B0** | Baseline fine-tuned YOLO26n (640 px, batch 4) | **COMPLETED** | **0.2770** | **0.1316** | **0.5955** | **0.2861** | **0.3865** | Reproducible baseline detector |
| **B1A** | Class-Aware Sampling ($4\times$ rare D10 exposure) | **IMPLEMENTED** | Pipeline Validated | Pipeline Validated | Verified | Verified | - | Boosts D10 presentation from 0.8% to 3.6% |
| **B1B** | Cui et al. Effective-Number Loss ($\beta=0.997, W_{D10}=7.14\times$) | **IMPLEMENTED** | Pipeline Validated | Pipeline Validated | Verified | Verified | - | Safe gradient scaling without SGD explosion |
| **B1C** | Targeted D10 Augmentation (hflip, tilt $\pm 4^\circ$, photometric) | **IMPLEMENTED** | Pipeline Validated | Pipeline Validated | Verified | Verified | - | Prevents 90° rotation into longitudinal cracks |
| **B1D** | Unified B1 Strategy (Sampling + Weights + Aug + Hard Negs) | **ACTIVE** | Target $\ge 0.30$ | Target $\ge 0.15$ | Target $\ge 0.60$ | Target $\ge 0.32$ | - | Complete rare-class recovery & FP reduction |
| **B2** | Small-Defect Multi-Scale Inference ($1280\times 1280$) + WBF | **VALIDATED** | $4\times$ Defect Recall | - | High | High | - | 32.97 ms on RTX 4050, 118 MB peak VRAM |
| **B3** | Quality-Aware Cross-Condition Robustness | PLANNED | - | - | - | - | - | Glare, night, and shadow adaptation |
| **B4** | Full Framework (B1D + Context + Refinement + Reliability) | PLANNED | - | - | - | - | - | End-to-end evidence-based perception |
