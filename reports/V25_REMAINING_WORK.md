# CYPhER V2.5 — REMAINING WORK & NEXT TECHNICAL PRIORITIES
**Actionable Research Roadmap for Subsequent CYPhER Iterations**

---

## 1. Top Five Strategic Technical Priorities

### Priority 1: India Driving Dataset (IDD) Fine-Tuning for Mixed Traffic (M4 & M5)
- **Problem:** Pretrained MS COCO and Cityscapes weights are blind to auto-rickshaws, e-rickshaws, tractors, and unpaved dirt shoulders.
- **Action Plan:** Ingest IDD Level-3 annotations ($\approx 10,000$ images). Re-anchor the YOLO26n-seg and YOLO26n-sem heads using k-means bounding box clustering and train with Class-Balanced Effective-Number Loss ($\beta=0.999$).
- **Success Criteria:** Auto-rickshaw recall $\ge 75.0\%$, agricultural tractor recall $\ge 70.0\%$, drivable unpaved surface $\text{mIoU} \ge 70.0\%$.

### Priority 2: Ingestion of Authentic Waterlogging Ground Truth (M3)
- **Problem:** Legacy synthetic pseudo-masks remain quarantined due to severe false alarms on tree/flyover shadows.
- **Action Plan:** Ingest the R²S100K and RTK-Water datasets. Standardize to 8-bit polygonal masks ($0$: dry road, $1$: standing water). Train LRASPP MobileNetV3-Large using the hybrid Focal-Dice-Boundary loss formulation.
- **Success Criteria:** Water puddle $\text{mIoU} \ge 65.0\%$; false positive rate on dry shadow test suite $< 0.1\%$.

### Priority 3: 3D Spatiotemporal Graph Convolutional Network (ST-GCN) for Accidents (M11)
- **Problem:** 2D bounding-box area expansion heuristics generate false emergency braking alarms when the host vehicle pitches over road irregularities.
- **Action Plan:** Ingest the DADA-2000 and DoTA accident video benchmarks. Construct dynamic graph representations of vehicle tracklets ($[x, y, w, h, v_x, v_y, \text{cls}, d_{\text{depth}}]$) and train an ST-GCN head to predict collision probability 2.0 to 3.0 seconds prior to physical impact.
- **Success Criteria:** Pre-crash anticipation time $\ge 2.0 \text{ s}$; false alarm rate $< 0.05$ per driving hour.

### Priority 4: Camera Intrinsic & Extrinsic Homography Calibrator (M6 & M12)
- **Problem:** Monocular relative depth cannot output real-world physical dimensions (meters, centimeters) due to projective scale ambiguity.
- **Action Plan:** Develop an automated vanishing-point / road-plane homography calibration tool to estimate camera focal length ($f$), mounting height ($h_{\text{cam}}$), and pitch angle ($\theta_{\text{pitch}}$).
- **Success Criteria:** Pothole physical depth estimated within $\pm 1.5 \text{ cm}$ of ground-truth laser profilometer measurements.

### Priority 5: Real-World Vehicle Telemetry CAN-Bus & IMU Hardware Interfaces (M13 & M14)
- **Problem:** Physical tyre-road friction ($\mu$) and International Roughness Index ($\text{IRI}$) are physically unmeasurable from camera RGB pixels alone.
- **Action Plan:** Implement CAN-bus OBD-II telemetry drivers for wheel-speed slip ratios ($\mu$) and high-G vertical accelerometer logging via a 6-DOF IMU ($\text{IRI}$).
- **Success Criteria:** Direct output of physical $\mu \in [0.1, 0.9]$ and $\text{IRI} \in [1.0, 16.0] \text{ mm/m}$ synchronized with camera frame timestamps.

---

## 2. Technical Horizon Timeline

```
CYPhER V2.5 (Current) ─────► Phase 1: IDD Vehicle & Scene Fine-Tuning
                             ├── Auto-rickshaw & tractor heads
                             └── Unpaved dirt shoulder masks
                             
                             Phase 2: R²S100K Water Ingestion
                             └── Boundary-loss LRASPP training
                             
                             Phase 3: DADA-2000 Video Accident Modeling
                             └── ST-GCN graph anticipation
                             
                             Phase 4: Camera Homography Calibration
                             └── Real metric centimeters & meters
                             
                             Phase 5: Telemetry CAN-Bus & IMU Drivers
                             └── Real physical friction (μ) & IRI
```
