# CYPhER V2.3 — SENSOR FUSION & GEOMETRIC CALIBRATION AUDIT

**Capabilities Audited:**  
- `M13_FRICTION_ESTIMATE` (Tyre-Road Friction Coefficient $\mu$)  
- `M14_ROUGHNESS_IRI` (Road Roughness / International Roughness Index)  
- `M15_POTHOLE_GEOMETRY` (Calibrated Pothole Metric Dimensions & Depth)  
**Taxonomy Statuses:** `SENSOR_REQUIRED` / `CALIBRATION_REQUIRED`  

---

## 1. Executive Summary & Verification of Safety Gates

A crucial architectural achievement of CYPhER V2.2 and V2.3 is the enforcement of **strict scientific boundaries** between pure visual estimation and physical ground-truth measurements. 

Prior systems frequently hallucinated exact friction values (e.g. $\mu = 0.82$) or pothole depths (e.g. "6.3 cm deep") from uncalibrated 2D RGB snapshots. The code audit confirms that all three capabilities are now protected by **hardware and calibration gates**:

1. **`M13_FRICTION_ESTIMATE`:** If vehicle CAN bus telemetry (wheel slip ratio, longitudinal acceleration, yaw rate) is absent, the system **blocks** physical friction claims and outputs strictly a `VISUAL_SLIP_RISK_ESTIMATE` $[0, 1]$. Status: **`SENSOR_REQUIRED`**.
2. **`M14_ROUGHNESS_IRI`:** If vehicle vertical accelerometers and suspension displacement sensors are absent, the system **blocks** physical IRI claims (m/km) and outputs strictly a `VISUAL_ROUGHNESS_CLASS` (Smooth, Slightly Uneven, Severely Uneven). Status: **`SENSOR_REQUIRED`**.
3. **`M15_POTHOLE_GEOMETRY`:** If camera intrinsic parameters, mounting height ($h_{\text{cam}}$), and pitch angle ($\theta_{\text{pitch}}$) are absent, the system **blocks** metric centimeter measurements and outputs strictly normalized bounding box pixel coordinates and relative depth ratios. Status: **`CALIBRATION_REQUIRED`**.

---

## 2. Friction Estimation Pipeline (`M13_FRICTION_ESTIMATE`)

- **Code Path:** `road_cv/fusion/friction_estimator.py`
- **Physical Model:** Magic Formula (Pacejka) tyre slip curves combined with Dugoff tire-road friction models.
- **Safety Gate Implementation:**
  ```python
  if can_telemetry is None or "wheel_speed" not in can_telemetry:
      return {
          "status": "SENSOR_REQUIRED",
          "metric_friction_mu": None,
          "visual_slip_risk": visual_surface_risk_score,
          "warning": "Physical friction coefficient requires vehicle wheel speed and IMU telemetry."
      }
  ```
- **Auditor Verification:** Confirmed that running `python -m road_cv.main --mode friction` without external telemetry correctly emits the visual risk fallback without fabricating numerical $\mu$.

---

## 3. Road Roughness & IRI Pipeline (`M14_ROUGHNESS_IRI`)

- **Code Path:** `road_cv/fusion/iri_estimator.py`
- **Physical Model:** Quarter-Car Golden Car simulation (ISO 8608 standard) integrating vertical acceleration $\ddot{z}_s$ over travel distance.
- **Safety Gate Implementation:**
  ```python
  if imu_telemetry is None or "vertical_accel" not in imu_telemetry:
      return {
          "status": "SENSOR_REQUIRED",
          "iri_m_per_km": None,
          "visual_unevenness": surface_classification["unevenness"],
          "warning": "Physical IRI calculation requires vertical accelerometer data."
      }
  ```
- **Auditor Verification:** Verified that the IRI module cleanly gates outputs to visual categories when physical accelerometer streams are absent.

---

## 4. Calibrated Pothole Geometry Pipeline (`M15_POTHOLE_GEOMETRY`)

- **Code Path:** `road_cv/geometry/pothole_dimensions.py`
- **Geometric Model:** Inverse Perspective Mapping (IPM) projecting pixel bounding box bounds onto the ground plane using camera intrinsics matrix $K$ and extrinsic pitch rotation $R(\theta)$.
- **Calibration Check:**
  ```python
  if camera_calibration is None or "focal_length" not in camera_calibration:
      return {
          "status": "CALIBRATION_REQUIRED",
          "metric_width_cm": None,
          "metric_length_cm": None,
          "metric_depth_cm": None,
          "pixel_dimensions": {"width_px": w_px, "height_px": h_px},
          "warning": "Metric pothole dimensions require camera intrinsic and extrinsic calibration."
      }
  ```
- **Auditor Verification:** Calibrated mode operates accurately when provided with synthetic calibration matrices, and cleanly gates to pixel extents when run on uncalibrated dashcam imagery.
