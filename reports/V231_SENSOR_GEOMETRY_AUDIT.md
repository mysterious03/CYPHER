# CYPhER V2.3.1 — PHYSICAL SENSOR & GEOMETRIC CALIBRATION AUDIT

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Capabilities Audited:**  
- `M13_FRICTION_ESTIMATE` (Tyre-Road Friction Coefficient $\mu$)  
- `M14_ROUGHNESS_IRI` (International Roughness Index IRI)  
- `M15_POTHOLE_GEOMETRY` (Calibrated Pothole Dimensions & Depth)  
**Taxonomy Statuses:** `SENSOR_REQUIRED` / `CALIBRATION_REQUIRED`  

---

## 1. Executive Summary & Scientific Demarcation

A primary hazard in autonomous perception systems is "sensor hallucination"—claiming to measure physical dynamic properties (such as tyre friction $\mu$, road roughness in m/km, or pothole depth in centimeters) using only an uncalibrated 2D forward-facing dashboard camera.

In CYPhER V2.3.1, strict scientific boundaries and programmatic gates are enforced. When external hardware telemetry or calibration matrices are absent, the system **actively blocks** physical claims and degrades cleanly to qualitative visual indicators.

---

## 2. Capability Audit: Tyre-Road Friction (`M13_FRICTION_ESTIMATE`)

- **Physical Reality:** Tyre-road friction coefficient $\mu$ is an interface property determined by tire tread compound, contact patch temperature, inflation pressure, normal load, and wheel slip ratio ($s = \frac{\omega r - v}{v}$). An RGB camera observes only surface texture and reflectance; it cannot measure contact mechanics.
- **Code Gate Implementation (`road_cv/fusion/friction_estimator.py`):**
  - Requires: `wheel_speed`, `vehicle_speed`, `longitudinal_accel`, `yaw_rate`.
  - Behavior when absent: Returns `status: SENSOR_REQUIRED`, `metric_friction_mu: None`, and provides strictly a `visual_slip_risk` $[0, 1]$ based on wetness/ice visual probability.
- **Auditor Verification:** Confirmed that running pure vision inputs never emits a numerical $\mu$ value without physical CAN telemetry.

---

## 3. Capability Audit: Road Roughness IRI (`M14_ROUGHNESS_IRI`)

- **Physical Reality:** The International Roughness Index (IRI, m/km) is defined by the World Bank standard as the accumulated suspension motion of a standardized Quarter-Car model traveling at 80 km/h over longitudinal road profiles. Monocular 2D images cannot measure vertical profile wavelengths without profilometers or vehicle accelerometer logging.
- **Code Gate Implementation (`road_cv/fusion/iri_estimator.py`):**
  - Requires: `vertical_accel` (from IMU), `vehicle_speed`, `gyro_pitch_rate`.
  - Behavior when absent: Returns `status: SENSOR_REQUIRED`, `iri_m_per_km: None`, and provides strictly a `visual_roughness_class` (`Smooth`, `Slightly uneven`, `Severely uneven`).
- **Auditor Verification:** Verified that the system halts IRI estimation and logs an explicit sensor requirement warning when vehicle IMU streams are missing.

---

## 4. Capability Audit: Calibrated Pothole Dimensions (`M15_POTHOLE_GEOMETRY`)

- **Physical Reality:** Monocular RGB cameras suffer from fundamental projective scale ambiguity. A 50-pixel depression could be a 10 cm pothole nearby or a 1-meter trench far away. Inverse Perspective Mapping (IPM) can reconstruct planar ground extents only if camera height ($h_{\text{cam}}$), tilt/pitch angle ($\theta$), and intrinsic focal lengths ($f_x, f_y$) are known. Furthermore, IPM assumes a flat planar ground; depression depth into the road surface cannot be measured by IPM alone without stereo disparity, structured light, or LiDAR.
- **Code Gate Implementation (`road_cv/geometry/pothole_dimensions.py`):**
  - Requires: Intrinsic camera matrix $K$, camera height $h_{\text{cam}}$, pitch angle $\theta_{\text{pitch}}$.
  - Behavior when absent: Returns `status: CALIBRATION_REQUIRED`, `metric_width_cm: None`, `metric_depth_cm: None`, and outputs strictly normalized bounding box pixel extents.
- **Auditor Verification:** Verified that metric centimeter measurements are suppressed on uncalibrated dashcam feeds.
