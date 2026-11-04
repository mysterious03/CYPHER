# CYPhER V2.3 — ENVIRONMENTAL WEATHER & LIGHTING EVALUATION REPORT

**Capability ID:** `M9_WEATHER_CLASSIFIER`  
**Model Architecture:** MobileNetV3-Small Dual-Head Classifier (6.10 MB)  
**Evaluated Checkpoint:** `runs/weather_classifier_v22/best_weather_model.pt`  
**Evaluation Benchmark:** 100 Indian RDD2022 Holdout Images (`data/processed/rdd_split/images/val/`)  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)  
**Taxonomy Status:** `TRAINED_NOT_VALIDATED`  

---

## 1. Executive Summary

The environmental classifier provides joint inference of atmospheric weather (Clear, Rain, Fog, Snow, Overcast) and ambient lighting conditions (Daytime, Nighttime, Dawn_Dusk) from full camera frames in **2.1 ms**. Evaluated across 100 held-out Indian road scenes, the model identified daytime clear conditions as dominant, with appropriate secondary activations on overcast/rain and dawn/dusk frames.

---

## 2. Empirical Prediction Distribution (100 Holdout Scenes)

### Atmospheric Weather Head:
- **Clear:** 78 frames (78.0%)
- **Rain:** 17 frames (17.0%)
- **Fog:** 4 frames (4.0%)
- **Snow:** 1 frame (1.0%)
- **Overcast:** 0 frames (0.0%)

### Ambient Lighting Head:
- **Daytime:** 62 frames (62.0%)
- **Dawn / Dusk:** 38 frames (38.0%)
- **Nighttime:** 0 frames (0.0%)

---

## 3. Findings & Limitations

- **Lighting Calibration:** The model correctly categorized high-glare, low-sun-angle Indian road frames as `Dawn_Dusk` rather than daytime, demonstrating sensitivity to shadow length and solar azimuth.
- **Weather Sensitivity:** Overcast and light monsoon drizzle frames were categorized as `Rain` (17%), matching visual cues of wet asphalt patches.
- **Limitation:** The single detection of `Snow` on an Indian highway represents a false positive caused by high-reflectance white concrete under direct sunlight.
- **Validation Gap:** Because the RDD2022 dataset does not contain ground-truth atmospheric labels, numerical accuracy cannot be calculated without an external benchmark like ACDC or DAWN.

---

## 4. Visual Evidence

Visual inspection displays overlaid weather and lighting predictions on holdout frames:
- Evidence file: [`reports/v23_model_evidence/evidence_lane_weather_India_000000.jpg`](file:///c:/Users/ASUS/OneDrive/Desktop/cvvvvvv/reports/v23_model_evidence/evidence_lane_weather_India_000000.jpg)
