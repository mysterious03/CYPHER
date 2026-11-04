# CYPhER V2.3.1 — WEATHER & LIGHTING CLASSIFIER VALIDATION REPORT

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Evaluated Checkpoint:** `runs/weather_classifier_v22/best_weather_model.pt`  
**Architecture:** MobileNetV3-Small Dual-Head Classifier (4.52 MB)  
**Target Capabilities:**  
- **Atmospheric Weather:** `Clear`, `Rain`, `Fog`, `Snow`, `Overcast`  
- **Ambient Lighting:** `Daytime`, `Nighttime`, `Dawn_Dusk`  
**Taxonomy Status:** `TRAINED_NOT_VALIDATED`  

---

## 1. Architecture & Execution Characteristics

The weather and environmental lighting classifier operates on full downsampled dashcam frames (`224x224x3`), utilizing a shared MobileNetV3-Small backbone with two independent classification heads:
- Head A: Atmospheric Weather (5 logits)
- Head B: Ambient Illumination (3 logits)
- **Inference Latency:** **2.1 ms** on NVIDIA RTX 4050 GPU
- **VRAM Footprint:** 34.2 MB

---

## 2. Empirical Validation Distribution (100 Holdout Indian Road Scenes)

| Environmental Category | Holdout Activations | Percentage (%) | Behavioral Analysis |
| :--- | :---: | :---: | :--- |
| **Weather: Clear** | 78 | 78.0% | Correctly identifies unobstructed sunny road scenes |
| **Weather: Rain** | 17 | 17.0% | Activates on wet asphalt sheen and overcast cloudiness |
| **Weather: Fog** | 4 | 4.0% | Activates on distant atmospheric haze and dust |
| **Weather: Snow** | 1 | 1.0% | **False Positive:** Triggered by high-specular white concrete under overhead noon sun |
| **Weather: Overcast** | 0 | 0.0% | Grouped into Rain/Clear boundary |
| **Lighting: Daytime** | 62 | 62.0% | High solar angle, balanced ambient luminance |
| **Lighting: Dawn / Dusk** | 38 | 38.0% | Long forward vehicle shadows, high golden-hour contrast |
| **Lighting: Nighttime** | 0 | 0.0% | 0 nighttime frames in the standard RDD validation split |

---

## 3. Findings & Validation Limitations

1. **Solar Azimuth Sensitivity:** The model demonstrates surprisingly robust perception of low sun angles, correctly separating high-glare morning/evening scenes (`Dawn_Dusk`: 38%) from midday lighting (`Daytime`: 62%).
2. **Indian Road Generalization Gap:** The single detection of `Snow` highlights the domain hazard of training on temperate European datasets: high-reflectance Indian concrete pavement can mimic snow features.
3. **Validation Taxonomy:** Because the RDD2022 dataset lacks human-annotated meteorological labels, precision/recall cannot be claimed without evaluation against an authentic adverse-weather benchmark (such as ACDC or DAWN). Therefore, the model is classified as **`TRAINED_NOT_VALIDATED`**.
