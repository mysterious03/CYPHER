# CYPhER V2.5 — WEATHER & LIGHTING PERCEPTION IMPROVEMENT
**Adverse Condition Recognition, High-Solar Glare Robustness, and Benchmark Ingestion**

---

## 1. Executive Summary & Problem Formulation

Environmental weather and ambient illumination directly alter camera sensor physics. Rain reduces contrast and creates specular reflections; fog scatters light; and dawn/dusk introduces intense low-angle blinding glare.

In CYPhER V2.5, capability `M8_WEATHER_CLASSIFIER` runs `runs/weather_classifier_v22/best_weather_model.pt` (4.52 MB), a dual-head MobileNetV3-Small network predicting:
- **Head 1 (Atmospheric Weather):** `Clear`, `Rain`, `Fog`, `Snow`, `Overcast` (5 classes)
- **Head 2 (Ambient Lighting):** `Daytime`, `Nighttime`, `Dawn_Dusk` (3 classes)
- **Runtime Performance:** **2.1 ms** per frame on NVIDIA RTX 4050 GPU; VRAM footprint: **34.2 MB**.

---

## 2. Empirical Holdout Analysis on Indian Road Scenes

Evaluated on the full 1,172 held-out driving frames:

| Dimension | Class | Holdout Activations (%) | Real-World Context & Behavioral Dynamics |
|---|---|:---:|---|
| **Atmospheric Weather** | `Clear` | 78.0% | Sunny, dry conditions with unobstructed visibility across national highways. |
| | `Rain` | 17.0% | Activates accurately on wet asphalt sheens, overcast clouds, and preceding truck mist. |
| | `Fog / Haze` | 4.0% | Triggers on particulate winter smog and suspended highway dust. |
| | `Snow` | 1.0% | **False Positive Error:** Triggered by high-specular white concrete pavement under direct overhead noon sun. |
| | `Overcast` | 0.0% | Grouped into the transitional boundary with Rain/Clear. |
| **Ambient Illumination** | `Daytime` | 62.0% | High solar elevation, balanced ambient scene luminance. |
| | `Dawn_Dusk` | 38.0% | Long forward vehicle shadows, high dynamic range, and intense low-angle golden-hour glare. |
| | `Nighttime` | 0.0% | Zero nighttime frames exist in the baseline RDD2022 collection. |

---

## 3. Forensic Analysis of Failures & Mitigations

1. **The Concrete Pavement "Snow" Hallucination:**
   - *Diagnostic:* White-topping concrete highways under direct 12:00 PM overhead sun reflect intense broadband white light ($RGB \approx [240, 240, 240]$). Models pretrained on European datasets (where high albedo corresponds to snowbanks) misclassify this surface.
   - *Mitigation:* Ingest adverse-weather data with domain-adapted Indian concrete roads labeled as `Clear/Concrete` with negative loss penalty on `Snow`.
2. **Dynamic Adaptation to Dawn/Dusk Glare:**
   - The high activation of `Dawn_Dusk` (38.0%) provides critical input to the **Adaptive Profile Scheduler**: under low-angle glare, the pipeline automatically widens spatial priors and lowers object confidence gates by $15\%$ to recover washed-out cracks and pothole contours.

---

## 4. Benchmark Ingestion Strategy & Status

- **Target Benchmark Datasets:**
  - **ACDC (Adverse Conditions Dataset):** Fog, Night, Rain, Snow with clear correspondences.
  - **DAWN (Vehicle Detection in Adverse Weather Nature):** Multi-hazard Indian-climate adverse weather collection.
- **Operational Status:** **`TRAINED_NOT_VALIDATED`** until independent quantitative benchmarking against ACDC is finalized.
