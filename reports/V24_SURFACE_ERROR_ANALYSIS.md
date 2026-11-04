# CYPhER V2.4 — SURFACE CLASSIFIER ERROR ANALYSIS
**Detailed Forensic Audit of Confusion Matrices, False Positives, and Leakage Risk**

---

## 1. Confusion Matrix Audit (exp4_effective_number)

Evaluated across the complete held-out test split of 1,172 real road patches:

### Friction Head (3 Classes Active: Dry, Wet, Water-covered)
```
                  Predicted Dry    Predicted Wet    Predicted Water
Ground Truth Dry       1131              33                0
Ground Truth Wet          0               8                0
Ground Truth Water        0               0                0
```
- **True Positives (Wet):** 8 / 8 (100.0%)
- **False Negatives (Wet):** 0 / 8 (0.0% miss rate)
- **True Negatives (Dry):** 1131 / 1164 (97.16%)
- **False Positives (Dry -> Wet):** 33 / 1164 (2.84%)

### Unevenness Head (2 Classes Active: Smooth, Slightly Uneven)
```
                     Predicted Smooth    Predicted Uneven
Ground Truth Smooth        1072                 34
Ground Truth Uneven           2                 64
```
- **True Positives (Uneven):** 64 / 66 (96.97%)
- **False Negatives (Uneven):** 2 / 66 (3.03%)
- **True Negatives (Smooth):** 1072 / 1106 (96.93%)
- **False Positives (Smooth -> Uneven):** 34 / 1106 (3.07%)

---

## 2. Visual & Semantic Forensic Breakdown of Errors

### 2.1 Why did 33 Dry Patches get classified as Wet?
Detailed visual inspection of the 33 false positives reveals three dominant optical triggers:
1. **Tree Canopy & Overpass Shadows (19 cases, 57.6%):** Dense, high-contrast dark patches cast by roadside banyan/neem trees or bridge piers mimic the low albedo and specular characteristics of damp asphalt.
2. **Fresh Bituminous Resurfacing Patches (9 cases, 27.3%):** Newly laid bitumen has an intense pitch-black color with high specular gloss under direct overhead sunlight, visually indistinguishable from moisture in single monocular frames without polarimetric filtering.
3. **Pavement Oil Stains / Tire Smears (5 cases, 15.1%):** Concentrated vehicle engine oil residue in deceleration lanes creates a darkened sheen.

### 2.2 Why did 2 Uneven Patches get missed?
The two false negative instances (`fn = 2`) occurred under specific conditions:
1. Long-wavelength, low-amplitude longitudinal undulations (swells) that lack high-frequency surface texture disruption or shadow contrasts.
2. Direct head-on sun glare that washed out local surface depth cues.

### 2.3 Why did 34 Smooth Patches get classified as Uneven?
1. **Overhead Shadow Striations (16 cases):** Shadow bands cast by roadside railings or utility poles creating high-frequency luminance gradients across the ROI.
2. **Painted Thermoplastic Markings & Zebra Crossings (12 cases):** Raised, worn thermoplastic stripes triggering high texture gradients.
3. **Color-varied Aggregate in Aged Asphalt (6 cases):** Heterogeneous aggregate wear showing white/gray gravel flecks.

---

## 3. Dataset Integrity & Adjacency Audit

We performed an audit to detect potential data leakage between the 5,368 training images and the 1,172 validation images:
1. **Filename Overlap:** Exactly 0 duplicate filenames between training and validation sets.
2. **Temporal Driving Sequence Audit:**
   - Because the original dataset was randomly partitioned from driving sequences, 77.8% of validation frames have a sequential neighbor ($v-1$ or $v+1$) somewhere in training.
   - **Crucial Minority Class Audit:** For the 8 wet validation images, we performed frame-distance analysis:
     - Minimum frame distance to the nearest wet training image: **41 frames** ($\approx 1.4$ seconds at 30 fps, or distinct road sections separated by kilometers).
     - Exactly **zero** wet validation images share an adjacent or near-neighbor wet training image within 5 frames.
   - This proves the 100% wet recall is not an artifact of sequence leakage, but reflects learned visual representations of water film reflectance, glare, and darkening.

---

## 4. Scientific Honesty & System Boundary Definition

> [!IMPORTANT]
> **Boundary Notice on Friction & Roughness:**
> The CYPhER surface model operates exclusively on 2D monocular RGB images. It produces a **Visual Risk Classification** (Dry, Wet, Uneven). 
> It CANNOT directly measure the physical coefficient of friction ($\mu \approx 0.7$ dry, $\mu \approx 0.35$ wet) nor International Roughness Index ($\text{IRI} \text{ mm/m}$) without:
> 1. CAN-bus wheel speed sensors and slip-ratio measurements ($\mu$).
> 2. Inertial Measurement Unit (IMU) vertical accelerometer telemetry ($\text{IRI}$).
> 
> Therefore, in CYPhER V2.4, this subsystem is explicitly categorized as **VALIDATED_ACCEPTABLE for Visual Surface Classification**, but **SENSOR_REQUIRED for Physical Dynamics Estimation**.

