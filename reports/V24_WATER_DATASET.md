# CYPhER V2.4 — AUTHENTIC WATERLOGGING DATASET PROTOCOL
**Dataset Ingestion Architecture & Forensic Quarantine of Synthetic Heuristics**

---

## 1. Forensic Audit & Quarantine of Legacy Water Pipeline

In legacy CYPhER iterations, attempts were made to train water segmentation models using heuristic pseudo-masks (e.g., thresholding low-luminance road areas, HSV color-space slicing for blue-gray tints, or specular highlight reflections).

### Why Synthetic Pseudo-Masks Were Completely Quarantined
1. **Severe Shadow & Asphalt Confusion:** Shadows from flyovers, trucks, trees, and newly resurfaced bitumen share identical low-luminance and low-saturation color profiles with shallow water film. Heuristic generators falsely labeled shadows as "waterlogged".
2. **Failure Under Turbid / Muddy Water Conditions:** Authentic puddles and waterlogged potholes on Indian monsoon roads are muddy, clay-brown, or light-scattering gray—not blue or specular. Color-heuristic pseudo-masks systematically failed to identify muddy water.
3. **Glare and Overexposure Hallucination:** Under direct sun reflection, dry smooth asphalt produces white specular glare that heuristic filters mistook for water reflection.
4. **Scientific Integrity Mandate:** Training a deep neural network (e.g., LRASPP MobileNetV3) on pseudo-masks merely trains the network to replicate the errors of a crude color filter, giving a false illusion of mIoU while collapsing completely on real roads.

**Quarantine Decision:** The legacy synthetic water pipeline remains quarantined. Its status in the CYPhER V2.4 Model Registry is strictly defined as **`DATASET_REQUIRED`**.

---

## 2. Ingestion Protocol for Authentic Pixel-Level Benchmarks

To transition waterlogging detection to a validated scientific state, CYPhER V2.4 specifies strict ingestion criteria for genuine, human-annotated pixel-level water segmentation datasets:

### Target Authentic Datasets
1. **RTK-Water Dataset:**
   - Real-world puddle and waterlogged road images captured under diverse weather conditions.
   - High-fidelity polygonal segmentation masks dividing scenes into `dry_road`, `puddle/water`, and `background`.
2. **ACDC (Adverse Conditions Dataset with Correspondences - Rain Split):**
   - Cityscapes-aligned semantic segmentation containing explicit water surface annotations under rainy and post-rain conditions.
3. **Indian Driving Puddle Benchmark (IDPB) Protocol:**
   - In-house collection of dashcam footage from Indian monsoon environments (Mumbai, Bengaluru, Delhi NCR) with manual polygon annotations for turbid puddles, deep waterlogging, and flooded gutters.

### Annotation Standards Specification
- **Format:** Single-channel 8-bit PNG masks where:
  - `0`: Background / Non-road
  - `1`: Dry drivable road surface
  - `2`: Waterlogged road / Puddle (depth $\ge 1 \text{ cm}$)
  - `255`: Ignore / Void (distant horizon, glare boundary)
- **Minimum Resolution:** $1920 \times 1080$ native, scaled to $640 \times 640$ for training.
- **Negative Sample Ratio:** At least 30% of training frames must be dry roads with harsh shadows and sun glare to enforce negative suppression.

---

## 3. Data Pipeline Directory Layout

```
data/
└── authentic_water/
    ├── raw/
    │   ├── images/
    │   └── masks/
    ├── splits/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test_hard_negatives.txt
    └── processed/
        ├── images_640/
        └── masks_640/
```

Until these authentic ground-truth archives are downloaded and verified, no water segmentation model weights will be certified for production autonomy.

