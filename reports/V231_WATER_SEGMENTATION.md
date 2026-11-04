# CYPhER V2.3.1 — AUTHENTIC WATERLOGGING SEGMENTATION AUDIT & RETRAINING SPECIFICATION

**Audit Date:** November 5, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Evaluated Checkpoint:** `runs/water_segmenter/best_water_model.pt` (Quarantined)  
**Target Architecture:** LRASPP MobileNetV3-Large (12.48 MB) & SegFormer-B0  
**Current Taxonomy Status:** `DATASET_REQUIRED`  

---

## 1. Executive Summary & Scientific Quarantine

In previous releases (V2.0/V2.1), the road waterlogging model reported a synthetic mIoU of 71.4%. An independent forensic audit confirmed that this model was trained exclusively on **heuristic pseudo-masks** generated via HSV color slicing and specular thresholding.

### Physical Failure Modes of Pseudo-Mask Training:
1. **Pavement Discoloration & Fresh Bitumen:** Dark, newly laid asphalt exhibits low luminance and specular reflections under direct sunlight. Heuristic pseudo-labels consistently tag dry, smooth bitumen as "flood water."
2. **Shadow Ingestion:** Dense tree shadows and bridge underpass cast shadows exhibit chromatic attenuation that matches heuristic puddle thresholds.
3. **Hydroplaning Risk Misclassification:** Pseudo-masks cannot differentiate between damp surface sheen ($<1\text{ mm}$ water depth, low hydroplaning risk) and deep standing pools ($>50\text{ mm}$ water depth, acute hydroplaning risk).

**Mandatory Action Taken:**
The legacy checkpoint `runs/water_segmenter/best_water_model.pt` is **permanently quarantined** and marked `DATASET_REQUIRED`. No synthetic pseudo-mask metrics are recognized as validated performance.

---

## 2. Ingestion Pipeline & Architecture Architecture

To establish authentic neural segmentation, the dedicated ingestion pipeline `road_cv/water/authentic_water_dataset.py` and `scripts/prepare_authentic_water_pipeline.py` have been implemented:

### A. Verified Class Taxonomy
- **Class 0:** Dry asphalt / concrete pavement
- **Class 1:** Standing flood water (continuous standing bodies spanning carriageways)
- **Class 2:** Localized water puddle (discrete surface depressions filled with water)
- **Class 3:** Specular wet reflection / glare (damp asphalt displaying optical light reflection without standing depth)

### B. Anti-Hallucination Data Integrity Rules
- The dataset loader requires verified polygon PNG/TIFF masks.
- If a sample does not have an authentic ground-truth mask on disk, the loader raises `FileNotFoundError` rather than silently generating heuristic pseudo-masks.

---

## 3. Candidate Benchmark Datasets for Production Retraining

To transition the waterlogging capability from `DATASET_REQUIRED` to `VALIDATED_STRONG`, the following authentic pixel-annotated benchmarks have been cataloged:

| Candidate Dataset | Ground Truth Quality | Annotation Type | Relevant Classes | License & Access |
| :--- | :--- | :--- | :--- | :--- |
| **R²S100K** | High-fidelity highway & urban roads | Pixel-level semantic polygons | Wet road, Puddles, Flooded road | Academic Research Only (Request required) |
| **RPI-Puddle / RTK-Water** | Focused puddle geometry | Multi-spectral & RGB polygon masks | Standing puddles, Reflections | Open Access (Creative Commons) |
| **ACDC (Adverse Conditions)** | Adverse weather dashcam scenes | Fine pixel-level segmentation | Wet road, Water spray, Flooded road | Open Access (Commercial/Academic) |

---

## 4. Retraining Protocol & Acceptance Gate

When the authentic benchmark masks are ingested into `data/processed/water_authentic/`:
1. **Training Loss:** Joint Cross-Entropy + 1.5 $\times$ Soft-Dice Loss:
   $$\mathcal{L} = \mathcal{L}_{\text{CE}} + 1.5 \left(1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i^2 + \sum g_i^2 + \epsilon}\right)$$
2. **Glaze Rejection Test:** Evaluation on a curated 100-frame test set consisting of:
   - Sunlit smooth dry asphalt (must yield 0% water detection)
   - Tree shadows on asphalt (must yield 0% water detection)
   - Real monsoon road flood images (target mIoU $\ge 70.0\%$)
