# CYPhER V2.2 Workstream B: Waterlogging Evaluation Protocols & Failure Taxonomy

**Auditor / Researcher:** CYPhER Computer Vision Research Team  
**Date:** November 5, 2026  
**Document Purpose:** Evaluation criteria, failure mode taxonomy, and verification guidelines for road water segmentation

---

## 1. Quantitative Evaluation Protocols

Once authentic pixel ground truth is trained, models must be benchmarked using the following metrics:

1. **Mean Intersection over Union (mIoU):**
   $$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}, \quad \text{mIoU} = \frac{1}{4} \sum_{c=0}^3 \text{IoU}_c$$
2. **Puddle-Specific Dice Coefficient:**
   $$\text{Dice}_{\text{puddle}} = \frac{2 \times \text{TP}_{\text{puddle}}}{2 \times \text{TP}_{\text{puddle}} + \text{FP}_{\text{puddle}} + \text{FN}_{\text{puddle}}}$$
3. **Specular Glare Rejection Rate:**
   Evaluated on dry road images with high solar illuminance ($>150$ mean luminance). A valid authentic model must exhibit **$\text{FP}_{\text{puddle}} < 0.5\%$ of road area** on dry asphalt.

---

## 2. Six-Case Visual Failure Taxonomy

The evaluation test suite must explicitly sample and verify predictions across these six critical road scenarios:

```
+-----------------------------------------------------------------------------------------+
| Scenario Case                  Visual Signature               Target Ground Truth Action|
+-----------------------------------------------------------------------------------------+
| 1. Dry Asphalt Reflection      Sunlight glare on smooth road  Class 0 (Dry) or 3 (Glare)|
| 2. Wet Damp Pavement           Darkened surface, no depth     Class 0 (Dry/Damp Road)   |
| 3. Genuine Localized Puddle    Boundaries with water meniscus Class 2 (Puddle)          |
| 4. Flooded Roadway             Continuous standing water sheet Class 1 (Flood)           |
| 5. Tree / Building Shadows     Dark high-contrast boundary    Class 0 (Dry Road)        |
| 6. Waterlogged Structural Pothole Water filling cavity depth   Class 2 + Pothole overlap |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Scientific Integrity Status

Pending the intake of authentic pixel masks into `data/processed/water_authentic/`, the existing pseudo-mask checkpoint remains quarantined. The capability is labeled `DATASET_REQUIRED` in the model registry to maintain complete research honesty.
