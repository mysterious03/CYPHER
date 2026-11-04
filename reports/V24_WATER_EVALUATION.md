# CYPhER V2.4 — WATERLOGGING EVALUATION & STATUS AUDIT
**Independent Evaluation Protocol & Scientific Status Certification**

---

## 1. Evaluation Methodology for Water Segmentation

When evaluating waterlogging segmenters, standard pixel accuracy is misleading because dry road pixels constitute $> 95\%$ of typical driving scenes. The CYPhER V2.4 evaluation framework mandates:

1. **Class Mean Intersection over Union (mIoU):**
   $$\text{IoU}_{\text{water}} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
2. **Boundary F1-Score ($\mathcal{BF}$):**
   Evaluates boundary contour alignment within an 8-pixel radius buffer, penalizing ragged or swollen puddle boundaries.
3. **Hard Negative Glare Rejection Rate:**
   Evaluated on a curated set of 200 dry-road images containing:
   - Direct low-angle sun glare.
   - Sharp shadows from flyovers, trees, and heavy trucks.
   - Fresh bituminous asphalt patches.
   - **Target Metric:** False Positive Puddle Area $< 0.1\%$ of total drivable road pixels.

---

## 2. Model Audit & Status Classification

### Current Status: `DATASET_REQUIRED`
- **Legacy Checkpoint:** Quarantined. Prior checkpoints trained on heuristic pseudo-masks failed the hard negative glare rejection test (exhibiting $> 34\%$ false positive water detection on dry shadow frames).
- **Current Operational Capability:**
  - Macro road wetness risk is reliably detected via the retrained **Multi-Task Surface Classifier** (`runs/surface_classifier_v24/best_surface_model.pt`), which achieves **100.0% recall** on wet road conditions.
  - Pixel-level water puddle boundary segmentation is classified as **`DATASET_REQUIRED`** until authentic polygonal annotations are ingested.

---

## 3. Scientific Verification Checklist

- [x] Quarantined pseudo-mask generation script.
- [x] Defined authentic dataset schema and ingestion requirements.
- [x] Established boundary-aware loss formulation and hard negative test criteria.
- [x] Gated model deployment in Model Registry to prevent fabricated evaluation metrics.

