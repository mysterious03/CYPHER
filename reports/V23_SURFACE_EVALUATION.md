# CYPhER V2.3 — SURFACE CONDITION CLASSIFIER EVALUATION REPORT

**Capability ID:** `M2_SURFACE_CLASSIFIER`  
**Model Architecture:** MobileNetV3-Large Multi-Task (3 Heads, 16.20 MB)  
**Evaluated Checkpoint:** `runs/surface_classifier/best_surface_model.pt`  
**Dataset:** RSCD Multi-Task Surface Validation Split (`val_patches.json`)  
**Total Held-Out Patches Evaluated:** 1,172 patches  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)  
**Taxonomy Status:** `VALIDATED_WEAK`  

---

## 1. Executive Summary & Headline Discrepancy

On paper, the multi-task surface classifier demonstrates an outstanding headline accuracy of **97.90%**. However, rigorous full-validation auditing reveals that this performance is an artifact of severe benchmark class imbalance. The model has degenerated into a majority-class predictor.

| Head | Number of Classes | Evaluated Support | Raw Accuracy (%) | Balanced Accuracy (%) | Macro F1-Score (%) | Weighted F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friction Head** | 6 classes | 1,172 patches | **99.32%** | **50.00%** | **49.83%** | 98.98% |
| **Material Head** | 6 classes | 1,172 patches | **100.00%** | **100.00%** | **100.00%** | 100.00% |
| **Unevenness Head** | 4 classes | 1,172 patches | **94.37%** | **50.00%** | **48.55%** | 91.63% |
| **Overall Multi-Task** | 16 total classes | 1,172 patches | **97.90%** | **66.67%** | **66.13%** | **96.87%** |

---

## 2. Friction Head Breakdown (6 Classes)

### Ground Truth Class Distribution:
- **Dry:** 1,164 patches (99.32%)
- **Wet:** 8 patches (0.68%)
- **Water-covered / Snow / Ice:** 0 patches (0.00%)

### Empirical Performance:
- **Dry:** Precision = 99.32%, Recall = 100.00%, F1 = 99.66% (TP = 1,164, FP = 8, FN = 0)
- **Wet:** Precision = 0.00%, Recall = 0.00%, F1 = 0.00% (TP = 0, FP = 0, FN = 8)
- **All other conditions:** N/A (zero test support)

### Confusion Matrix:
$$\begin{pmatrix} 1164 & 0 & 0 & 0 & 0 & 0 \\ 8 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \end{pmatrix}$$

Every single wet patch (8 out of 8) was misclassified as **Dry**. The model achieves 99.32% raw accuracy purely because 99.32% of the validation split is dry asphalt. Its true macro-F1 is only **49.83%**.

---

## 3. Material Head Breakdown (6 Classes)

### Ground Truth Class Distribution:
- **Asphalt:** 1,172 patches (100.00%)
- **Concrete / Gravel / Dirt / Mud / Loose Stone:** 0 patches (0.00%)

### Empirical Performance:
- **Asphalt:** Precision = 100.00%, Recall = 100.00%, F1 = 100.00% (TP = 1,172)
- While the model correctly predicts asphalt for every patch, the current validation split contains zero representation of gravel, concrete, or dirt roads. Its generalisation to unpaved or non-asphalt roads remains unverified.

---

## 4. Unevenness Head Breakdown (4 Classes)

### Ground Truth Class Distribution:
- **Smooth:** 1,106 patches (94.37%)
- **Slightly Uneven:** 66 patches (5.63%)
- **Severely Uneven / Deteriorated:** 0 patches (0.00%)

### Empirical Performance:
- **Smooth:** Precision = 94.37%, Recall = 100.00%, F1 = 97.10% (TP = 1,106, FP = 66, FN = 0)
- **Slightly Uneven:** Precision = 0.00%, Recall = 0.00%, F1 = 0.00% (TP = 0, FP = 0, FN = 66)
- **Severely Uneven / Deteriorated:** N/A (zero test support)

### Confusion Matrix:
$$\begin{pmatrix} 1106 & 0 & 0 & 0 \\ 66 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

All 66 slightly uneven patches were predicted as **Smooth**. The model completely fails to identify subtle road undulations, yielding an effective recall of 0.0% on uneven surfaces.

---

## 5. Auditor Conclusion & Taxonomy Assignment

The multi-task surface classifier is assigned **`VALIDATED_WEAK`**.
While the MobileNetV3-Large backbone runs extremely efficiently (~4 ms latency, negligible VRAM), training for only 5 epochs on an unweighted, non-resampled dataset composed of 99% benign highway patches forced the optimization to settle into a majority-class local minimum. 

**Corrective Action Required:**
1. Collect or re-sample balanced validation splits with explicit representation of wet, waterlogged, concrete, and unpaved roads.
2. Retrain with class-balanced cross-entropy loss or focal loss with $\alpha_c \propto 1 / N_c$.
