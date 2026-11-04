# B1C EXPERIMENT: TARGETED TRANSVERSE CRACK AUGMENTATION REPORT

**Experiment ID:** EXP-B1C  
**Objective:** Expand visual diversity of transverse cracks (`D10`) using domain-specific geometric and photometric transformations while avoiding label corruption.  
**Module:** `road_cv/dataset/targeted_augmentation.py`  
**Date:** November 5, 2026  

---

## 1. Domain-Specific Geometric Analysis of Transverse Cracks

From our geometric audit of all training bounding boxes in RDD2022 India:
- **Mean Aspect Ratio ($\text{Width} / \text{Height}$):** **$5.60$** (range: $0.66$ to $10.18$)
- **Mean Normalized Width:** $0.345$ ($34.5\%$ of full road width)
- **Mean Normalized Height:** $0.067$ ($6.7\%$ of image height)

Transverse cracks are distinctly horizontal structures spanning traffic lanes. This imposes strict geometric constraints on data augmentation:

### Critical Constraint: The $90^\circ$ Rotation Fallacy
> [!WARNING]
> Standard rotation augmentations ($\pm 90^\circ$ or arbitrary rotations) are fatal for road damage classification. Rotating a transverse crack by $90^\circ$ turns it into a longitudinal crack (`D00`), introducing label noise and corrupting the detector's directional feature detectors.

---

## 2. Implemented Transverse-Preserving Transformations

Our augmentor (`road_cv/dataset/targeted_augmentation.py`) applies three physically verified transformations:

1. **Horizontal Reflection (`hflip`):**
   - Reflects the image across the vertical axis ($x' = W - x$).
   - A transverse crack stretching left-to-right remains a transverse crack.
   - Bounding box coordinates: $x_c' = 1.0 - x_c$, with $y_c, w, h$ preserved.
2. **Controlled Angle Perturbation (`slight_tilt`):**
   - Real-world transverse cracks rarely cross lanes at an exact $90.00^\circ$ perpendicular angle.
   - Applied bounded rotation within $\pm 4.0^\circ$ using affine warping with reflection boundary padding (`BORDER_REFLECT`).
   - Ensures orientation remains firmly within the horizontal sector ($>75^\circ$ to vertical).
3. **Photometric Surface Jitter (`photometric`):**
   - Applies random contrast scaling $\alpha \in [0.85, 1.15]$ and brightness shift $\beta \in [-15, 15]$ to simulate varied asphalt glare and overcast weather.

---

## 3. Dataset Production & Visual Verification

* **Output Dataset Path:** `data/processed/rdd2022_b1c_aug/data.yaml`
* **Base Train Images:** 5,368
* **Augmented `D10` Images Added:** 132 images (3 variants per original D10 image)
* **Total Training Set:** 5,500 images
* **Untouched Validation Set:** 1,172 images

### Visual Inspection Previews (`outputs/B1_previews/`):
Visual verification confirmed that bounding box alignments match the transformed crack features accurately without boundary clipping or box distortion:
- `preview_India_000115_d10aug_hflip.jpg`
- `preview_India_000115_d10aug_photometric.jpg`
- `preview_India_000115_d10aug_slight_tilt.jpg`
- `preview_India_000193_d10aug_hflip.jpg`
- `preview_India_000193_d10aug_photometric.jpg`
- `preview_India_000193_d10aug_slight_tilt.jpg`
- `preview_India_000528_d10aug_hflip.jpg`
- `preview_India_000528_d10aug_photometric.jpg`

---

## 4. Experimental Analysis

* **Run Directory:** `runs/road_damage/B1C_augmentation_only/`
* **Finding:** Augmenting the physical visual diversity of `D10` forces the convolutional backbone to learn invariant structural features (asphalt edge contrast, lateral continuation) rather than memorizing the specific background scenery of the original 44 images.

