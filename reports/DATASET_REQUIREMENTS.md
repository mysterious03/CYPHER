# DATASET REQUIREMENTS & AUDIT SPECIFICATION

## 1. Overview of Required Datasets

| Capability | Primary Dataset | Format | Annotations | License / Restrictions | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Surface Friction, Material, Unevenness** | **RSCD** (Tsinghua RSXD) | Patch Images (360×240) | Friction (6), Material (6), Unevenness (4) | Non-commercial (CC BY-NC) | Specification Ready (Phase 1) |
| **Waterlogging & Flooding** | **R²S100K** | 1080p Road Frames | Pixel Masks (Puddle, Flooded, Wet) | Academic / Research | Specification Ready (Phase 1) |
| **Adverse Weather** | **ACDC** | 1080p Adverse Frames | Fog, Night, Rain, Snow (Pixel/Image) | Non-commercial Academic | Phase 2 |
| **Lane & Markings** | **BDD100K** | 720p Frames | Lane Lines, Drivable Area, Signals | BSD / Commercial check | Phase 2 |
| **Accident Recognition** | **ACCIDENT** | Video Clips | Temporal timestamps & collision types | Academic Benchmark | Phase 4 |

---

## 2. Phase 1 Datasets Deep-Dive

### 2.1 RSCD (Road Surface Condition Dataset)
* **Source:** [https://thu-rsxd.com/rscd/](https://thu-rsxd.com/rscd/)
* **Volume:** ~1,000,000 road surface image patches (typically cropped patches of 360 × 240 pixels).
* **Target Label Taxonomy:**
  1. **Friction-Related Condition (6 classes):**
     - `0: Dry`
     - `1: Wet`
     - `2: Water-covered`
     - `3: Fresh snow`
     - `4: Melted snow`
     - `5: Ice-covered`
  2. **Road Material (6 classes):**
     - `0: Asphalt`
     - `1: Concrete`
     - `2: Gravel`
     - `3: Dirt / Unpaved`
     - `4: Mud`
     - `5: Loose stone / Cobblestone`
  3. **Surface Unevenness (4 classes):**
     - `0: Smooth`
     - `1: Slightly uneven`
     - `2: Severely uneven / rutted`
     - `3: Deteriorated / broken pavement`

### 2.2 R²S100K (Road Region & Distress Dataset)
* **Source:** [https://r2s100k.github.io/](https://r2s100k.github.io/) / GitHub: [https://github.com/r2s100k/data](https://github.com/r2s100k/data)
* **Volume:** ~14,000 pixel-annotated frames + 86,000 unlabelled frames.
* **Target Classes for Water Segmentation:**
  - `Standing water / Large flood`
  - `Water puddle`
  - `Wet road reflection`
  - `Water-covered pothole cavity`

---

## 3. Storage Budget & Hardware Verification
- **Free Disk Space:** **496.6 GB** available (Dataset storage budget: ~15 GB for Phase 1 splits).
- **VRAM Footprint:** Patch-based multi-task classifier consumes ~0.3 GB VRAM during training at batch size 32.
- **Data Integrity Rule:** Train/val/test splits strictly separated with SHA-256 / dHash deduplication to eliminate data leakage.

