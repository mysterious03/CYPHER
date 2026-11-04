# CYPhER DATASET INVENTORY
## Local and Target Datasets for Multi-Model Road Perception

**Document Version:** 1.0  
**Audit Date:** November 5, 2026  
**Auditor:** Senior Computer Vision Engineer & Dataset Architect  

---

## 1. Discovered Local Datasets on Disk

| Dataset Identifier | Local Directory Path | File Format | Train Split | Validation Split | Test Split | Primary Annotation Task | Verified Integrity |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`rdd2022_india`** | `data/processed/rdd2022_india/` | YOLO `.txt` / `.jpg` | 5,368 | 1,172 | 1,166 | Road cracks (`D00, D10, D20`) and potholes (`D40`) | **100% Verified** |
| **`rscd_surface`** | `data/processed/rscd_surface/` | JSON Patch Index | 5,368 | 1,172 | - | Multi-task surface condition, material, unevenness | **100% Verified** |
| **`hard_negatives`**| `data/hard_negatives/` | JPEG crops / JSON | 19 | - | - | Difficult clean asphalt background patches | **100% Verified** |
| **`water_authentic`**| `data/processed/water_authentic/` | PNG Masks / JPEG | Prepared | Prepared | - | Genuine pixel-level road water and flood masks | **Pipeline Ready** |
| **`rdd2022_b1a`** | `data/processed/rdd2022_b1a_sampling/` | YOLO `.txt` / `.jpg` | 5,500 | 1,172 | - | $4\times$ replication of rare $D10$ images | **100% Verified** |
| **`rdd2022_b1c`** | `data/processed/rdd2022_b1c_aug/` | YOLO `.txt` / `.jpg` | 5,500 | 1,172 | - | Targeted horizontal flip & tilt augmentation | **100% Verified** |
| **`rdd2022_b1d`** | `data/processed/rdd2022_b1d_combined/` | YOLO `.txt` / `.jpg` | 5,519 | 1,172 | - | Sampling + targeted aug + 19 hard negatives | **100% Verified** |

---

## 2. Target External Datasets Specification

| Target Dataset | Domain / Capability | Official License | Annotation Format | Volume Needed | Priority | Ingestion Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **R²S100K** | Road Surface & Waterlogging | Academic / Non-Commercial | Semantic segmentation PNG masks (4 classes) | 1,500 images | **P0** | Ingestion pipeline initialized in `data/processed/water_authentic/` |
| **ACDC** | Adverse Driving Conditions | Creative Commons BY-NC-SA 4.0 | Multi-class weather (Fog, Rain, Snow, Night) | 2,000 images | **P0** | Ingestion pipeline ready |
| **BDD100K** | Vehicles, Lanes, Traffic Lights | BSD 3-Clause | 2D Bounding Boxes, Lane Polylines | Subset (5,000 frames) | **P1** | Pretrained foundation models currently active |
| **CADP / DAD** | Dashcam Accident Videos | Academic Research | Video clips (16-50 frames) with crash onset frame | 1,000 video clips | **P1** | Heuristic temporal tracking active; video net planned |
| **TuSimple / CULane**| Lane Markings | Academic Free | Polyline lane coordinate arrays | 3,000 frames | **P1** | Semantic road edge extraction active |
| **Bosch Small Lights**| Traffic Signal State | Academic | Bounding boxes with light color states | 2,500 crops | **P2** | YOLO crop + HSV hue classifier active |
| **Lost & Found** | Open-Vocabulary Obstacles | Research | Object masks for small road hazards | 1,000 images | **P2** | Open-vocabulary text prompt detector active |

---

## 3. Dataset Integrity and Leakage Prevention

1. **Untouched Validation Split:**
   - The 1,172 images in `data/processed/rdd2022_india/images/val/` are strictly isolated as an untouchable holdout evaluation benchmark.
   - All experimental iterations (B1A, B1B, B1C, B1D) maintain this exact validation split with zero image or annotation contamination.
2. **Negative Patch Provenance:**
   - All 19 hard-negative background patches were verified to originate from `Road.jpg` (an unlabelled external photo). Zero patches originate from the validation split.
3. **No Heuristic Labeling Policy:**
   - All future training on Workstream C (Waterlogging) and Workstream D (Weather) strictly forbids synthetic threshold pseudo-masks. Only genuine human-annotated ground-truth labels are accepted.

