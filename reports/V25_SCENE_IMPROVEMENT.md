# CYPhER V2.5 — ROAD SCENE SEMANTIC SEGMENTATION IMPROVEMENT
**Cityscapes Domain Gap Forensic, IDD Semantic Ingestion, and Drivable Corridor Segmentation**

---

## 1. Executive Summary & Domain Shift Forensic

In modern autonomous road perception, semantic scene segmentation provides the foundational drivable spatial prior that bounds all other hazard detectors. In CYPhER V2.5, capability `M5_SCENE_SEM` employs `yolo26n-sem.pt` (7.0 MB).

### The Cityscapes Domain Gap on Indian Roadways:
The baseline model was pretrained on Cityscapes, reflecting European urban road architectures:
1. **Paved Curbs vs Unpaved Shoulders:** Cityscapes assumes every drivable asphalt surface terminates at an elevated concrete curb or manicured sidewalk. On Indian arterial and rural state corridors, roads transition directly into unpaved dirt shoulders, gravel berms, or open monsoon gutters. The model produces severe boundary oscillations ($\pm 35 \text{ cm}$ variance) at the asphalt border.
2. **Terrain Inconsistency:** Dried mud, sandy road patches, and unpaved bypass detours are falsely classified as non-drivable `terrain` or `building`, creating false vehicle stop gates.
3. **Vegetation Overhang:** Tropical tree boughs (neem, banyan, coconut palms) overhanging the road corridor cause false `vegetation` occlusion boundaries in the upper driving perspective.

---

## 2. Ingestion Strategy for India Driving Dataset (IDD)

To eliminate the domain gap, CYPhER V2.5 establishes the semantic mapping to the **India Driving Dataset (IDD)**:

### 2.1 Class Ontology Alignment

| Target Drivable Semantic Class | Cityscapes Baseline Mapping | IDD Level-2 / Level-3 Target Class | Function in CYPhER Fusion |
|---|---|---|---|
| `drivable_paved` | `road` (Class 0) | `road`, `asphalt`, `concrete` | Primary high-speed corridor |
| `drivable_unpaved` | **MISSING** (falsely `terrain`) | `unpaved_road`, `gravel_track` | Secondary slow-speed drivable |
| `road_shoulder` | `sidewalk` (Class 1) | `shoulder`, `berm`, `curb` | Lateral safety boundary margin |
| `barrier_divider` | `fence` / `wall` (Classes 3, 4) | `concrete_median`, `guardrail` | Collision avoidance boundary |
| `vegetation_adjacent` | `vegetation` (Class 8) | `overhang_trees`, `hedges` | Optical occlusion masking |
| `sky_horizon` | `sky` (Class 10) | `sky` | Horizon line vanishing-point prior |

---

## 3. Network Architecture & Evaluation Criteria

### 3.1 Model Architecture
- **Architecture:** `YOLO26n-sem` (or SegFormer-B0 for pure transformer spatial attention).
- **Resolution:** $640 \times 640$ native inference.
- **Latency Budget:** $\le 14 \text{ ms}$ on NVIDIA RTX 4050 GPU ($> 70 \text{ FPS}$).
- **VRAM Footprint:** $\le 180 \text{ MB}$.

### 3.2 Evaluation Metrics Specification
- **Drivable Surface IoU:**
  $$\text{IoU}_{\text{drivable}} = \frac{\text{TP}_{\text{road}}}{\text{TP}_{\text{road}} + \text{FP}_{\text{road}} + \text{FN}_{\text{road}}}$$
- **Boundary F1-Score ($\mathcal{BF}$):** Measures edge alignment accuracy at the asphalt-to-shoulder transition.
- **Current Operational Status:** **`PRETRAINED_NOT_VALIDATED`** until full training on IDD semantic polygons.
