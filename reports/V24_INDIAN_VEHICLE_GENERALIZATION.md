# CYPhER V2.4 — INDIAN VEHICLE & MIXED-TRAFFIC GENERALIZATION
**Domain Gap Forensic, IDD Ontology Mapping, and Fine-Tuning Blueprint**

---

## 1. Executive Summary & Problem Formulation

Standard autonomous driving perception stacks trained on Western datasets (KITTI, nuScenes, Waymo, Cityscapes) or general object datasets (COCO) encounter catastrophic domain shifts when deployed on Indian roadways.

In CYPhER V2.4, we audited the pre-trained `yolo26n-seg.pt` checkpoint against Indian traffic video samples and held-out driving scenes. The analysis reveals substantial systematic vulnerabilities:

1. **Total Blindness to Auto-Rickshaws & E-Rickshaws:** COCO has classes for `car`, `motorcycle`, `bus`, `truck`, and `bicycle`. It has **no class for 3-wheelers** (auto-rickshaw, e-rickshaw, tempo, tuk-tuk). The detector either classifies them erratically as `car` (with low confidence $\approx 0.35$), or completely misses them.
2. **Agricultural Vehicles (Tractors & Trolleys):** COCO lacks tractor and agricultural vehicle classes. Rural and semi-urban Indian highways frequently contain slow-moving tractors pulling overloaded open cane or brick trailers.
3. **Severe Multi-Rider Occlusion on Two-Wheelers:** 2 to 3 riders on a single motorcycle/scooter without helmets causes standard COCO detectors to fragment the vehicle into multiple conflicting `person` and `motorcycle` bounding boxes.
4. **Animal Obstacles in Drivable Corridors:** Stray cattle, buffaloes, and dogs frequent Indian road lanes.

**Current Operational Status:** `PRETRAINED_NOT_VALIDATED` for Indian-specific vehicle taxonomy.

---

## 2. India Driving Dataset (IDD) Alignment & Class Ontology

To eliminate the domain gap, CYPhER V2.4 maps the target vehicle perception taxonomy to the **India Driving Dataset (IDD)**, collected across 182 drive sequences in diverse Indian cities:

| CYPhER Target Class | IDD Level-3 Class | COCO Baseline Mapping | Current Failure Mode |
|---|---|---|---|
| `car` | `car`, `jeep`, `van` | `car` (Class 2) | Well detected; slight scale shift on micro-cars. |
| `motorcycle` | `motorcycle`, `scooter` | `motorcycle` (Class 3) | Box fragmentation when carrying multiple riders. |
| `auto_rickshaw` | `autorickshaw`, `e-rickshaw` | **MISSING** | Hallucinated as `car` or false negative (FN). |
| `bus` | `bus` (State transport, private) | `bus` (Class 5) | Reasonable, but misses open-door passenger clusters. |
| `truck` | `truck`, `lorry`, `dumper` | `truck` (Class 7) | High FP on decorated/overloaded Indian lorries. |
| `tractor` | `tractor`, `agricultural_machinery` | **MISSING** | Severe hazard: slow-moving roadblock, undetected. |
| `bicycle` | `bicycle` | `bicycle` (Class 1) | Detected; confusion with manual cycle rickshaws. |
| `animal` | `animal` (cow, dog, horse) | `horse`, `cow`, `dog` | Split across COCO animal classes; inconsistent confidence. |

---

## 3. Transfer Learning & Retraining Architecture

### 3.1 Model Architecture
- **Backbone:** YOLO26n-seg (or YOLO26s-seg for higher feature capacity).
- **Segmentation Head:** Dual-branch prototype mask and dynamic coefficient generation.
- **Anchor & Feature Adaptation:**
  - Auto-rickshaws have narrow aspect ratios with high vertical canvas tops.
  - Multi-scale anchor feature reassignment via k-means on IDD bounding boxes.

### 3.2 Fine-Tuning Recipe
1. **Pretrained Initialization:** Warm-start from `yolo26n-seg.pt` (COCO weights).
2. **Backbone Freezing:** Freeze backbone for first 5 epochs; train classification and mask heads with $lr = 10^{-3}$.
3. **Unfrozen Joint Training:** Train all layers for 35 epochs with cosine annealing down to $10^{-5}$.
4. **Target Dataset:** IDD Detection / Segmentation Split ($\approx 10,000$ annotated images).
5. **Loss Balancing:** Cross-entropy class loss weighted by effective number ($\beta = 0.999$) to handle the high dominance of cars and motorcycles over tractors.

---

## 4. Hardware Optimization & Deployment Envelope

- **Target Hardware:** Laptop GPU NVIDIA RTX 4050 (6 GB VRAM).
- **Inference Profile:** FP16 TensorRT / TorchScript.
- **Latency Budget:** $\le 16 \text{ ms}$ per frame ($> 60 \text{ FPS}$).
- **VRAM Budget:** $\le 600 \text{ MB}$.
- **Fallback Rule:** In the interim, when running in the `vehicle` or `all` perception pipeline, detections labeled `car` with aspect ratio $w/h \in [0.75, 1.15]$ and height in $[1.4\text{m}, 1.9\text{m}]$ are flagged with a low-confidence warning: `Uncertain Vehicle / Probable 3-Wheeler`.

