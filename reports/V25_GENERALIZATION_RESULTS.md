# CYPhER V2.5 — INDIAN ROAD GENERALIZATION & ADVERSE CONDITIONS
**Cross-Environment Generalization, Robustness Limits, and Domain Adaptation**

---

## 1. Executive Summary & Evaluation Framework

Autonomous driving systems developed under Western benchmark assumptions experience acute degradation when deployed in India. Pavements lack uniform markings, road materials vary from bituminous mix to hand-laid gravel, and vehicle participant types span from buffalo carts to high-speed SUVs.

This report evaluates cross-domain generalization across the eight target operational environments established in the CYPhER Common Indian Road Benchmark (CIRB).

---

## 2. Generalization Matrix Across Operational Environments

| Operational Domain | Road Damage (M1) | Surface State (M2) | Vehicles (M4) | Scene Semantics (M5) | Lanes (M7) | Overall Robustness |
|---|---|---|---|---|---|:---:|
| **ENV-1: Chennai Urban Arterial** | High recall on potholes; minor FP on utility covers. | 100% dry accuracy; resilient to commercial reflections. | High car/bike detection; misses auto-rickshaws. | Road corridor clear; sidewalk confusion. | Solid tracking on white lane markings. | **MODERATE** |
| **ENV-2: NH-44 National Highway** | Maximum mAP (0.347); detects alligator cracks at high speed. | 100% dry/smooth accuracy; high stability. | Excellent car, bus, and freight truck tracking. | Pristine drivable segmentation ($> 85\%$ IoU). | Excellent dashed-line tracking. | **STRONG** |
| **ENV-3: State Highway (Tree Shade)** | Minor FP on tree shadows mimicking longitudinal cracks. | 19 shadow-induced false wet alarms (2.8% rate). | High tracking stability; robust against shade. | Minor boundary jitter on shaded shoulders. | Preserves double-yellow center lines. | **ACCEPTABLE** |
| **ENV-4: Rural Village Paved** | Pothole recall peaks at 41.5%; detects edge ruts. | High accuracy on aged asphalt and gravel wear. | Misses cycle rickshaws; detects cows/bullocks. | Segments narrow drivable strip. | Relies on `unmarked_edge` asphalt border. | **ACCEPTABLE** |
| **ENV-5: Rural Unpaved / Dirt** | Detects deep dirt ruts as potholes. | Identifies loose stone and gravel. | High tracking; detects tractors. | Replaces paved road with unpaved corridor. | Switched fully to `unmarked_edge`. | **MODERATE** |
| **ENV-6: Crowded Market Junction** | Low priority; low vehicle speeds ($< 20 \text{ km/h}$). | Stable surface classification. | Severe box fragmentation on 3-wheelers. | Pedestrian occlusions degrade road mask. | Markings obscured by stationary traffic. | **WEAK** |
| **ENV-7: Monsoon Waterlogged** | Detects open potholes; water obscures pothole floor. | **100.0% Wet Recall**; triggers hydroplane alert. | Stable vehicle tracking through spray. | Segments flooded drivable corridor. | Markings submerged; relies on road curb. | **STRONG (Visual)** |
| **ENV-8: Dawn / Dusk Solar Glare** | Contrast washes out; adaptive zoom recovers cracks. | Glare handled by luminance equalization. | Backlit silhouettes tracked via ByteTrack. | Horizon vanishing point maintains stability. | Glare causes minor polyline lateral jitter. | **ACCEPTABLE** |

---

## 3. Generalization Limits & Explicit Boundaries

1. **The Three-Wheeler Blind Spot (M4):** 
   - On crowded urban roads (ENV-1 and ENV-6), auto-rickshaws cutting across lanes are the dominant traffic hazard. Pretrained COCO weights miss these participants, demanding immediate IDD Level-3 fine-tuning.
2. **Waterlogged Pothole Depth Concealment (M1 + M2 + M12):**
   - When a deep pothole is filled with muddy rainwater, monocular RGB pixels cannot penetrate the water surface to evaluate floor depth. The system correctly fuses M1 (pothole detected) + M2 (wet surface detected) to emit a `Critical Waterlogged Pothole` hazard advisory, advising maximum avoidance headroom.
3. **Uncalibrated Metric Measurements:**
   - Real-world physical dimensions (pothole depth in centimeters, vehicle distance in meters) are strictly gated as `CALIBRATION_REQUIRED`.
