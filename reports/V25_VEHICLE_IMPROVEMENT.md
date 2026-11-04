# CYPhER V2.5 — VEHICLE & MIXED-TRAFFIC PARTICIPANT IMPROVEMENT
**MS COCO Domain Shift Audit, Three-Wheeler Detection, and Multi-Rider Handling**

---

## 1. Executive Summary & Diagnostic Breakdown

In Indian road scenes, mixed traffic dynamics depart radically from Western driving assumptions. Autonomous perception must track high-speed cars alongside slow-moving agricultural machinery, three-wheelers, and vulnerable road users.

Capability `M4_VEHICLE_SEG` currently runs `yolo26n-seg.pt` (7.1 MB), pretrained on MS COCO.

### 1.1 The Critical Class Blind Spots:
1. **Total Absence of Auto-Rickshaws & E-Rickshaws:** MS COCO possesses no class for 3-wheelers. Auto-rickshaws, tempos, and tuk-tuks represent up to 25% of vehicles in Indian metropolitan traffic. The detector either hallucinates them as `car` (at low confidence $\approx 0.32$), or completely misses them.
2. **Agricultural Tractors and Open Trailers:** Heavy tractors towing unlit, overloaded trailers of sugar cane or bricks are common on Indian highways. COCO classifies them erratically as `truck` or fails entirely.
3. **Multi-Rider Two-Wheelers:** In India, motorcycles and scooters frequently carry two or three passengers without helmets. COCO detectors suffer severe box fragmentation, generating overlapping, conflicting bounding boxes for each person and the motorcycle frame, leading to tracker ID switches and trajectory instability.
4. **Stray Cattle and Animals:** Cows, bullocks, and dogs on drivable highways require immediate detection for emergency braking.

---

## 2. Ingestion Strategy & Class Ontology Harmonization (IDD Level-3)

CYPhER V2.5 establishes the mapping between target Indian traffic participants and the **India Driving Dataset (IDD)**:

| CYPhER Target Class | COCO Baseline Status | IDD Level-3 Mapping | Target Recall Goal |
|---|---|---|---|
| `car` | Operational | `car`, `jeep`, `van` | $\ge 85.0\%$ |
| `motorcycle` | Severe box fragmentation | `motorcycle`, `scooter` | $\ge 80.0\%$ |
| `auto_rickshaw` | **MISSING (Fatal Blind Spot)** | `autorickshaw`, `e-rickshaw` | $\ge 75.0\%$ |
| `tractor` | **MISSING (Slow Obstacle)** | `tractor`, `agricultural_trolley` | $\ge 70.0\%$ |
| `bus` | Operational | `bus` (State transport, private) | $\ge 85.0\%$ |
| `truck` | Operational (high FP on lorries) | `truck`, `lorry`, `dumper` | $\ge 80.0\%$ |
| `bicycle` | Operational | `bicycle`, `cycle_rickshaw` | $\ge 65.0\%$ |
| `person` | Operational | `pedestrian` | $\ge 75.0\%$ |
| `animal` | Inconsistent | `cow`, `buffalo`, `dog`, `horse` | $\ge 75.0\%$ |

---

## 3. Retraining Recipe & Tracking Architecture

### 3.1 Model Architecture & Anchor Recalibration
- **Architecture:** `YOLO26n-seg` (Dual-branch box detection and prototype mask segmentation).
- **Aspect Ratio Optimization:** Three-wheelers have vertical aspect ratios ($w/h \in [0.75, 1.10]$) distinct from standard low-profile passenger cars ($w/h \in [1.25, 1.80]$). Anchor k-means clustering on IDD bounding boxes ensures accurate candidate generation.

### 3.2 Loss Function & Class Balancing
- Class-Balanced Effective-Number Loss ($\beta = 0.999$) to handle the heavy imbalance between cars/two-wheelers and tractors/animals.
- Deep supervision on prototype masks to ensure crisp boundary separation between riders and motorcycle frames.

### 3.3 ByteTrack Multi-Object Tracking Configuration
- Dual-threshold association: high-confidence detections ($\tau_{\text{high}} = 0.50$) matched first via Kalman filter motion vectors; low-confidence detections ($\tau_{\text{low}} = 0.15$) matched second to maintain track continuity across partial occlusions.
- Current Status: **`PRETRAINED_NOT_VALIDATED`** until full fine-tuning on IDD is completed.
