# EXPANSION ARCHITECTURE: MULTI-MODEL ROAD PERCEPTION SYSTEM

## 1. Executive Summary
This architectural document outlines the expansion of the existing Road Intelligence project into an enterprise-grade, multi-task road perception and safety intelligence system. While the baseline system reliably detects localized structural distress (cracks and potholes via custom fine-tuned YOLO26n), this expansion equips the vehicle with comprehensive situational awareness covering:
1. **Surface Friction-Condition Classification** (RSCD)
2. **Road Material Classification** (RSCD)
3. **Road Unevenness & Roughness Classification** (RSCD)
4. **Waterlogging & Flood Segmentation** (R²S100K)
5. **Adverse Weather & Environmental Perception** (ACDC)
6. **Lane & Road-Marking Geometry** (BDD100K)
7. **Open-Vocabulary Road Obstacle Detection** (Grounding DINO / SAM 2)
8. **Traffic Signal State Recognition** (BDD100K)
9. **Video Accident & Collision Event Recognition** (ACCIDENT)

---

## 2. Multi-Model Perception Pipeline & Data Flow

```mermaid
flowchart TD
    subgraph Inputs
        RGB["Raw RGB Road Image / Video Frame"]
    end

    subgraph CorePerception["Level 1: Existing Road Perception (Preserved)"]
        YOLO_DET["Custom YOLO26n (best.pt)<br/>Potholes & Cracks (D00, D10, D20, D40)"]
        YOLO_SEG["YOLO26n-seg<br/>Vehicle & Moving Traffic Masks"]
        YOLO_SEM["YOLO26n-sem<br/>Drivable Asphalt vs Sidewalk/Sky"]
        YOLO_DEP["YOLO26n-depth<br/>Monocular Relative Depth Map"]
    end

    subgraph ExpansionP1["Level 2: Surface & Water Perception (Phase 1)"]
        SURF_MT["Multi-Task Surface Backbone (MobileNetV3-Large)<br/>Head 1: Friction (Dry, Wet, Water-covered, Snow, Ice)<br/>Head 2: Material (Asphalt, Concrete, Gravel, Dirt)<br/>Head 3: Unevenness (Smooth, Slight, Severe, Rutting)"]
        WATER_SEG["Water & Flooding Segmenter (SegFormer/YOLO-Seg)<br/>Standing Water, Puddle, Flooded Lane Masks"]
    end

    subgraph ExpansionP2["Level 3: Environmental & Structural Perception (Phase 2 & 3)"]
        WEATHER["Weather Classifier (ACDC)<br/>Clear, Rain, Fog, Night, Snow"]
        LANE_GEOM["Lane & Marking Segmenter (BDD100K)<br/>Boundaries, Crossing, Markings"]
        SIGNAL["Traffic Light State Tracker<br/>Red, Amber, Green, Unknown"]
        OBSTACLE["Open-Vocabulary Obstacle Detector (On-Demand)<br/>Debris, Rocks, Animals, Cargo"]
    end

    subgraph DecisionEngine["Level 4: Multi-Model Decision Controller & Scheduler"]
        ROUTER["Task-Aware Dynamic Router<br/>(Resource & VRAM Governor)"]
        FUSION["Perception Evidence Fusion<br/>(Cross-Check Waterlogged Potholes & Hazards)"]
        UNCERTAINTY["Reliability & Uncertainty Evaluator<br/>(Separates Visual Perception from Metric Dynamics)"]
    end

    subgraph Output["Level 5: Structured Perception Output"]
        OUT["Validated Road Safety & Surface Assessment<br/>(80%+ High-Confidence Reliability Gate)"]
    end

    RGB --> CorePerception
    RGB --> ExpansionP1
    RGB --> ExpansionP2

    CorePerception --> DecisionEngine
    ExpansionP1 --> DecisionEngine
    ExpansionP2 --> DecisionEngine

    DecisionEngine --> Output
```

---

## 3. Hardware Constraints & VRAM Scheduling (RTX 4050, 6 GB)
To guarantee strict compliance with the **6.0 GB VRAM limit**:
- **Sequential Pipeline Execution:** Models are loaded and executed sequentially or in small decoupled micro-batches with explicit CUDA cache clearing (`torch.cuda.empty_cache()`, `gc.collect()`).
- **Shared Backbone Efficiency:** Surface friction, road material, and unevenness are unified into a single shared **MobileNetV3-Large Multi-Task Network** (5.48M parameters, ~0.15 GB VRAM).
- **On-Demand Activation:** Heavy modules (such as open-vocabulary obstacle detection or accident video recognition) are triggered conditionally via confidence thresholds rather than running continuously on every frame.

---

## 4. Physical & Scientific Disclaimers
1. **Friction:** Visual classification predicts apparent surface wetness/icing categories (`visual_surface_condition`). It does not compute the physical tyre-road friction coefficient ($\mu$), which requires CAN-bus wheel-slip dynamics.
2. **Roughness:** Visual classification detects surface distress and rutting categories (`visual_unevenness`). It does not compute metric International Roughness Index (IRI), which requires calibrated profilometry or IMU accelerometry.
3. **Pothole Depth:** Monocular depth outputs relative depression ratios. Metric centimeter depths are only asserted after physical calibration against known optical geometry.
4. **Water vs Shadow:** Dark road regions undergo dual-validation against semantic drivable masks and photometric specular reflection analysis to prevent misclassifying tree shadows as water puddles.

