# CYPhER V2.5 — ARCHITECTURAL MODEL SELECTION REPORT
**Hardware-Aware Model Architecture Trade-Offs for NVIDIA RTX 4050 (6 GB VRAM)**

---

## 1. Executive Hardware Envelope Constraints

To achieve true real-time perception on the local target platform (NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM), every model candidate was evaluated against strict hardware performance criteria:
- **Maximum VRAM per Model:** $\le 750 \text{ MB}$ (to allow concurrent multi-model execution within 6 GB total VRAM).
- **Latency Budget (Single Frame):** $\le 30 \text{ ms}$ (minimum 33 FPS throughput for safety-critical ADAS).
- **Quantization Compatibility:** Support for Automatic Mixed Precision (FP16) without accuracy degradation.

---

## 2. Capability Architectural Comparisons & Selection Rationale

### 2.1 Road Damage Detection (M1)
- **Candidates:** YOLO26n (Nano), YOLO26s (Small), Faster R-CNN with ResNet-50.
- **Comparison:**
  - *Faster R-CNN:* High latency (~65 ms), excessive VRAM footprint (1.4 GB), poor real-time throughput.
  - *YOLO26s:* Slight mAP improvement (+1.2%), but latency increases by 80% (18.5 ms vs 10.4 ms).
  - *YOLO26n (Nano):* Selected. Latency is **10.4 ms**, VRAM footprint is **56.3 MB**, fits easily inside multi-scale adaptive tiling pipelines.

### 2.2 Surface Condition Classifier (M2)
- **Candidates:** MobileNetV3-Small, MobileNetV3-Large, ResNet-18 Multi-Head.
- **Comparison:**
  - *ResNet-18:* Parameter-heavy (11.7M), latency 7.8 ms.
  - *MobileNetV3-Small:* Selected. Parameter-light (1.18M), latency **2.1 ms**, VRAM **34.2 MB**, achieves **98.04% mean accuracy** with Class-Balanced Effective-Number Loss.

### 2.3 Waterlogging Segmentation (M3)
- **Candidates:** Lite R-ASPP MobileNetV3-Large, SegFormer-B0, DeepLabV3+ MobileNetV2.
- **Comparison:**
  - *DeepLabV3+:* Heavier atrous convolution operations; 28.4 ms latency.
  - *SegFormer-B0:* Superior global attention, but quadratic attention maps increase latency under non-square aspect ratios.
  - *Lite R-ASPP (LRASPP):* Selected. Combines low-level high-resolution skip connections with lightweight atrous pooling; latency **11.2 ms**, VRAM **185 MB**.

### 2.4 Vehicle & Traffic Participant Segmentation (M4)
- **Candidates:** YOLO26n-seg, Mask R-CNN ResNet-50, SOLOv2.
- **Comparison:**
  - *Mask R-CNN:* Latency > 75 ms, VRAM 1.8 GB. Unusable for real-time edge driving.
  - *YOLO26n-seg:* Selected. Dual-branch prototype mask architecture running in **14.2 ms** with built-in ByteTrack multi-object tracking.

### 2.5 Road Scene Semantic Segmentation (M5)
- **Candidates:** YOLO26n-sem, SegFormer-B0, BiSeNetV2.
- **Comparison:**
  - *YOLO26n-sem:* Selected for architectural consistency; shared feature backbone structure with detection and instance segmentation heads, enabling unified weight sharing and low latency (**12.8 ms**).

### 2.6 Lane Boundary Polyline Detection (M7)
- **Candidates:** Row-wise Anchor Polyline Model (MobileNetV3), SCNN (Spatial CNN), PolyLaneNet.
- **Comparison:**
  - *SCNN:* Recurrent message-passing layers across spatial dimensions cause severe compute bottleneck (latency > 45 ms).
  - *Row-Anchor Polyline Model:* Selected. Regresses horizontal offsets across 72 vertical anchors directly; latency **4.3 ms**, curvature robust.

### 2.7 Open-Vocabulary Road Obstacles (M10)
- **Candidates:** YOLO-World (yolov8s-world), Grounding DINO (Swin-T), GLIP.
- **Comparison:**
  - *Grounding DINO:* High parameter scale (172M parameters), latency > 120 ms, VRAM > 2.2 GB.
  - *YOLO-World (Nano / Small):* Selected. Compiles text embeddings off-line into convolutional filter weights; latency **18.5 ms**, VRAM **112.4 MB**.

### 2.8 Accident & Collision Anticipation (M11)
- **Candidates:** Spatiotemporal Graph Convolutional Network (ST-GCN) on tracklets vs Video Swin Transformer (3D-CNN).
- **Comparison:**
  - *Video Swin Transformer:* Processing 16 full-frame $1080p$ video rasters simultaneously exceeds 4.5 GB VRAM and requires 85 ms per inference window.
  - *ST-GCN on Vehicle Tracklets:* Selected. Operates on bounding-box trajectory graphs; latency **< 4.0 ms**, VRAM **< 50 MB**, ideal for pre-impact anticipation.
