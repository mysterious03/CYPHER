# CYPhER V2.4 — WATERLOGGING SEGMENTATION TRAINING SPECIFICATION
**Model Architecture, Loss Formulation, and Training Recipe**

---

## 1. Network Architecture Specification

To balance segmentation accuracy with real-time inference on edge and laptop hardware (NVIDIA RTX 4050, 6 GB VRAM), CYPhER V2.4 selects:

- **Backbone:** `MobileNetV3-Large` (pretrained on ImageNet-1K).
- **Segmentation Head:** `Lite Reduced Atrous Spatial Pyramid Pooling` (LRASPP).
  - Lightweight dual-branch design combining low-level high-resolution feature maps with deep atrous contextual pooling.
  - Total Parameters: $\approx 3.22\text{M}$ parameters.
  - Expected FP16 Latency: $< 12 \text{ ms}$ on RTX 4050 ($> 80 \text{ FPS}$).
  - VRAM Consumption: $< 350 \text{ MB}$.

---

## 2. Loss Function Formulation for Waterlogging

Waterlogging detection suffers from high class imbalance and blurry boundary transitions between damp asphalt and standing water. The specified loss function is a hybrid compound loss:

$$\mathcal{L}_{\text{water}} = \alpha \mathcal{L}_{\text{Focal}} + \beta \mathcal{L}_{\text{Dice}} + \gamma \mathcal{L}_{\text{Boundary}}$$

1. **Focal Cross-Entropy Loss ($\mathcal{L}_{\text{Focal}}$):**
   $$\mathcal{L}_{\text{Focal}} = - (1 - p_t)^\gamma \log(p_t)$$
   Down-weights well-classified dry road background pixels, focusing gradients on subtle puddle edges.
2. **Soft Dice Loss ($\mathcal{L}_{\text{Dice}}$):**
   $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i + \epsilon}{\sum_i p_i^2 + \sum_i y_i^2 + \epsilon}$$
   Directly optimizes the intersection-over-union metric, preventing small puddles from being suppressed by vast non-water road surfaces.
3. **Hyperparameters:** $\alpha = 0.5$, $\beta = 1.0$, $\gamma = 2.0$.

---

## 3. Training & Data Augmentation Recipe

### Augmentation Pipeline (Water-Specific)
- **Random Crop & Scale:** Jitter scale between 0.8 and 1.2, random crop $512 \times 512$.
- **Photometric Distortions:**
  - Random contrast adjustment ($[0.7, 1.3]$) to simulate overcast vs direct sun.
  - Hue/Saturation jittering to represent turbid/muddy water variations.
- **Simulated Glare & Shadow Injection:**
  - Random linear polygonal mask darkening ($\alpha \in [0.4, 0.7]$) to teach shadow invariance.
  - Specular bright spots (Gaussian additive blobs) to prevent false positives on sun glare.

### Optimization Recipe
- **Batch Size:** 16 (gradient accumulation steps = 2, effective batch = 32).
- **Optimizer:** AdamW with initial learning rate $3 \times 10^{-4}$ and cosine annealing schedule to $1 \times 10^{-6}$ over 50 epochs.
- **Precision:** Mixed Precision (`torch.cuda.amp.autocast`).

---

## 4. Operational Gate

> [!CAUTION]
> **Training Gate Activated:**
> Consistent with project directives, training will not be executed using heuristic pseudo-masks. The training pipeline script will remain staged until authentic human-annotated datasets are downloaded and validated.

