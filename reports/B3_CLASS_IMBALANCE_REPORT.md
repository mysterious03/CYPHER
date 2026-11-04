# Experiment B3: Class Imbalance Mitigation Strategy

## 1. Distribution Analysis

| Class Code | Class Name | Total Instances | Imbalance Ratio | Computed Weight |
|---|---|---:|---:|---:|
| `D00` | `D00_longitudinal_crack` | 1555 | 22.87x | **0.1552** |
| `D10` | `D10_transverse_crack` | 68 | 1.0x | **3.5494** |
| `D20` | `D20_alligator_crack` | 2021 | 29.72x | **0.1194** |
| `D40` | `D40_pothole` | 1372 | 20.18x | **0.1759** |

## 2. Key Findings
- **Severe Deficit in Transverse Cracks (`D10`):** Only 68 instances exist in the entire dataset (1.35% of all boxes), compared to 2,021 alligator cracks (`D20`).
- **Impact on B0:** In standard fine-tuning without class weighting, D10 exhibits low initial recall due to gradient dominance from D20 and D00.

## 3. Recommended Interventions
1. **Loss Reweighting:** Scale per-class classification loss using the computed weights (`D10` weight = 2.04).
2. **Targeted Crop Augmentation:** Apply localized affine scaling and directional rotation to D10 crops during dataset expansion.
