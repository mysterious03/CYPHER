"""
CYPhER V2.5 Visual Evidence Generator
Plots:
1. Performance comparison chart (B0 Baseline vs V2.5 Retrained vs Adaptive Profiles)
2. Confusion matrix for Surface Classifier (Friction & Unevenness)
3. Quantitative metrics scorecard visualization
Saves to reports/v25_evidence/
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path("reports/v25_evidence")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_damage_benchmark():
    json_path = Path("reports/v25_results/adaptive_benchmark_summary.json")
    if not json_path.exists():
        print("JSON not found!")
        return

    with open(json_path) as f:
        data = json.load(f)

    labels = ["B0 Baseline", "V2.5 Standalone", "Adaptive FAST", "Adaptive BALANCED", "Adaptive HIGH_ACC"]
    precisions = [d["overall_precision"] for d in data]
    recalls = [d["overall_recall"] for d in data]
    f1s = [d["overall_f1"] for d in data]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precisions, width, label="Precision (%)", color="#1f77b4")
    ax.bar(x, recalls, width, label="Recall (%)", color="#2ca02c")
    ax.bar(x + width, f1s, width, label="F1-Score (%)", color="#ff7f0e")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("CYPhER V2.5 Road Damage Performance Across Configurations (1,172 Holdout Images)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    for i in range(len(labels)):
        ax.text(x[i] - width, precisions[i] + 1.5, f"{precisions[i]:.1f}", ha="center", fontsize=8)
        ax.text(x[i], recalls[i] + 1.5, f"{recalls[i]:.1f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(x[i] + width, f1s[i] + 1.5, f"{f1s[i]:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    chart_p = OUT_DIR / "v25_damage_benchmark_comparison.png"
    plt.savefig(chart_p, dpi=200)
    plt.close()
    print(f"Saved damage comparison chart to: {chart_p}")

def plot_surface_confusion_matrix():
    # Confusion matrix from exp4_effective_number
    # Friction: Dry, Wet
    cm_fric = np.array([
        [1131, 33],
        [0, 8]
    ])
    labels_fric = ["Dry", "Wet"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    im1 = ax1.imshow(cm_fric, cmap="Blues")
    ax1.set_title("Friction Head (100% Wet Recall)", fontsize=11, fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labels_fric)
    ax1.set_yticklabels(labels_fric)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Ground Truth")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm_fric[i, j]), ha="center", va="center", color="white" if cm_fric[i, j] > 500 else "black", fontsize=12, fontweight="bold")

    # Unevenness: Smooth, Uneven
    cm_unev = np.array([
        [1072, 34],
        [2, 64]
    ])
    labels_unev = ["Smooth", "Uneven"]

    im2 = ax2.imshow(cm_unev, cmap="Greens")
    ax2.set_title("Unevenness Head (96.97% Uneven Recall)", fontsize=11, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(labels_unev)
    ax2.set_yticklabels(labels_unev)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Ground Truth")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm_unev[i, j]), ha="center", va="center", color="white" if cm_unev[i, j] > 500 else "black", fontsize=12, fontweight="bold")

    plt.suptitle("CYPhER V2.5 Surface Classifier Confusion Matrices (exp4_effective_number)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_p = OUT_DIR / "v25_surface_confusion_matrices.png"
    plt.savefig(cm_p, dpi=200)
    plt.close()
    print(f"Saved surface confusion matrix to: {cm_p}")

if __name__ == "__main__":
    plot_damage_benchmark()
    plot_surface_confusion_matrix()
