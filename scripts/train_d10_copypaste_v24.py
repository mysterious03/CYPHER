"""
CYPhER V2.4: Road Damage Fine-Tuning with D10 Copy-Paste Augmentation
Fine-tunes the verified B0 baseline checkpoint on the balanced V2.4 training split.
Preserves all previous checkpoints untouched.
Output: runs/detect/road_damage_v24
"""

import os
import sys
import time
import torch
from pathlib import Path
from ultralytics import YOLO

def train_v24_damage():
    print("=" * 70)
    print("CYPhER V2.4: ROAD DAMAGE MODEL RETRAINING (D10 RESCUE & FINE-TUNING)")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is not available. Must run on RTX 4050 GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    print(f"Hardware Target: {gpu_name} ({vram_gb} GB VRAM)")

    data_yaml = Path("data/processed/rdd2022_india_v24/data_v24.yaml").resolve().as_posix()
    assert os.path.exists(data_yaml), f"Dataset configuration not found at {data_yaml}"

    # Initialize from verified B0 checkpoint
    base_checkpoint = Path("runs/detect/road_damage_v22/weights/best.pt").resolve().as_posix()
    assert os.path.exists(base_checkpoint), f"B0 checkpoint not found at {base_checkpoint}"

    print(f"\nInitializing fine-tuning from verified checkpoint: {base_checkpoint}")
    model = YOLO(base_checkpoint)

    start_time = time.time()

    print("\nTraining Configuration:")
    print("- Architecture: YOLO26n")
    print("- Base Weights: runs/detect/road_damage_v22/weights/best.pt")
    print("- Dataset: RDD2022 India + D10 Copy-Paste (5,618 train, 1,172 val)")
    print("- Epochs: 10")
    print("- Image Size: 640")
    print("- Batch Size: 16")
    print("- Optimizer: Auto (AdamW)")
    print("- Initial LR (lr0): 0.001")
    print("- Final LR factor (lrf): 0.01")
    print("- Device: 0 (RTX 4050)")
    print("- Workers: 2")
    print("- AMP: True")
    print("- Output Project: runs/detect")
    print("- Experiment Name: road_damage_v24")

    results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        amp=True,
        lr0=0.001,
        lrf=0.01,
        seed=42,
        cache=False,
        project="runs/detect",
        name="road_damage_v24",
        exist_ok=True,
        verbose=True
    )

    total_duration = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETED IN {total_duration / 60.0:.2f} MINUTES!")
    print("V2.4 Best Checkpoint: runs/detect/road_damage_v24/weights/best.pt")
    print("=" * 70)

if __name__ == "__main__":
    train_v24_damage()

