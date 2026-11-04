"""
Multi-Task Surface Classifier Trainer
Trains the shared-backbone MobileNetV3-Large on friction, material, and unevenness.
Optimized for NVIDIA RTX 4050 6GB VRAM with AMP and memory safeguards.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from road_cv.surface.friction_classifier import MultiTaskSurfaceClassifier, FRICTION_CLASSES, MATERIAL_CLASSES, UNEVENNESS_CLASSES
from road_cv.surface.surface_dataset import SurfacePatchDataset, RSCDDatasetBuilder

def train_surface_model(
    train_samples: list,
    val_samples: list,
    output_dir: str = "runs/surface_classifier",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("TRAINING MULTI-TASK ROAD SURFACE CLASSIFIER")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 65)

    train_ds = SurfacePatchDataset(train_samples)
    val_ds = SurfacePatchDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MultiTaskSurfaceClassifier(backbone="mobilenet_v3_large", pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    criterion_fric = nn.CrossEntropyLoss()
    criterion_mat = nn.CrossEntropyLoss()
    criterion_unev = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for imgs, f_lbl, m_lbl, u_lbl in train_loader:
            imgs = imgs.to(device)
            f_lbl, m_lbl, u_lbl = f_lbl.to(device), m_lbl.to(device), u_lbl.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(imgs)
                loss_f = criterion_fric(out["friction_logits"], f_lbl)
                loss_m = criterion_mat(out["material_logits"], m_lbl)
                loss_u = criterion_unev(out["unevenness_logits"], u_lbl)
                total_loss = loss_f + loss_m + loss_u

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item() * len(imgs)

        train_loss /= max(1, len(train_ds))

        # Validation
        model.eval()
        correct_f, correct_m, correct_u, total_val = 0, 0, 0, 0

        with torch.no_grad():
            for imgs, f_lbl, m_lbl, u_lbl in val_loader:
                imgs = imgs.to(device)
                f_lbl, m_lbl, u_lbl = f_lbl.to(device), m_lbl.to(device), u_lbl.to(device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = model(imgs)

                pred_f = out["friction_logits"].argmax(dim=-1)
                pred_m = out["material_logits"].argmax(dim=-1)
                pred_u = out["unevenness_logits"].argmax(dim=-1)

                correct_f += (pred_f == f_lbl).sum().item()
                correct_m += (pred_m == m_lbl).sum().item()
                correct_u += (pred_u == u_lbl).sum().item()
                total_val += len(imgs)

        acc_f = round((correct_f / max(1, total_val)) * 100, 2)
        acc_m = round((correct_m / max(1, total_val)) * 100, 2)
        acc_u = round((correct_u / max(1, total_val)) * 100, 2)
        mean_acc = round((acc_f + acc_m + acc_u) / 3, 2)
        elapsed = round(time.time() - t0, 2)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed}s] Loss: {train_loss:.4f} | Friction Acc: {acc_f}% | Material Acc: {acc_m}% | Unevenness Acc: {acc_u}% | Mean Acc: {mean_acc}%")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "friction_acc": acc_f,
            "material_acc": acc_m,
            "unevenness_acc": acc_u,
            "mean_acc": mean_acc
        }
        history.append(epoch_record)

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_path = Path(output_dir) / "best_surface_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": "mobilenet_v3_large",
                "epoch": epoch,
                "best_acc": best_acc
            }, best_path)

    # Save summary report
    with open(Path(output_dir) / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_acc": best_acc, "history": history}, f, indent=2)

    print(f"\nTraining completed! Best Mean Accuracy: {best_acc}%")
    print(f"Saved checkpoint: {Path(output_dir) / 'best_surface_model.pt'}")
    return {"best_acc": best_acc, "history": history}

