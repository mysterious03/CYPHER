"""
Waterlogging Segmentation Trainer
Trains LRASPP MobileNetV3-Large on road water masks using combined
CrossEntropy and Dice loss with mixed-precision support.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from road_cv.water.water_segmenter import RoadWaterSegmenter, WATER_CLASSES
from road_cv.water.water_dataset import WaterSegmentationDataset

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)

def train_water_model(
    train_samples: list,
    val_samples: list,
    output_dir: str = "runs/water_segmenter",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("TRAINING WATERLOGGING & FLOOD SEGMENTATION MODEL")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 65)

    train_ds = WaterSegmentationDataset(train_samples)
    val_ds = WaterSegmentationDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = RoadWaterSegmenter(num_classes=len(WATER_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    ce_criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss()

    best_iou = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(imgs)
                loss_ce = ce_criterion(logits, masks)
                loss_dice = dice_criterion(logits, masks)
                loss = loss_ce + loss_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(imgs)

        train_loss /= max(1, len(train_ds))

        # Evaluation
        model.eval()
        total_intersection = 0
        total_union = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    preds, _ = model.predict_mask(imgs)

                # Binary water detection evaluation (classes 1 and 2: standing water and puddle)
                pred_water = (preds == 1) | (preds == 2)
                gt_water = (masks == 1) | (masks == 2)

                inter = (pred_water & gt_water).sum().item()
                union = (pred_water | gt_water).sum().item()

                total_intersection += inter
                total_union += union

        mean_iou = round((total_intersection / max(1, total_union)) * 100, 2)
        elapsed = round(time.time() - t0, 2)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed}s] Loss: {train_loss:.4f} | Water IoU: {mean_iou}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "water_iou": mean_iou
        })

        if mean_iou >= best_iou:
            best_iou = mean_iou
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_iou": best_iou
            }, Path(output_dir) / "best_water_model.pt")

    with open(Path(output_dir) / "water_training_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_iou": best_iou, "history": history}, f, indent=2)

    print(f"\nWater model training finished! Best Water IoU: {best_iou}%")
    print(f"Saved: {Path(output_dir) / 'best_water_model.pt'}")
    return {"best_iou": best_iou, "history": history}

