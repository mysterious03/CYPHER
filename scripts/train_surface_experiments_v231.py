"""
CYPhER V2.3.1 - Priority 1: Surface Classifier Controlled Retraining & Ablation Study
Ablates 6 training configurations to solve the severe class imbalance and Macro-F1 collapse:
1. exp1_ce_baseline: Standard unweighted Cross-Entropy
2. exp2_weighted_ce: Inverse-frequency class weighted Cross-Entropy
3. exp3_focal_loss: Class-balanced Focal Loss (gamma=2.0)
4. exp4_effective_number: Effective-number class weighting (Cui et al., CVPR 2019, beta=0.999)
5. exp5_class_aware_sampler: WeightedRandomSampler balancing batch class representations
6. exp6_combined: Class-aware sampling + Effective-number loss

Evaluates all heads on the full 1,172 held-out validation patches with strict Macro-F1 and confusion matrices.
Uses high-performance uint8 tensor caching for instant zero-overhead epoch iterations.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from road_cv.surface.friction_classifier import MultiTaskSurfaceClassifier, FRICTION_CLASSES, MATERIAL_CLASSES, UNEVENNESS_CLASSES

class FastCachedSurfaceDataset(Dataset):
    """
    High-performance in-memory / disk-cached dataset.
    Avoids repeatedly decompressing 1920x1080 JPEG images every epoch.
    """
    def __init__(self, samples: List[Dict], cache_file: str, augment: bool = False):
        self.samples = samples
        self.augment = augment
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"Loading pre-cached patches from {cache_path}...")
            cached = torch.load(cache_path, map_location="cpu")
            self.tensors = cached["tensors"]
            self.f_lbls = cached["f_lbls"]
            self.m_lbls = cached["m_lbls"]
            self.u_lbls = cached["u_lbls"]
            print(f"Loaded {len(self.tensors)} cached patches.")
        else:
            print(f"Pre-cropping {len(samples)} patches and saving to {cache_path}...")
            tensors = []
            f_lbls, m_lbls, u_lbls = [], [], []
            t0 = time.time()
            for idx, s in enumerate(samples):
                img = cv2.imread(s["image_path"])
                if img is None:
                    patch = np.full((224, 224, 3), 128, dtype=np.uint8)
                else:
                    if "crop_box" in s and s["crop_box"]:
                        x1, y1, x2, y2 = s["crop_box"]
                        h, w = img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            patch = img[y1:y2, x1:x2]
                        else:
                            patch = img
                    else:
                        patch = img
                    patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                tensors.append(torch.from_numpy(patch).permute(2, 0, 1))
                f_lbls.append(s["friction_label"])
                m_lbls.append(s["material_label"])
                u_lbls.append(s["unevenness_label"])

                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1}/{len(samples)} patches...")

            self.tensors = torch.stack(tensors) # uint8 [N, 3, 224, 224]
            self.f_lbls = torch.tensor(f_lbls, dtype=torch.long)
            self.m_lbls = torch.tensor(m_lbls, dtype=torch.long)
            self.u_lbls = torch.tensor(u_lbls, dtype=torch.long)

            torch.save({
                "tensors": self.tensors,
                "f_lbls": self.f_lbls,
                "m_lbls": self.m_lbls,
                "u_lbls": self.u_lbls
            }, cache_path)
            dt = time.time() - t0
            print(f"Cached {len(samples)} patches in {dt:.1f}s ({self.tensors.element_size() * self.tensors.nelement() / (1024**2):.1f} MB).")

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.tensors[idx].float() / 255.0
        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])
        x = (x - self.mean) / self.std
        return x, self.f_lbls[idx], self.m_lbls[idx], self.u_lbls[idx]

class FocalLoss(nn.Module):
    """Multi-class focal loss with class weight alpha and focusing parameter gamma."""
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def compute_effective_weights(class_counts: Dict[int, int], num_classes: int, beta: float = 0.999) -> torch.Tensor:
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        n = class_counts.get(c, 0)
        if n > 0:
            effective_num = 1.0 - np.power(beta, n)
            weights[c] = (1.0 - beta) / max(1e-8, effective_num)
        else:
            weights[c] = 0.0
    present_mask = weights > 0
    if np.sum(present_mask) > 0:
        weights[present_mask] = weights[present_mask] / np.mean(weights[present_mask])
    return torch.from_numpy(weights).float()

def compute_inverse_freq_weights(class_counts: Dict[int, int], num_classes: int) -> torch.Tensor:
    weights = np.zeros(num_classes, dtype=np.float32)
    total = sum(class_counts.values())
    for c in range(num_classes):
        n = class_counts.get(c, 0)
        if n > 0:
            weights[c] = total / (float(num_classes) * n)
        else:
            weights[c] = 0.0
    present_mask = weights > 0
    if np.sum(present_mask) > 0:
        weights[present_mask] = weights[present_mask] / np.mean(weights[present_mask])
    return torch.from_numpy(weights).float()

def evaluate_surface_model(model: nn.Module, val_loader: DataLoader, device: str) -> Dict:
    model.eval()
    all_f_pred, all_f_true = [], []
    all_m_pred, all_m_true = [], []
    all_u_pred, all_u_true = [], []

    with torch.no_grad():
        for imgs, f_lbl, m_lbl, u_lbl in val_loader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                out = model(imgs)

            all_f_pred.extend(out["friction_logits"].argmax(dim=-1).cpu().numpy().tolist())
            all_f_true.extend(f_lbl.numpy().tolist())
            all_m_pred.extend(out["material_logits"].argmax(dim=-1).cpu().numpy().tolist())
            all_m_true.extend(m_lbl.numpy().tolist())
            all_u_pred.extend(out["unevenness_logits"].argmax(dim=-1).cpu().numpy().tolist())
            all_u_true.extend(u_lbl.numpy().tolist())

    def get_head_metrics(y_true, y_pred, class_names):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_classes = len(class_names)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        acc = float(np.mean(y_true == y_pred) * 100)
        per_class = {}
        recalls = []
        f1s = []

        for c, name in enumerate(class_names):
            tp = int(cm[c, c])
            fp = int(np.sum(cm[:, c]) - tp)
            fn = int(np.sum(cm[c, :]) - tp)
            support = int(np.sum(cm[c, :]))

            prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            per_class[name] = {
                "precision": round(prec, 2),
                "recall": round(rec, 2),
                "f1": round(f1, 2),
                "support": support,
                "tp": tp, "fp": fp, "fn": fn
            }
            if support > 0:
                recalls.append(rec)
                f1s.append(f1)

        bal_acc = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        return {
            "accuracy": round(acc, 2),
            "balanced_accuracy": round(bal_acc, 2),
            "macro_f1": round(macro_f1, 2),
            "per_class": per_class,
            "confusion_matrix": cm.tolist()
        }

    return {
        "friction": get_head_metrics(all_f_true, all_f_pred, FRICTION_CLASSES),
        "material": get_head_metrics(all_m_true, all_m_pred, MATERIAL_CLASSES),
        "unevenness": get_head_metrics(all_u_true, all_u_pred, UNEVENNESS_CLASSES),
        "mean_accuracy": round(float(np.mean([
            np.mean(np.array(all_f_true) == np.array(all_f_pred)),
            np.mean(np.array(all_m_true) == np.array(all_m_pred)),
            np.mean(np.array(all_u_true) == np.array(all_u_pred))
        ])) * 100, 2)
    }

def run_experiment(
    exp_name: str,
    loss_type: str,
    use_sampler: bool,
    train_ds: Dataset,
    val_ds: Dataset,
    train_samples: List[Dict],
    out_dir: Path,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: str = "cuda"
) -> Dict:
    exp_dir = out_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n" + "=" * 65)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"Loss Type: {loss_type} | Class-Aware Sampler: {use_sampler} | Epochs: {epochs}")
    print("=" * 65)

    f_counts, u_counts = {}, {}
    for s in train_samples:
        f_counts[s["friction_label"]] = f_counts.get(s["friction_label"], 0) + 1
        u_counts[s["unevenness_label"]] = u_counts.get(s["unevenness_label"], 0) + 1

    if loss_type == "weighted_ce":
        w_f = compute_inverse_freq_weights(f_counts, len(FRICTION_CLASSES)).to(device)
        w_u = compute_inverse_freq_weights(u_counts, len(UNEVENNESS_CLASSES)).to(device)
        crit_f = nn.CrossEntropyLoss(weight=w_f)
        crit_m = nn.CrossEntropyLoss()
        crit_u = nn.CrossEntropyLoss(weight=w_u)
    elif loss_type == "focal_loss":
        w_f = compute_inverse_freq_weights(f_counts, len(FRICTION_CLASSES)).to(device)
        w_u = compute_inverse_freq_weights(u_counts, len(UNEVENNESS_CLASSES)).to(device)
        crit_f = FocalLoss(alpha=w_f, gamma=2.0)
        crit_m = nn.CrossEntropyLoss()
        crit_u = FocalLoss(alpha=w_u, gamma=2.0)
    elif loss_type == "effective_number":
        w_f = compute_effective_weights(f_counts, len(FRICTION_CLASSES), beta=0.999).to(device)
        w_u = compute_effective_weights(u_counts, len(UNEVENNESS_CLASSES), beta=0.999).to(device)
        crit_f = nn.CrossEntropyLoss(weight=w_f)
        crit_m = nn.CrossEntropyLoss()
        crit_u = nn.CrossEntropyLoss(weight=w_u)
    else: # baseline unweighted ce
        crit_f = nn.CrossEntropyLoss()
        crit_m = nn.CrossEntropyLoss()
        crit_u = nn.CrossEntropyLoss()

    sampler = None
    shuffle = True
    if use_sampler:
        sample_weights = []
        for s in train_samples:
            f_l = s["friction_label"]
            u_l = s["unevenness_label"]
            weight = (1.0 / max(1, f_counts[f_l])) + (1.0 / max(1, u_counts[u_l]))
            sample_weights.append(weight)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_samples), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    torch.manual_seed(42)
    model = MultiTaskSurfaceClassifier(backbone="mobilenet_v3_large", pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best_macro_f1 = -1.0
    best_metrics = None
    best_weights_path = exp_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t_start = time.time()
        running_loss = 0.0

        for imgs, f_lbl, m_lbl, u_lbl in train_loader:
            imgs = imgs.to(device)
            f_lbl, m_lbl, u_lbl = f_lbl.to(device), m_lbl.to(device), u_lbl.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                out = model(imgs)
                l_f = crit_f(out["friction_logits"], f_lbl)
                l_m = crit_m(out["material_logits"], m_lbl)
                l_u = crit_u(out["unevenness_logits"], u_lbl)
                total_loss = l_f + l_m + l_u

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item() * len(imgs)

        train_loss = running_loss / len(train_ds)
        val_eval = evaluate_surface_model(model, val_loader, device)
        comb_macro_f1 = (val_eval["friction"]["macro_f1"] + val_eval["unevenness"]["macro_f1"]) / 2.0

        if comb_macro_f1 > best_macro_f1:
            best_macro_f1 = comb_macro_f1
            best_metrics = val_eval
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "metrics": val_eval,
                "config": {"exp_name": exp_name, "loss_type": loss_type, "use_sampler": use_sampler}
            }, best_weights_path)

        duration = time.time() - t_start
        print(f"Epoch {epoch:02d}/{epochs:02d} [{duration:.1f}s] | Loss: {train_loss:.4f} | "
              f"Fric Macro-F1: {val_eval['friction']['macro_f1']:.1f}% (Wet R: {val_eval['friction']['per_class']['Wet']['recall']:.1f}%) | "
              f"Unev Macro-F1: {val_eval['unevenness']['macro_f1']:.1f}% (Unev R: {val_eval['unevenness']['per_class']['Slightly uneven']['recall']:.1f}%)",
              flush=True)

    with open(exp_dir / "evaluation_results.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    return best_metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Surface Ablations on: {device}")

    with open("data/processed/rscd_surface/train_patches.json", "r") as f:
        train_samples = json.load(f)
    with open("data/processed/rscd_surface/val_patches.json", "r") as f:
        val_samples = json.load(f)

    train_ds = FastCachedSurfaceDataset(train_samples, "data/processed/rscd_surface/cached_train.pt", augment=True)
    val_ds = FastCachedSurfaceDataset(val_samples, "data/processed/rscd_surface/cached_val.pt", augment=False)

    out_base = Path("runs/surface_classifier_v231")
    out_base.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("exp1_ce_baseline", "ce", False),
        ("exp2_weighted_ce", "weighted_ce", False),
        ("exp3_focal_loss", "focal_loss", False),
        ("exp4_effective_number", "effective_number", False),
        ("exp5_class_aware_sampler", "ce", True),
        ("exp6_combined_sampling_effective", "effective_number", True)
    ]

    all_exp_results = {}
    for name, loss_t, sampler_flag in experiments:
        res = run_experiment(
            exp_name=name,
            loss_type=loss_t,
            use_sampler=sampler_flag,
            train_ds=train_ds,
            val_ds=val_ds,
            train_samples=train_samples,
            out_dir=out_base,
            epochs=8,
            batch_size=32,
            lr=5e-4,
            device=device
        )
        all_exp_results[name] = res

    # Summary table
    print("\n" + "=" * 95)
    print("SURFACE RETRAINING ABLATION COMPARISON SUMMARY")
    print("=" * 95)
    print(f"{'Experiment':<34} | {'Mean Acc':<9} | {'Fric BalAcc':<11} | {'Fric MacF1':<11} | {'Wet Rec':<8} | {'Unev BalAcc':<11} | {'Unev MacF1':<11} | {'Unev Rec':<8}")
    print("-" * 95)

    summary_rows = []
    for name, res in all_exp_results.items():
        m_acc = res["mean_accuracy"]
        f_bal = res["friction"]["balanced_accuracy"]
        f_f1 = res["friction"]["macro_f1"]
        f_wet_r = res["friction"]["per_class"]["Wet"]["recall"]
        u_bal = res["unevenness"]["balanced_accuracy"]
        u_f1 = res["unevenness"]["macro_f1"]
        u_unev_r = res["unevenness"]["per_class"]["Slightly uneven"]["recall"]
        print(f"{name:<34} | {m_acc:<9.2f} | {f_bal:<11.2f} | {f_f1:<11.2f} | {f_wet_r:<8.2f} | {u_bal:<11.2f} | {u_f1:<11.2f} | {u_unev_r:<8.2f}")
        summary_rows.append({
            "experiment": name,
            "mean_accuracy": m_acc,
            "friction_balanced_accuracy": f_bal,
            "friction_macro_f1": f_f1,
            "wet_recall": f_wet_r,
            "unevenness_balanced_accuracy": u_bal,
            "unevenness_macro_f1": u_f1,
            "uneven_recall": u_unev_r
        })

    with open(out_base / "surface_ablation_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

if __name__ == "__main__":
    main()
