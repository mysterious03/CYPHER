"""
Targeted Transverse Crack (D10) Augmentation Module
Applies physically sound transformations specifically preserving transverse crack semantics:
- Horizontal reflection (horizontal crack remains transverse)
- Controlled angle perturbation (+/- 5 to 8 degrees, preventing rotation into longitudinal D00)
- Local contrast and brightness shifts
- Horizontal perspective scaling
Saves visual inspection previews to outputs/B1_previews/.
"""

import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class TargetedD10Augmentor:
    def __init__(
        self,
        base_train_img_dir: str = "data/processed/rdd2022_india/images/train",
        base_train_lbl_dir: str = "data/processed/rdd2022_india/labels/train",
        preview_dir: str = "outputs/B1_previews"
    ):
        self.img_dir = Path(base_train_img_dir)
        self.lbl_dir = Path(base_train_lbl_dir)
        self.preview_dir = Path(preview_dir)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def augment_transverse_sample(
        self,
        img: np.ndarray,
        boxes: List[Dict],
        mode: str = "hflip"
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Applies a semantic-preserving augmentation to an image and its YOLO normalized boxes.
        boxes: [{'cls': int, 'xc': float, 'yc': float, 'w': float, 'h': float}]
        """
        h, w = img.shape[:2]
        aug_img = img.copy()
        aug_boxes = []

        if mode == "hflip":
            aug_img = cv2.flip(img, 1)
            for b in boxes:
                # xc becomes 1.0 - xc
                aug_boxes.append({
                    "cls": b["cls"],
                    "xc": round(1.0 - b["xc"], 6),
                    "yc": b["yc"],
                    "w": b["w"],
                    "h": b["h"]
                })

        elif mode == "photometric":
            # Brightness & contrast jitter
            alpha = np.random.uniform(0.85, 1.15) # contrast
            beta = np.random.uniform(-15, 15)     # brightness
            aug_img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            aug_boxes = [b.copy() for b in boxes]

        elif mode == "slight_tilt":
            # Small rotation (+/- 4 degrees) strictly avoiding longitudinal confusion
            angle = np.random.choice([-4.0, 4.0])
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # For small angles, box center moves slightly; w and h stay approximately constant
            for b in boxes:
                cx, cy = b["xc"] * w, b["yc"] * h
                new_pt = M @ np.array([cx, cy, 1.0])
                aug_boxes.append({
                    "cls": b["cls"],
                    "xc": round(max(0.01, min(0.99, new_pt[0] / w)), 6),
                    "yc": round(max(0.01, min(0.99, new_pt[1] / h)), 6),
                    "w": b["w"],
                    "h": b["h"]
                })

        return aug_img, aug_boxes

    def build_augmented_dataset(
        self,
        output_data_dir: str = "data/processed/rdd2022_b1c_aug",
        num_variants_per_d10: int = 3
    ) -> Dict:
        out_p = Path(output_data_dir)
        train_img_out = out_p / "images" / "train"
        train_lbl_out = out_p / "labels" / "train"
        val_img_out = out_p / "images" / "val"
        val_lbl_out = out_p / "labels" / "val"

        train_img_out.mkdir(parents=True, exist_ok=True)
        train_lbl_out.mkdir(parents=True, exist_ok=True)
        val_img_out.mkdir(parents=True, exist_ok=True)
        val_lbl_out.mkdir(parents=True, exist_ok=True)

        # 1. Copy base training data
        copied_base = 0
        d10_files = []
        for lp in self.lbl_dir.glob("*.txt"):
            stem = lp.stem
            img_src = self.img_dir / f"{stem}.jpg"
            if not img_src.exists():
                continue

            shutil.copy2(lp, train_lbl_out / f"{stem}.txt")
            shutil.copy2(img_src, train_img_out / f"{stem}.jpg")
            copied_base += 1

            # Check if has D10
            with open(lp, "r", encoding="utf-8") as f:
                if any(line.strip().startswith("1 ") for line in f):
                    d10_files.append((stem, img_src, lp))

        print(f"Base train images copied: {copied_base}")
        print(f"D10 images to augment: {len(d10_files)}")

        # 2. Generate augmented variants for D10 images
        aug_count = 0
        preview_count = 0

        modes = ["hflip", "photometric", "slight_tilt"]

        for stem, img_p, lbl_p in d10_files:
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            # Parse original boxes
            boxes = []
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append({
                            "cls": int(parts[0]),
                            "xc": float(parts[1]),
                            "yc": float(parts[2]),
                            "w": float(parts[3]),
                            "h": float(parts[4])
                        })

            for i in range(num_variants_per_d10):
                mode = modes[i % len(modes)]
                aug_img, aug_boxes = self.augment_transverse_sample(img, boxes, mode=mode)

                aug_stem = f"{stem}_d10aug_{mode}"
                cv2.imwrite(str(train_img_out / f"{aug_stem}.jpg"), aug_img)

                with open(train_lbl_out / f"{aug_stem}.txt", "w", encoding="utf-8") as f:
                    for b in aug_boxes:
                        f.write(f"{b['cls']} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

                aug_count += 1

                # Save preview with box overlay for visual verification
                if preview_count < 8:
                    vis = aug_img.copy()
                    h, w = vis.shape[:2]
                    for b in aug_boxes:
                        x1 = int((b["xc"] - b["w"] / 2) * w)
                        y1 = int((b["yc"] - b["h"] / 2) * h)
                        x2 = int((b["xc"] + b["w"] / 2) * w)
                        y2 = int((b["yc"] + b["h"] / 2) * h)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(vis, f"D10 ({mode})", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imwrite(str(self.preview_dir / f"preview_{aug_stem}.jpg"), vis)
                    preview_count += 1

        # 3. Copy validation split untouched
        orig_val_img_dir = Path("data/processed/rdd2022_india/images/val")
        orig_val_lbl_dir = Path("data/processed/rdd2022_india/labels/val")
        for v_lp in orig_val_lbl_dir.glob("*.txt"):
            v_stem = v_lp.stem
            v_img = orig_val_img_dir / f"{v_stem}.jpg"
            shutil.copy2(v_lp, val_lbl_out / f"{v_stem}.txt")
            if v_img.exists():
                shutil.copy2(v_img, val_img_out / f"{v_stem}.jpg")

        # 4. Write data.yaml
        data_yaml_p = out_p / "data.yaml"
        yaml_content = f"""path: {out_p.resolve().as_posix()}
train: images/train
val: images/val
test: null

nc: 4
names:
  0: D00_longitudinal_crack
  1: D10_transverse_crack
  2: D20_alligator_crack
  3: D40_pothole
"""
        with open(data_yaml_p, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        return {
            "base_train_images": copied_base,
            "augmented_d10_images": aug_count,
            "total_new_train_images": copied_base + aug_count,
            "previews_saved": preview_count,
            "data_yaml": str(data_yaml_p.resolve())
        }

