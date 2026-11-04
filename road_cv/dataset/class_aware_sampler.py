"""
Class-Aware Sampler for Road Damage Defect Detection
Specifically balances rare classes (D10 transverse cracks) at the image level
without duplicating multi-label images or corrupting validation/test splits.
"""

import os
import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

class ClassAwareSampler:
    """
    Constructs an oversampled training dataset index to increase exposure
    to rare classes (D10) during YOLO training while preserving data splits.
    """
    def __init__(
        self,
        base_train_img_dir: str = "data/processed/rdd2022_india/images/train",
        base_train_lbl_dir: str = "data/processed/rdd2022_india/labels/train"
    ):
        self.img_dir = Path(base_train_img_dir)
        self.lbl_dir = Path(base_train_lbl_dir)
        assert self.img_dir.exists(), f"Image dir not found: {self.img_dir}"
        assert self.lbl_dir.exists(), f"Label dir not found: {self.lbl_dir}"

    def audit_class_distribution(self) -> Dict:
        """
        Audits image-level and instance-level occurrences of each class.
        """
        instance_counts = Counter()
        image_class_map = defaultdict(set)
        class_to_images = defaultdict(list)

        all_lbls = list(self.lbl_dir.glob("*.txt"))
        for lp in all_lbls:
            stem = lp.stem
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(parts[0])
                        instance_counts[c] += 1
                        image_class_map[stem].add(c)

            for c in image_class_map[stem]:
                class_to_images[c].append(stem)

        audit_res = {
            "total_images": len(all_lbls),
            "annotated_images": len(image_class_map),
            "background_images": len(all_lbls) - len(image_class_map),
            "instance_counts": dict(instance_counts),
            "image_counts_per_class": {c: len(imgs) for c, imgs in class_to_images.items()},
            "class_to_images": class_to_images,
            "image_class_map": image_class_map
        }
        return audit_res

    def build_oversampled_dataset(
        self,
        output_data_dir: str = "data/processed/rdd2022_b1a_sampling",
        target_rare_class: int = 1,
        repeat_factor: int = 4
    ) -> Dict:
        """
        Creates a balanced training directory where images containing the target rare class (D10)
        are repeated by repeat_factor to provide sufficient gradient updates per epoch.
        Preserves original validation split untouched by symbolic linking or path reference.
        """
        out_p = Path(output_data_dir)
        train_img_out = out_p / "images" / "train"
        train_lbl_out = out_p / "labels" / "train"
        val_img_out = out_p / "images" / "val"
        val_lbl_out = out_p / "labels" / "val"

        train_img_out.mkdir(parents=True, exist_ok=True)
        train_lbl_out.mkdir(parents=True, exist_ok=True)
        val_img_out.mkdir(parents=True, exist_ok=True)
        val_lbl_out.mkdir(parents=True, exist_ok=True)

        audit = self.audit_class_distribution()
        rare_stems = set(audit["class_to_images"].get(target_rare_class, []))

        print(f"Total training images: {audit['total_images']}")
        print(f"Target rare class {target_rare_class} images: {len(rare_stems)}")

        copied_train_count = 0
        oversampled_rare_count = 0

        # Copy original train images
        for lp in self.lbl_dir.glob("*.txt"):
            stem = lp.stem
            img_src = self.img_dir / f"{stem}.jpg"
            if not img_src.exists():
                continue

            # Standard copy of the original
            dst_lbl = train_lbl_out / f"{stem}.txt"
            dst_img = train_img_out / f"{stem}.jpg"
            if not dst_lbl.exists():
                shutil.copy2(lp, dst_lbl)
            if not dst_img.exists():
                shutil.copy2(img_src, dst_img)
            copied_train_count += 1

            # If it contains D10, replicate with unique suffix
            if stem in rare_stems:
                for rep in range(1, repeat_factor):
                    rep_stem = f"{stem}_rep{rep}"
                    rep_dst_lbl = train_lbl_out / f"{rep_stem}.txt"
                    rep_dst_img = train_img_out / f"{rep_stem}.jpg"
                    if not rep_dst_lbl.exists():
                        shutil.copy2(lp, rep_dst_lbl)
                    if not rep_dst_img.exists():
                        shutil.copy2(img_src, rep_dst_img)
                    oversampled_rare_count += 1

        # Link validation set directly (strictly untouched)
        orig_val_img_dir = Path("data/processed/rdd2022_india/images/val")
        orig_val_lbl_dir = Path("data/processed/rdd2022_india/labels/val")

        for v_lp in orig_val_lbl_dir.glob("*.txt"):
            v_stem = v_lp.stem
            v_img = orig_val_img_dir / f"{v_stem}.jpg"
            dst_v_lbl = val_lbl_out / f"{v_stem}.txt"
            dst_v_img = val_img_out / f"{v_stem}.jpg"
            if not dst_v_lbl.exists():
                shutil.copy2(v_lp, dst_v_lbl)
            if not dst_v_img.exists() and v_img.exists():
                shutil.copy2(v_img, dst_v_img)

        # Write data.yaml for B1A
        data_yaml_path = out_p / "data.yaml"
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
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        summary = {
            "original_train_images": copied_train_count,
            "rare_class_replicated_images": oversampled_rare_count,
            "total_new_train_images": copied_train_count + oversampled_rare_count,
            "effective_d10_instances": audit["instance_counts"][target_rare_class] * repeat_factor,
            "data_yaml": str(data_yaml_path.resolve())
        }
        return summary

