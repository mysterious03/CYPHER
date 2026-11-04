"""
CYPhER V2.4: D10 Transverse Crack Copy-Paste Dataset Preparation
Extracts 50 authentic D10 cracks from RDD2022 India training split,
applies photometric matching, edge feathering, and realistic perspective pasting
onto non-cracked road training images.
CRITICAL CONSTRAINT: Operates STRICTLY on training split. Validation split is 100% UNTOUCHED.
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

RDD_DIR = Path("data/processed/rdd2022_india")
OUT_DIR = Path("data/processed/rdd2022_india_v24")

TRAIN_IMG_DIR = RDD_DIR / "images" / "train"
TRAIN_LBL_DIR = RDD_DIR / "labels" / "train"

VAL_IMG_DIR = RDD_DIR / "images" / "val"
VAL_LBL_DIR = RDD_DIR / "labels" / "val"

def extract_authentic_d10_patches():
    print("Extracting authentic D10 patches from training split...")
    d10_patches = []
    train_lbls = sorted(list(TRAIN_LBL_DIR.glob("*.txt")))

    for lf in train_lbls:
        img_p = TRAIN_IMG_DIR / f"{lf.stem}.jpg"
        if not img_p.exists():
            continue

        with open(lf, "r") as f:
            lines = f.readlines()

        img = None
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id == 1: # D10 Transverse Crack
                if img is None:
                    img = cv2.imread(str(img_p))
                if img is None:
                    continue

                h, w = img.shape[:2]
                xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                x1 = max(0, int((xc - bw / 2.0) * w))
                y1 = max(0, int((yc - bh / 2.0) * h))
                x2 = min(w, int((xc + bw / 2.0) * w))
                y2 = min(h, int((yc + bh / 2.0) * h))

                pw = x2 - x1
                ph = y2 - y1
                if pw >= 20 and ph >= 8:
                    patch = img[y1:y2, x1:x2].copy()
                    d10_patches.append({
                        "patch": patch,
                        "w": pw,
                        "h": ph,
                        "aspect_ratio": pw / max(1, ph),
                        "source_img": img_p.name
                    })

    print(f"Extracted {len(d10_patches)} authentic D10 crack patches.")
    return d10_patches

def create_feathered_mask(h, w, feather_px=4):
    mask = np.ones((h, w), dtype=np.float32)
    # Feather top and bottom
    for i in range(min(feather_px, h // 2)):
        alpha = (i + 1) / float(feather_px + 1)
        mask[i, :] *= alpha
        mask[h - 1 - i, :] *= alpha
    # Feather left and right
    for j in range(min(feather_px, w // 2)):
        alpha = (j + 1) / float(feather_px + 1)
        mask[:, j] *= alpha
        mask[:, w - 1 - j] *= alpha
    return mask

def match_patch_photometrics(src_patch, dst_roi):
    src_h, src_w = src_patch.shape[:2]
    dst_h, dst_w = dst_roi.shape[:2]

    if (src_h, src_w) != (dst_h, dst_w):
        src_patch = cv2.resize(src_patch, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    # Convert to LAB for luminance matching
    src_lab = cv2.cvtColor(src_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Shift mean L channel to match background road
    l_diff = np.mean(dst_lab[:, :, 0]) - np.mean(src_lab[:, :, 0])
    src_lab[:, :, 0] = np.clip(src_lab[:, :, 0] + l_diff * 0.7, 0, 255)

    matched = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return matched

def blend_patch(dst_img, patch, target_x1, target_y1):
    th, tw = patch.shape[:2]
    img_h, img_w = dst_img.shape[:2]

    # Bounds check
    target_x2 = min(img_w, target_x1 + tw)
    target_y2 = min(img_h, target_y1 + th)
    tw_adj = target_x2 - target_x1
    th_adj = target_y2 - target_y1

    if tw_adj < 10 or th_adj < 5:
        return False, None

    patch = patch[:th_adj, :tw_adj]
    dst_roi = dst_img[target_y1:target_y2, target_x1:target_x2]

    matched_patch = match_patch_photometrics(patch, dst_roi)
    mask = create_feathered_mask(th_adj, tw_adj, feather_px=min(6, th_adj // 3))
    mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    blended = (matched_patch * mask_3c + dst_roi * (1.0 - mask_3c)).astype(np.uint8)
    dst_img[target_y1:target_y2, target_x1:target_x2] = blended

    # Calculate YOLO normalized bbox
    xc = (target_x1 + target_x2) / (2.0 * img_w)
    yc = (target_y1 + target_y2) / (2.0 * img_h)
    bw = tw_adj / float(img_w)
    bh = th_adj / float(img_h)

    return True, (1, xc, yc, bw, bh)

def generate_v24_copypaste_dataset(num_augmented_samples=250):
    d10_patches = extract_authentic_d10_patches()
    assert len(d10_patches) > 0, "No D10 patches found!"

    out_train_img = OUT_DIR / "images" / "train"
    out_train_lbl = OUT_DIR / "labels" / "train"
    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_lbl.mkdir(parents=True, exist_ok=True)

    # Collect candidate training background images (clean asphalt preferred)
    train_lbl_files = sorted(list(TRAIN_LBL_DIR.glob("*.txt")))
    background_candidates = []
    for lf in train_lbl_files:
        if lf.stat().st_size == 0:
            img_p = TRAIN_IMG_DIR / f"{lf.stem}.jpg"
            if img_p.exists():
                background_candidates.append(img_p)

    print(f"Found {len(background_candidates)} candidate clean training road images.")
    random.shuffle(background_candidates)

    augmented_img_paths = []
    pasted_count = 0

    for i in range(min(num_augmented_samples, len(background_candidates))):
        bg_path = background_candidates[i]
        img = cv2.imread(str(bg_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Select 1 or 2 D10 patches to paste
        n_paste = random.choice([1, 1, 2])
        new_labels = []

        for _ in range(n_paste):
            p_info = random.choice(d10_patches)
            patch = p_info["patch"].copy()

            # Geometric augmentations
            # Horizontal flip
            if random.random() > 0.5:
                patch = cv2.flip(patch, 1)

            # Slight scale jitter
            scale = random.uniform(0.85, 1.25)
            new_pw = max(30, int(patch.shape[1] * scale))
            new_ph = max(10, int(patch.shape[0] * scale))
            patch = cv2.resize(patch, (new_pw, new_ph), interpolation=cv2.INTER_LINEAR)

            # Place in lower road perspective area: y in [0.65h, 0.90h], x centered
            max_x = max(1, w - new_pw - 20)
            x_min = int(w * 0.15)
            x_max = max(x_min + 1, min(max_x, int(w * 0.65)))
            target_x1 = random.randint(x_min, x_max)

            y_min = int(h * 0.65)
            y_max = max(y_min + 1, int(h * 0.90 - new_ph))
            target_y1 = random.randint(y_min, y_max)

            success, bbox = blend_patch(img, patch, target_x1, target_y1)
            if success:
                new_labels.append(bbox)
                pasted_count += 1

        if new_labels:
            aug_name = f"aug_d10_{i:04d}"
            aug_img_path = out_train_img / f"{aug_name}.jpg"
            aug_lbl_path = out_train_lbl / f"{aug_name}.txt"

            cv2.imwrite(str(aug_img_path), img)
            with open(aug_lbl_path, "w") as f:
                for lbl in new_labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

            augmented_img_paths.append(str(aug_img_path.resolve()))

    print(f"Generated {len(augmented_img_paths)} augmented images with {pasted_count} D10 instances.")

    # Create train.txt and val.txt manifest files
    print("\nCreating manifests for YOLO training...")
    train_manifest = OUT_DIR / "train.txt"
    val_manifest = OUT_DIR / "val.txt"

    # Original train images
    orig_train_imgs = sorted([str(p.resolve()) for p in TRAIN_IMG_DIR.glob("*.jpg")])
    all_train_imgs = orig_train_imgs + augmented_img_paths

    with open(train_manifest, "w") as f:
        for p in all_train_imgs:
            f.write(f"{os.path.normpath(p)}\n")

    # Original val images (STRICTLY UNMODIFIED)
    orig_val_imgs = sorted([str(p.resolve()) for p in VAL_IMG_DIR.glob("*.jpg")])
    with open(val_manifest, "w") as f:
        for p in orig_val_imgs:
            f.write(f"{os.path.normpath(p)}\n")

    print(f"train.txt: {len(all_train_imgs)} images (5,368 original + {len(augmented_img_paths)} copy-paste)")
    print(f"val.txt: {len(orig_val_imgs)} images (100% held-out authentic ground truth)")

    # Create data_v24.yaml
    data_yaml_path = OUT_DIR / "data_v24.yaml"
    data_yaml_content = f"""# CYPhER V2.4 Road Damage Detection (RDD2022 India + D10 Copy-Paste Augmentation)
path: {OUT_DIR.resolve().as_posix()}
train: train.txt
val: val.txt

names:
  0: D00_longitudinal_crack
  1: D10_transverse_crack
  2: D20_alligator_crack
  3: D40_pothole
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)

    print(f"Dataset configuration written to: {data_yaml_path}")
    return data_yaml_path

if __name__ == "__main__":
    generate_v24_copypaste_dataset(num_augmented_samples=250)

