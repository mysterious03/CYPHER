"""
Water Segmentation Dataset Loader
Provides PyTorch Dataset and batch collators for R²S100K water segmentation.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
import torchvision.transforms as T

class WaterSegmentationDataset(Dataset):
    def __init__(self, samples: List[Dict], target_size: Tuple[int, int] = (512, 512)):
        self.samples = samples
        self.target_size = target_size
        self.img_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_p = sample["image_path"]
        mask_p = sample.get("mask_path", None)

        img = cv2.imread(img_p)
        if img is None:
            img = np.full((self.target_size[1], self.target_size[0], 3), 128, dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Generate or load mask
        if mask_p and os.path.exists(mask_p):
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        else:
            # Synthetic / heuristic mask generation based on dark specular asphalt & puddle thresholds
            mask = self._generate_heuristic_water_mask(img)

        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = self.img_norm(img_tensor)
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor

    def _generate_heuristic_water_mask(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Extracts plausible water bodies using specular intensity, low-saturation,
        and road region priors for initial baseline calibration.
        """
        h, w = img_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Water puddles typically occur in lower half (road)
        road_region = img_rgb[int(h * 0.5):, :]
        gray = cv2.cvtColor(road_region, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(road_region, cv2.COLOR_RGB2HSV)

        # Dark specular puddles: low value + high local contrast or sky reflection (high value, low sat)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        puddle_cond = (sat < 40) & ((val < 50) | (val > 210))
        sub_mask = np.zeros((h - int(h * 0.5), w), dtype=np.uint8)
        sub_mask[puddle_cond] = 2 # Water puddle

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, kernel)
        mask[int(h * 0.5):, :] = sub_mask

        return mask

