"""
Dataset Quality, Leakage Audit, and Class-Imbalance Management Module
Detects corrupted annotations, identifies near-duplicate frames via perceptual hashing,
verifies train/val isolation against sequence leakage, and calculates class balancing weights.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import numpy as np
import cv2

class DatasetQualityManager:
    """
    Audits dataset integrity, detects corrupted or zero-area annotations,
    and identifies identical or near-duplicate frames.
    """

    @staticmethod
    def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
        """
        Computes difference hash (dHash) for fast perceptual duplicate detection.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        decimal_val = 0
        for bit in diff.flatten():
            decimal_val = (decimal_val << 1) | int(bit)
        return decimal_val

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count('1')

    def find_near_duplicates(
        self,
        image_paths: List[str],
        max_hamming_dist: int = 4
    ) -> List[Tuple[str, str, int]]:
        """
        Identifies pairs of near-duplicate images (e.g. video frames captured while stationary).
        """
        hashes = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is not None:
                h = self.compute_dhash(img)
                hashes.append((p, h))

        duplicates = []
        n = len(hashes)
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.hamming_distance(hashes[i][1], hashes[j][1])
                if dist <= max_hamming_dist:
                    duplicates.append((hashes[i][0], hashes[j][0], dist))

        return duplicates

    @staticmethod
    def audit_pascal_voc_xml(xml_path: str) -> Dict[str, Any]:
        """
        Validates VOC XML annotation: checks for malformed XML, missing tags,
        and invalid box coordinates (zero area, negative dims, coords outside image).
        """
        issues = []
        boxes = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size_node = root.find("size")
            if size_node is None:
                issues.append("missing_size_node")
                w, h = None, None
            else:
                w = int(size_node.find("width").text or 0)
                h = int(size_node.find("height").text or 0)

            for obj in root.findall("object"):
                name = obj.find("name").text
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    issues.append("missing_bndbox_node")
                    continue

                xmin = float(bndbox.find("xmin").text or 0)
                ymin = float(bndbox.find("ymin").text or 0)
                xmax = float(bndbox.find("xmax").text or 0)
                ymax = float(bndbox.find("ymax").text or 0)

                if xmax <= xmin or ymax <= ymin:
                    issues.append(f"invalid_degenerate_box_{name}")
                elif w and h and (xmax > w or ymax > h or xmin < 0 or ymin < 0):
                    issues.append(f"out_of_bounds_box_{name}")
                else:
                    boxes.append({"class_name": name, "box": [xmin, ymin, xmax, ymax]})

        except Exception as e:
            issues.append(f"parse_error_{str(e)}")

        return {
            "file": xml_path,
            "valid": len(issues) == 0,
            "issues": issues,
            "box_count": len(boxes),
            "boxes": boxes
        }

    @staticmethod
    def audit_train_val_leakage(
        train_image_names: List[str],
        val_image_names: List[str]
    ) -> Dict[str, Any]:
        """
        Checks for sequence/route leakage between train and validation splits
        by comparing sequence prefix IDs (e.g. India_00123... vs India_00124...).
        """
        # Exact duplicate filenames
        exact_overlap = set(train_image_names).intersection(set(val_image_names))

        # Check sequence IDs (extracting root prefix before last numeric token)
        def get_sequence_id(name: str) -> str:
            stem = Path(name).stem
            parts = stem.split('_')
            return '_'.join(parts[:-1]) if len(parts) > 1 else stem

        train_seqs = set(get_sequence_id(n) for n in train_image_names)
        val_seqs = set(get_sequence_id(n) for n in val_image_names)
        shared_seqs = train_seqs.intersection(val_seqs)

        return {
            "has_leakage": len(exact_overlap) > 0 or len(shared_seqs) > 0,
            "exact_duplicate_count": len(exact_overlap),
            "shared_sequence_count": len(shared_seqs),
            "shared_sequences": list(shared_seqs)[:10],
            "recommendation": "Partition train and validation splits strictly by sequence/route, not random shuffle."
        }


class ClassImbalanceManager:
    """
    Computes class statistics and balanced sampling weights for RDD categories:
    D00 (longitudinal crack), D10 (transverse crack), D20 (alligator crack), D40 (pothole).
    """

    RDD_CLASSES = ["D00", "D10", "D20", "D40"]

    @staticmethod
    def compute_class_weights(class_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Computes inverse frequency weights normalized such that mean weight is 1.0.
        """
        total = sum(class_counts.values())
        if total == 0:
            return {k: 1.0 for k in class_counts}

        weights = {}
        n_classes = len(class_counts)
        for cls, count in class_counts.items():
            if count > 0:
                weights[cls] = total / (n_classes * count)
            else:
                weights[cls] = 1.0

        # Normalize weights so average is 1.0
        avg_w = np.mean(list(weights.values()))
        return {cls: round(float(w / avg_w), 4) for cls, w in weights.items()}

