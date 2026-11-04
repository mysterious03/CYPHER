"""
CYPhER Adaptive Perception: Road-Aware Spatial Selector
Leverages semantic road segmentation masks to produce soft spatial attention priors
and region-of-interest proposals for downstream defect detectors.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

class RoadAwareSpatialSelector:
    """
    Computes drivable corridor bounding hulls and soft spatial priors
    to guide high-resolution tiling and hazard filtering.
    """

    def __init__(self, boundary_dilation_px: int = 40):
        self.boundary_dilation_px = boundary_dilation_px

    def extract_road_roi(self, road_mask: np.ndarray) -> Dict[str, Any]:
        """
        Extracts dominant road contour, bounding box, and surface area ratio.
        """
        if road_mask is None or road_mask.max() == 0:
            return {
                "has_road": False,
                "bbox": None,
                "area_ratio": 0.0,
                "contour": None
            }

        h, w = road_mask.shape[:2]
        total_pixels = float(h * w)

        # Find largest connected component (drivable road)
        contours, _ = cv2.findContours(road_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"has_road": False, "bbox": None, "area_ratio": 0.0, "contour": None}

        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        x, y, bw, bh = cv2.boundingRect(largest_cnt)

        return {
            "has_road": True,
            "bbox": (x, y, x + bw, y + bh), # (x1, y1, x2, y2)
            "area_ratio": round(area / total_pixels, 4),
            "contour": largest_cnt
        }

    def compute_soft_spatial_prior(
        self,
        box: List[float],
        road_mask: Optional[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> float:
        """
        Calculates a soft spatial weighting factor in [0.2, 1.0].
        - Completely within road mask: 1.0
        - Within boundary dilation margin: 0.80
        - Completely off-road: 0.30 (preserves road-edge/curb defects without strict rejection)
        """
        if road_mask is None:
            return 1.0

        h, w = image_shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = np.clip((x1 + x2) // 2, 0, w - 1)
        cy = np.clip((y1 + y2) // 2, 0, h - 1)

        # Direct center hit on road mask
        if road_mask[cy, cx] > 0:
            return 1.0

        # Check dilated boundary margin
        dilated = cv2.dilate(
            road_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (self.boundary_dilation_px, self.boundary_dilation_px))
        )
        if dilated[cy, cx] > 0:
            return 0.80

        # Off-road soft prior (never 0.0 to prevent hard suppression of off-edge hazards)
        return 0.30

    def generate_road_tiling_grid(
        self,
        road_mask: Optional[np.ndarray],
        image_shape: Tuple[int, int],
        tile_size: int = 640,
        overlap_ratio: float = 0.25
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generates overlapping high-resolution tiles focused only on the drivable corridor.
        Returns list of (x1, y1, x2, y2) tile coordinates.
        """
        h, w = image_shape[:2]
        tiles = []

        if road_mask is None:
            # Fallback uniform horizontal slice on lower half of frame
            y_start = max(0, h // 3)
            step_x = int(tile_size * (1.0 - overlap_ratio))
            for x in range(0, max(1, w - tile_size + 1), step_x):
                tiles.append((x, y_start, min(w, x + tile_size), min(h, y_start + tile_size)))
            return tiles

        roi = self.extract_road_roi(road_mask)
        if not roi["has_road"]:
            return tiles

        rx1, ry1, rx2, ry2 = roi["bbox"]
        step_x = max(64, int(tile_size * (1.0 - overlap_ratio)))
        step_y = max(64, int(tile_size * (1.0 - overlap_ratio)))

        for y in range(ry1, max(ry1 + 1, ry2 - tile_size // 2), step_y):
            for x in range(rx1, max(rx1 + 1, rx2 - tile_size // 2), step_x):
                x1 = max(0, min(x, w - tile_size))
                y1 = max(0, min(y, h - tile_size))
                x2 = min(w, x1 + tile_size)
                y2 = min(h, y1 + tile_size)

                # Ensure tile intersects meaningfully with the road mask (>10% road)
                sub_mask = road_mask[y1:y2, x1:x2]
                if (np.count_nonzero(sub_mask) / float(tile_size * tile_size)) >= 0.10:
                    tile_coords = (x1, y1, x2, y2)
                    if tile_coords not in tiles:
                        tiles.append(tile_coords)

        return tiles
