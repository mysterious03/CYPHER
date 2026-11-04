"""
Edge-Aware Feature Preservation and Gradient Enhancement for Thin Cracks
Provides controlled, edge-preserving enhancements specifically for thin hairline cracks.
"""

from typing import Tuple, Optional
import numpy as np
import cv2

class EdgePreservingEnhancer:
    """
    Applies gradient and edge-guided enhancement to local crops containing potential thin cracks.
    Preserves subtle line features without introducing artificial artifacts.
    """

    @staticmethod
    def compute_local_edge_density(image_crop: np.ndarray) -> float:
        """
        Computes normalized edge density using a Canny detector tuned for pavement texture.
        """
        if image_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop
        edges = cv2.Canny(gray, threshold1=40, threshold2=120)
        return float(np.count_nonzero(edges) / edges.size)

    @staticmethod
    def enhance_thin_cracks(image_crop: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """
        Enhances directional linear contrast using a guided filter / unsharp masking
        with an edge mask, accentuating dark crack tracks against road surface grain.
        
        Args:
            image_crop: BGR input crop (H x W x 3)
            alpha: blending weight of high-frequency edge component (0.0 to 1.0)
        """
        if image_crop.size == 0:
            return image_crop

        # Convert to LAB for luminance-only processing to avoid color shifting
        lab = cv2.cvtColor(image_crop, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Bilateral filter preserves sharp boundaries while smoothing asphalt grain
        smoothed = cv2.bilateralFilter(l_channel, d=5, sigmaColor=35, sigmaSpace=35)
        
        # High-frequency difference (crack details)
        high_freq = cv2.subtract(l_channel, smoothed)
        
        # Dark crack enhancement: crack valleys are typically darker than local background
        enhanced_l = cv2.subtract(l_channel, (high_freq * alpha).astype(np.uint8))
        
        merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    @staticmethod
    def extract_oriented_crack_mask(image_crop: np.ndarray) -> np.ndarray:
        """
        Extracts multi-directional line responses using morphological black-hat transforms,
        which specifically isolate thin structures darker than their surroundings.
        """
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop
        
        # Linear kernels at 0, 45, 90, 135 degrees
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),   # horizontal (transverse)
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)),   # vertical (longitudinal)
            np.eye(7, dtype=np.uint8),                           # diagonal
            np.fliplr(np.eye(7, dtype=np.uint8))                 # anti-diagonal
        ]
        
        accumulated_response = np.zeros_like(gray)
        for k in kernels:
            # Black-hat isolates dark lines
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
            accumulated_response = np.maximum(accumulated_response, blackhat)
            
        return accumulated_response

