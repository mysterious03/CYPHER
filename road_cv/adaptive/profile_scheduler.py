"""
CYPhER Adaptive Perception: Compute-Aware Profile Scheduler
Defines operational execution profiles balancing detection recall against latency and GPU VRAM budget.
"""

from typing import Dict, Any

class AdaptiveProfileScheduler:
    """
    Configures perception pipelines based on hardware latency constraints
    and precision requirements.
    """

    PROFILES = {
        "fast": {
            "name": "Fast Real-Time Profile",
            "imgsz": 640,
            "enable_road_prior": True,
            "enable_candidate_zoom": True,
            "max_candidate_crops": 1,
            "enable_independent_tiling": False,
            "max_independent_tiles": 0,
            "tile_overlap_ratio": 0.15,
            "confidence_threshold": 0.30,
            "uncertainty_trigger_threshold": 0.55,
            "target_latency_ms": 100.0,
            "target_fps": 10.0,
            "max_vram_mb": 250.0
        },
        "balanced": {
            "name": "Balanced Road Intelligence Profile",
            "imgsz": 640,
            "enable_road_prior": True,
            "enable_candidate_zoom": True,
            "max_candidate_crops": 3,
            "enable_independent_tiling": False,
            "max_independent_tiles": 0,
            "tile_overlap_ratio": 0.20,
            "confidence_threshold": 0.25,
            "uncertainty_trigger_threshold": 0.45,
            "target_latency_ms": 200.0,
            "target_fps": 5.0,
            "max_vram_mb": 350.0
        },
        "high_accuracy": {
            "name": "Research High-Accuracy & Rare-Class Recovery Profile",
            "imgsz": 640,
            "enable_road_prior": True,
            "enable_candidate_zoom": True,
            "max_candidate_crops": 4,
            "enable_independent_tiling": True,
            "max_independent_tiles": 4,
            "tile_overlap_ratio": 0.25,
            "confidence_threshold": 0.20,
            "uncertainty_trigger_threshold": 0.35,
            "target_latency_ms": 400.0,
            "target_fps": 2.5,
            "max_vram_mb": 500.0
        }
    }

    def __init__(self, default_profile: str = "balanced"):
        if default_profile not in self.PROFILES:
            raise ValueError(f"Unknown profile: {default_profile}. Choose from: {list(self.PROFILES.keys())}")
        self.current_profile = default_profile

    def get_config(self, profile: str = None) -> Dict[str, Any]:
        p = profile or self.current_profile
        return dict(self.PROFILES.get(p, self.PROFILES["balanced"]))

    def select_profile_for_speed(self, vehicle_speed_kmh: float) -> str:
        """
        Dynamically adapts profile based on vehicle velocity:
        - Highway speeds (>75 km/h): fast real-time profile for reaction time.
        - Urban speeds (35-75 km/h): balanced profile.
        - Low speed / crawl (<35 km/h) or inspection: high_accuracy research profile.
        """
        if vehicle_speed_kmh > 75.0:
            return "fast"
        elif vehicle_speed_kmh > 35.0:
            return "balanced"
        else:
            return "high_accuracy"
