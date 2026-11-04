"""
CYPhER Comprehensive Test Suite (Phase 8)
Tests import, checkpoint loading, image inference, sensor safeguards,
calibration safeguards, CPU fallback, and multi-model pipeline execution.
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_cv.core.orchestrator import CYPhEROrchestrator, VALID_MODES
from road_cv.adapters.vision_adapters import (
    DamageAdapter, VehicleAdapter, RoadSceneAdapter,
    WaterAdapter, DepthAdapter, SurfaceAdapter
)
from road_cv.adapters.specialized_adapters import (
    FrictionAdapter, RoughnessAdapter, PotholeGeometryAdapter, WeatherAdapter
)
from road_cv.core.fusion import CYPhERHazardFusion

@pytest.fixture
def dummy_frame():
    """Generates synthetic 640x640 BGR road frame."""
    return np.full((640, 640, 3), 120, dtype=np.uint8)

def test_valid_modes_list():
    assert len(VALID_MODES) == 15
    assert "all" in VALID_MODES
    assert "damage" in VALID_MODES
    assert "friction" in VALID_MODES
    assert "roughness" in VALID_MODES

def test_damage_adapter(dummy_frame):
    adapter = DamageAdapter()
    res = adapter.predict(dummy_frame)
    assert res["status"] in ["success", "failed"]
    assert "predictions" in res
    assert isinstance(res["predictions"], list)

def test_surface_adapter(dummy_frame):
    adapter = SurfaceAdapter()
    res = adapter.predict(dummy_frame)
    assert res["status"] == "success"
    preds = res["predictions"]
    assert "visual_surface_condition" in preds
    assert "road_material" in preds
    assert "visual_unevenness" in preds

def test_weather_adapter(dummy_frame):
    adapter = WeatherAdapter()
    res = adapter.predict(dummy_frame)
    assert res["status"] == "success"
    assert "predicted_weather" in res["predictions"]
    assert "lighting_condition" in res["predictions"]

def test_friction_sensor_required_safeguard(dummy_frame):
    """Ensures uncalibrated RGB camera reports requires_sensor, not fake mu."""
    adapter = FrictionAdapter()
    res = adapter.predict(dummy_frame, sensors=None)
    assert res["status"] == "requires_sensor"
    assert "visual_slip_risk_estimate" in res["predictions"]
    assert "scientific_disclaimer" in res["predictions"]

def test_roughness_sensor_required_safeguard(dummy_frame):
    """Ensures visual roughness reports requires_sensor for true IRI."""
    adapter = RoughnessAdapter()
    res = adapter.predict(dummy_frame, sensors=None)
    assert res["status"] == "requires_sensor"
    assert "visual_roughness_class" in res["predictions"]
    assert res["predictions"]["iri_m_per_km"] is None

def test_pothole_geometry_calibration_safeguard(dummy_frame):
    """Ensures uncalibrated camera returns requires_calibration."""
    adapter = PotholeGeometryAdapter()
    res = adapter.predict(dummy_frame, damage=[{"box": [100, 200, 250, 350], "class_name": "D40_pothole"}], calibration=None)
    assert res["status"] == "requires_calibration"
    assert res["predictions"]["calibration_state"] == "UNCALIBRATED_RELATIVE"

def test_cpu_fallback(dummy_frame):
    """Tests execution on CPU device."""
    adapter = SurfaceAdapter()
    adapter.load(device="cpu")
    assert adapter.device == "cpu"
    res = adapter.predict(dummy_frame)
    assert res["status"] == "success"

def test_hazard_fusion():
    """Tests explainable rule-based fusion logic."""
    mock_caps = {
        "damage": {"predictions": [{"class_id": 3, "class_name": "D40_pothole", "box": [100, 100, 200, 200]}]},
        "water": {"predictions": {"standing_water_present": True, "waterlogged_potholes": [{"pothole_idx": 0}]}},
        "surface": {"predictions": {"visual_surface_condition": {"class_name": "Water-covered"}}},
        "weather": {"predictions": {"predicted_weather": "Rain_or_Wet_Reflective"}}
    }
    fusion = CYPhERHazardFusion.fuse(mock_caps)
    assert fusion["overall_road_safety_risk"] == "SEVERE_HAZARD"
    assert fusion["detected_hazards_count"] >= 1

def test_orchestrator_all_mode_on_frame(dummy_frame):
    """Tests full pipeline orchestrator execution on dummy image."""
    orchestrator = CYPhEROrchestrator()
    output = orchestrator.run(source=dummy_frame, mode="damage,surface,weather", save_visualizations=False)
    assert "capabilities" in output
    assert "fusion" in output
    assert "performance" in output
    assert output["executed_modes_count"] == 3
    assert output["capabilities"]["damage"]["status"] == "success"
