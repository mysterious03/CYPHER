"""
CYPhER Perception Adapters Registry Export
"""

from road_cv.adapters.base import BaseAdapter
from road_cv.adapters.vision_adapters import (
    DamageAdapter,
    VehicleAdapter,
    RoadSceneAdapter,
    WaterAdapter,
    DepthAdapter,
    SurfaceAdapter
)
from road_cv.adapters.specialized_adapters import (
    AccidentAdapter,
    WeatherAdapter,
    LaneAdapter,
    TrafficSignalAdapter,
    ObstacleAdapter,
    FrictionAdapter,
    RoughnessAdapter,
    PotholeGeometryAdapter
)

ADAPTER_MAP = {
    "damage": DamageAdapter,
    "vehicles": VehicleAdapter,
    "road_scene": RoadSceneAdapter,
    "water": WaterAdapter,
    "depth": DepthAdapter,
    "surface": SurfaceAdapter,
    "accident": AccidentAdapter,
    "weather": WeatherAdapter,
    "lanes": LaneAdapter,
    "traffic_signals": TrafficSignalAdapter,
    "obstacles": ObstacleAdapter,
    "friction": FrictionAdapter,
    "roughness": RoughnessAdapter,
    "pothole_geometry": PotholeGeometryAdapter
}
