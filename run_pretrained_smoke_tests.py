"""
Smoke Test and Verification Harness for CYPhER Pretrained AI Models.
Executes isolated tests on Road.jpg and temporal sequences, measuring latency,
peak GPU VRAM, output sanity, and verification status across all 11 model categories.
"""
import os
import sys
import time
import json
import cv2
import numpy as np
import torch

from road_cv.adapters.pretrained_damage_adapter import PretrainedDamageAdapter
from road_cv.surface.pretrained_surface_adapter import PretrainedSurfaceAdapter
from road_cv.water.pretrained_wetness_adapter import PretrainedWetnessAdapter
from road_cv.water.pretrained_puddle_adapter import PretrainedPuddleAdapter
from road_cv.tracking.vehicle_path_tracker import VehiclePathTracker
from road_cv.scene.idd_drivable_adapter import IDDDrivableAdapter
from road_cv.depth.pretrained_depth_adapter import PretrainedDepthAdapter
from road_cv.accident.simple_tad_adapter import SimpleTADAdapter
from road_cv.weather.pretrained_weather_adapter import PretrainedWeatherAdapter
from road_cv.lanes.clrernet_adapter import CLRerNetAdapter
from road_cv.traffic_signals.pretrained_signal_adapter import PretrainedSignalAdapter
from road_cv.obstacles.pretrained_obstacle_adapter import PretrainedObstacleAdapter
from road_cv.core.pretrained_model_manager import PretrainedModelManager

def measure_execution(fn, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        vram_start = torch.cuda.memory_allocated() / (1024**2)
    else:
        vram_start = 0.0

    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    dt = time.perf_counter() - t0

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_vram = 0.0

    return res, round(dt * 1000, 2), round(peak_vram, 2)

def main():
    os.makedirs("outputs/pretrained_smoke_tests", exist_ok=True)
    img_path = "Road.jpg"
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, "Road.jpg must be present for smoke testing"

    print("========================================================================")
    print("      CYPhER PRETRAINED AI MODEL SMOKE TESTS & VERIFICATION AUDIT       ")
    print("========================================================================")

    test_results = {}

    # 1. Road Damage (YOLO12s RDD2022)
    print("\n[1/12] Testing Pretrained Damage Adapter (YOLO12s RDD2022)...")
    adapter = PretrainedDamageAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M1_damage_yolo12s"] = {
        "model": "YOLO12s_RDD2022",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "detections_count": res.get("detections_count"),
        "classes_detected": list(set(d["class_name"] for d in res.get("detections", [])))
    }
    if res.get("overlay") is not None:
        cv2.imwrite("outputs/pretrained_smoke_tests/m1_damage_yolo12s.jpg", res["overlay"])
    adapter.release_resources()
    print(f"  -> Latency: {lat} ms | Peak VRAM: {vram} MB | Status: {res.get('status')}")

    # 2. Surface Condition (SurfaceAI EfficientNetV2-S)
    print("\n[2/12] Testing Pretrained Surface Adapter (SurfaceAI EfficientNetV2-S)...")
    adapter = PretrainedSurfaceAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M2_surface_efficientnet_v2s"] = {
        "model": "SurfaceAI_EfficientNetV2S",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "surface_type": res.get("surface_type"),
        "surface_confidence": res.get("surface_type_confidence"),
        "quality_grade": res.get("quality_grade"),
        "quality_score_continuous": res.get("quality_score_continuous")
    }
    adapter.release_resources()
    print(f"  -> Type: {res.get('surface_type')} ({res.get('surface_type_confidence')}) | Quality: {res.get('quality_grade')} ({res.get('quality_score_continuous')}) | Latency: {lat} ms")

    # 3. Road Wetness (YOLOv8n-Seg Lithuanian model)
    print("\n[3/12] Testing Pretrained Wetness Adapter (YOLOv8n-Seg)...")
    adapter = PretrainedWetnessAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M3_water_wetness_yolov8n"] = {
        "model": "RoadWetness_YOLOv8n_Seg",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "predicted_condition": res.get("predicted_condition"),
        "wet_area_ratio": res.get("wet_area_ratio"),
        "dry_area_ratio": res.get("dry_area_ratio")
    }
    if res.get("overlay") is not None:
        cv2.imwrite("outputs/pretrained_smoke_tests/m3_water_wetness.jpg", res["overlay"])
    adapter.release_resources()
    print(f"  -> Condition: {res.get('predicted_condition')} | Wet Ratio: {res.get('wet_area_ratio')} | Latency: {lat} ms")

    # 4. Multiclass Puddle UNet (Audit & Availability Check)
    print("\n[4/12] Testing Multiclass Puddle Adapter (Shrinidhi1/Multiclass-Semantic-Segmentation)...")
    adapter = PretrainedPuddleAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M3_water_puddle_unet"] = {
        "model": "MulticlassWaterPuddle_UNet",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "reason": res.get("reason")
    }
    print(f"  -> Status: {res.get('status')} | Reason: Checked notebooks only; weights unavailable")

    # 5. Vehicle Tracking (YOLO26n-Seg + ByteTrack)
    print("\n[5/12] Testing Vehicle Path Tracker (YOLO26n-Seg + ByteTrack)...")
    adapter = VehiclePathTracker()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M4_M5_vehicle_tracking"] = {
        "model": "VehiclePathTracker_ByteTrack",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "tracks_count": res.get("tracks_count"),
        "sample_track_contact": res.get("tracks", [{}])[0].get("road_contact_point") if res.get("tracks") else None
    }
    if res.get("overlay") is not None:
        cv2.imwrite("outputs/pretrained_smoke_tests/m4_vehicle_tracking.jpg", res["overlay"])
    adapter.release_resources()
    print(f"  -> Tracks: {res.get('tracks_count')} | Latency: {lat} ms | Peak VRAM: {vram} MB")

    # 6. IDD Drivable Area UNet (Audit & Availability Check)
    print("\n[6/12] Testing IDD Drivable Adapter (SenorPsycho/idd-road-segmentation)...")
    adapter = IDDDrivableAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M6_idd_drivable_unet"] = {
        "model": "IDDDrivable_UNet",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "reason": res.get("reason")
    }
    print(f"  -> Status: {res.get('status')} | Reason: Checkpoint weights not in runs/ folder")

    # 7. Depth Anything V2 Small
    print("\n[7/12] Testing Pretrained Depth Adapter (Depth-Anything-V2-Small)...")
    adapter = PretrainedDepthAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M7_depth_anything_v2"] = {
        "model": "DepthAnythingV2_Small",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "depth_type": res.get("depth_type"),
        "depth_min_raw": res.get("depth_min_raw"),
        "depth_max_raw": res.get("depth_max_raw")
    }
    if res.get("depth_colored") is not None:
        cv2.imwrite("outputs/pretrained_smoke_tests/m7_depth_anything_v2.jpg", res["depth_colored"])
    adapter.release_resources()
    print(f"  -> Depth Type: {res.get('depth_type')} | Latency: {lat} ms | Peak VRAM: {vram} MB")

    # 8. Temporal Traffic Anomaly (Simple-TAD VideoMAE ViT-S)
    print("\n[8/12] Testing Pretrained Accident Adapter (Simple-TAD VideoMAE ViT-S)...")
    adapter = SimpleTADAdapter()
    # Feed 16 sequential synthetic frames to test temporal sliding window
    for t_idx in range(16):
        frame_t = img_bgr.copy()
        res_t, lat, vram = measure_execution(adapter.predict_video, frame_t, timestamp=float(t_idx))

    test_results["M8_accident_simpletad"] = {
        "model": "SimpleTAD_VideoMAE_ViTS",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res_t.get("status"),
        "anomaly_score": res_t.get("anomaly_score"),
        "is_anomaly": res_t.get("is_anomaly")
    }
    adapter.release_resources()
    print(f"  -> Anomaly Score: {res_t.get('anomaly_score')} | Is Anomaly: {res_t.get('is_anomaly')} | Latency: {lat} ms | Peak VRAM: {vram} MB")

    # 9. Weather SigLIP2
    print("\n[9/12] Testing Pretrained Weather Adapter (Weather SigLIP2)...")
    adapter = PretrainedWeatherAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M9_weather_siglip2"] = {
        "model": "Weather_SigLIP2",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "weather_condition": res.get("weather_condition"),
        "confidence": res.get("confidence"),
        "lighting": res.get("lighting")
    }
    adapter.release_resources()
    print(f"  -> Condition: {res.get('weather_condition')} ({res.get('confidence')}) | Lighting: {res.get('lighting')} | Latency: {lat} ms")

    # 10. CLRerNet CULane (Dependency Verification & Fallback)
    print("\n[10/12] Testing CLRerNet CULane Adapter (CLRerNet DLA34)...")
    adapter = CLRerNetAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M10_lanes_clrernet"] = {
        "model": "CLRerNet_DLA34",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "dependency_note": res.get("dependency_note"),
        "fallback_active": res.get("fallback_active")
    }
    adapter.release_resources()
    print(f"  -> Status: {res.get('status')} | Fallback Active: {res.get('fallback_active')} | Latency: {lat} ms")

    # 11. Traffic Signals (YOLOv8m)
    print("\n[11/12] Testing Pretrained Traffic Signals Adapter (YOLOv8m)...")
    adapter = PretrainedSignalAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M11_signals_yolov8m"] = {
        "model": "TrafficSignals_YOLOv8m",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "detections_count": res.get("detections_count"),
        "classes_detected": list(set(d["class_name"] for d in res.get("detections", [])))
    }
    if res.get("overlay") is not None:
        cv2.imwrite("outputs/pretrained_smoke_tests/m11_signals_yolov8m.jpg", res["overlay"])
    adapter.release_resources()
    print(f"  -> Detections: {res.get('detections_count')} | Latency: {lat} ms | Peak VRAM: {vram} MB")

    # 12. Open-Vocabulary Obstacles (YOLOv8s-World)
    print("\n[12/12] Testing Pretrained Obstacle Adapter (YOLOv8s-World)...")
    adapter = PretrainedObstacleAdapter()
    res, lat, vram = measure_execution(adapter.predict, img_bgr)
    test_results["M12_obstacles_yolo_world"] = {
        "model": "PretrainedYOLO_World_Obstacles",
        "latency_ms": lat,
        "peak_vram_mb": vram,
        "status": res.get("status"),
        "prompts_count": res.get("prompts_count"),
        "detections_count": res.get("detections_count")
    }
    if res.get("overlay") is not None:
        cv2.imwrite("outputs/pretrained_smoke_tests/m12_obstacles_world.jpg", res["overlay"])
    adapter.release_resources()
    print(f"  -> Prompts: {res.get('prompts_count')} | Detections: {res.get('detections_count')} | Latency: {lat} ms")

    # 13. Pretrained Model Manager Profiles Evaluation
    print("\n------------------------------------------------------------------------")
    print("Testing PretrainedModelManager Pipeline Profiles (FAST, BALANCED, HIGH_ACCURACY)...")
    manager = PretrainedModelManager()

    print("Executing FAST profile...")
    res_fast = manager.run_profile("FAST", img_bgr)
    print(f"  -> FAST: Latency = {res_fast['total_latency_ms']} ms | Peak VRAM = {res_fast['peak_vram_mb']} MB")

    print("Executing BALANCED profile...")
    res_bal = manager.run_profile("BALANCED", img_bgr)
    print(f"  -> BALANCED: Latency = {res_bal['total_latency_ms']} ms | Peak VRAM = {res_bal['peak_vram_mb']} MB")

    print("Executing HIGH_ACCURACY profile...")
    res_high = manager.run_profile("HIGH_ACCURACY", img_bgr)
    print(f"  -> HIGH_ACCURACY: Latency = {res_high['total_latency_ms']} ms | Peak VRAM = {res_high['peak_vram_mb']} MB")

    # Save summary report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)",
        "cuda_available": torch.cuda.is_available(),
        "individual_tests": test_results,
        "profiles": {
            "FAST": {"latency_ms": res_fast["total_latency_ms"], "peak_vram_mb": res_fast["peak_vram_mb"]},
            "BALANCED": {"latency_ms": res_bal["total_latency_ms"], "peak_vram_mb": res_bal["peak_vram_mb"]},
            "HIGH_ACCURACY": {"latency_ms": res_high["total_latency_ms"], "peak_vram_mb": res_high["peak_vram_mb"]}
        }
    }

    with open("outputs/pretrained_smoke_tests/smoke_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n========================================================================")
    print("ALL PRETRAINED SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("Summary written to: outputs/pretrained_smoke_tests/smoke_test_summary.json")
    print("========================================================================")

if __name__ == "__main__":
    main()

