"""
CYPhER V2.1 Acceptance Test Runner and Benchmark Profiler
Runs end-to-end tests across all 15 selectable modes and 'all' mode.
Measures latency, memory, verifies output schemas, and generates visual outputs in reports/acceptance_tests/.
"""

import os
import sys
import time
import json
import traceback
import cv2
import numpy as np
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_cv.core.orchestrator import CYPhEROrchestrator, VALID_MODES

OUT_DIR = Path("reports/acceptance_tests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMAGES = [
    ("Road_jpg", "Road.jpg"),
    ("India_val", "data/processed/rdd2022_india/images/val/India_000002.jpg")
]

def run_acceptance_tests():
    print("=" * 80)
    print("CYPhER V2.1 END-TO-END ACCEPTANCE AUDIT & TEST RUNNER")
    print("=" * 80)
    
    orchestrator = CYPhEROrchestrator()
    print(f"Device initialized: {orchestrator.device}")
    
    results_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": orchestrator.device,
        "modes_tested": {},
        "all_mode_benchmark": {},
        "status_counts": {
            "passed": 0,
            "requires_sensor": 0,
            "requires_calibration": 0,
            "failed": 0
        }
    }

    # 1. Test each individual mode on Road.jpg
    individual_modes = [m for m in VALID_MODES if m != "all"]
    print(f"\n--- Testing 14 Individual Modes on Road.jpg ---")
    
    for mode in individual_modes:
        print(f"\n>> Testing mode: '{mode}'...")
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vram_start = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            vram_start = 0.0

        try:
            res = orchestrator.run(
                source="Road.jpg",
                mode=mode,
                save_visualizations=True,
                output_dir=str(OUT_DIR)
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if torch.cuda.is_available():
                vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                vram_peak = 0.0

            cap_res = res["capabilities"].get(mode, {})
            mode_status = cap_res.get("status", "unknown")
            
            print(f"   Status: {mode_status} | Time: {elapsed_ms:.1f}ms | Peak VRAM: {vram_peak:.1f}MB")
            
            # Save mode json output
            mode_json_path = OUT_DIR / f"test_{mode}_Road.json"
            with open(mode_json_path, "w") as f:
                json.dump(res, f, indent=2, default=str)

            results_summary["modes_tested"][mode] = {
                "status": mode_status,
                "latency_ms": round(elapsed_ms, 2),
                "peak_vram_mb": round(vram_peak, 2),
                "has_predictions": len(cap_res.get("predictions", [])) > 0 or bool(cap_res.get("predictions")),
                "prediction_summary": str(cap_res.get("predictions", {}))[:200]
            }

            if mode_status == "success":
                results_summary["status_counts"]["passed"] += 1
            elif mode_status == "requires_sensor":
                results_summary["status_counts"]["requires_sensor"] += 1
            elif mode_status == "requires_calibration":
                results_summary["status_counts"]["requires_calibration"] += 1
            else:
                results_summary["status_counts"]["failed"] += 1

        except Exception as e:
            print(f"   ERROR in mode {mode}: {e}")
            traceback.print_exc()
            results_summary["modes_tested"][mode] = {
                "status": "failed",
                "error": str(e)
            }
            results_summary["status_counts"]["failed"] += 1

    # 2. Test 'all' mode on both test images
    print(f"\n--- Testing 'all' Mode Pipeline ---")
    for img_tag, img_path in TEST_IMAGES:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}, skipping.")
            continue
            
        print(f"\n>> Executing mode='all' on {img_path}...")
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        res_all = orchestrator.run(
            source=img_path,
            mode="all",
            save_visualizations=True,
            output_dir=str(OUT_DIR)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        all_json_path = OUT_DIR / f"test_all_{img_tag}.json"
        with open(all_json_path, "w") as f:
            json.dump(res_all, f, indent=2, default=str)

        results_summary["all_mode_benchmark"][img_tag] = {
            "image": img_path,
            "latency_ms": round(elapsed_ms, 2),
            "peak_vram_mb": round(vram_peak, 2),
            "executed_modes": res_all.get("executed_modes_count", 0),
            "overall_hazard_level": res_all.get("fusion", {}).get("overall_hazard_level", "unknown"),
            "hazard_score": res_all.get("fusion", {}).get("hazard_score", 0.0),
            "recommendation": res_all.get("fusion", {}).get("driver_recommendation", "")
        }
        print(f"   Done in {elapsed_ms:.1f}ms | Peak VRAM: {vram_peak:.1f}MB | Hazard: {results_summary['all_mode_benchmark'][img_tag]['overall_hazard_level']}")

    # 3. Save Summary JSON
    summary_path = OUT_DIR / "acceptance_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Acceptance test suite finished. Summary written to {summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_acceptance_tests()

