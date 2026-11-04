"""
CYPhER: Multi-Model Road Perception and Hazard Intelligence System
Main Application Entry Point (Phase 5)

Supports:
- Command-line execution:
    python main.py --mode damage --source Road.jpg
    python main.py --mode all --source Road.jpg
    python main.py --mode vehicles --source test_video.mp4
    python main.py --mode accident --source test_video.mp4
    python main.py --mode all --source 0
- Interactive mode selector when run without arguments.
- Standard JSON schema output and composite annotated visualization.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import cv2

# Add workspace to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from road_cv.core.orchestrator import CYPhEROrchestrator, VALID_MODES

def print_banner():
    banner = r"""
========================================================================
   ____ ____    _     ____  _       ___  ____  
  / ___|  _ \  / \   |  _ \| |     / _ \|  _ \ 
 | |   | |_) |/ _ \  | |_) | |    | | | | |_) |
 | |___|  __/ / ___ \ |  _ <| |___ | |_| |  _ < 
  \____|_|   /_/   \_\|_| \_\_____(_)___/|_| \_\
========================================================================
 MULTI-MODEL ROAD PERCEPTION & HAZARD INTELLIGENCE SYSTEM (v2.0)
 Target Hardware: NVIDIA RTX 4050 6GB VRAM | Windows 11 | PyTorch CUDA
========================================================================
"""
    print(banner)

def interactive_mode_selector() -> tuple:
    print_banner()
    print("Available Perception Modes:")
    for idx, m in enumerate(VALID_MODES, 1):
        print(f"  [{idx:2d}] {m}")

    print("\nSelect mode(s):")
    print("  - Type '1' for all modes")
    print("  - Type a number (e.g. '2' for damage)")
    print("  - Type comma-separated names (e.g. 'damage,surface,water')")

    choice = input("\nEnter selection [default: all]: ").strip()
    if not choice:
        selected_mode = "all"
    elif choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(VALID_MODES):
            selected_mode = VALID_MODES[idx - 1]
        else:
            selected_mode = "all"
    else:
        selected_mode = choice

    source = input("Enter input source (image path, video path, or '0' for camera) [default: Road.jpg]: ").strip()
    if not source:
        source = "Road.jpg"

    return selected_mode, source

def main():
    parser = argparse.ArgumentParser(
        description="CYPhER Multi-Model Road Perception and Hazard Intelligence System"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help=f"Perception mode to execute. Choices: {', '.join(VALID_MODES)}"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input image path, video path, or camera device index (e.g. 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output JSON and visualization images (default: outputs)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save structured JSON output to disk"
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable rendering composite image overlay"
    )

    args = parser.parse_args()

    # If no arguments provided, launch interactive selector
    if args.mode is None and args.source is None:
        selected_mode, source_path = interactive_mode_selector()
    else:
        selected_mode = args.mode or "all"
        source_path = args.source or "Road.jpg"

    # Default to available test image if default Road.jpg not found
    if source_path == "Road.jpg" and not Path("Road.jpg").exists():
        val_imgs = list(Path("data/processed/rdd2022_india/images/val").glob("*.jpg"))
        if val_imgs:
            source_path = str(val_imgs[0])

    print_banner()
    print(f"[CYPhER Engine] Target Mode:   {selected_mode}")
    print(f"[CYPhER Engine] Input Source:  {source_path}")
    print(f"[CYPhER Engine] Output Folder: {args.output_dir}\n")

    orchestrator = CYPhEROrchestrator()

    try:
        results = orchestrator.run(
            source=source_path,
            mode=selected_mode,
            save_visualizations=not args.no_vis,
            output_dir=args.output_dir
        )

        # Print structured summary to console
        print("-" * 72)
        print("CYPhER PERCEPTION EXECUTION SUMMARY")
        print("-" * 72)
        perf = results.get("performance", {})
        print(f"Latency:        {perf.get('total_pipeline_latency_ms', 0)} ms (~{perf.get('fps', 0)} FPS)")
        print(f"Peak GPU VRAM:  {perf.get('peak_gpu_memory_mb', 0)} MB (Device: {perf.get('device', 'cpu')})")
        print(f"Modes Executed: {results.get('executed_modes_count', 0)}")

        fusion = results.get("fusion", {})
        print(f"\nOverall Safety Risk: {fusion.get('overall_road_safety_risk', 'UNKNOWN')}")
        print(f"Fusion Confidence:   {fusion.get('fusion_confidence', 0):.2f}")
        print(f"Hazards Detected:    {fusion.get('detected_hazards_count', 0)}")

        for h in fusion.get("hazards", []):
            print(f"  [!] {h.get('hazard_type')}: {h.get('reason')}")

        if "visualization_path" in results:
            print(f"\nVisualization Saved: {results['visualization_path']}")

        # Save full structured JSON
        out_json_path = os.path.join(args.output_dir, f"cypher_{Path(str(source_path)).stem}_result.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"JSON Result Saved:   {out_json_path}")
        print("-" * 72)

    except Exception as e:
        print(f"\n[ERROR] CYPhER Execution Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
