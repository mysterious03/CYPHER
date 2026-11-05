"""
CYPhER: Real-Time Live Camera Streaming & Multi-Model Vision Perception Pipeline
Connects to:
- Mobile Phone Camera via IP Webcam / DroidCam / RTSP URL (e.g. http://192.168.1.X:8080/video)
- Mobile Phone Camera via USB / Virtual Webcam (e.g. DroidCam / Iriun device index 0, 1, 2)
- Built-in Laptop / USB Webcams

Usage:
    # Real-time multi-hazard perception (Vehicles + Damage + Water + Accident):
    python stream.py --source http://192.168.0.213:8080/video --mode all

    # Vehicle detection & tracking with custom class colors:
    python stream.py --source http://192.168.0.213:8080/video --mode vehicles

    # Fast multi-hazard mode:
    python stream.py --source http://192.168.0.213:8080/video --mode vehicles,damage,water,accident
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

# Ensure workspace is in sys.path and working directory is project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from road_cv.core.orchestrator import CYPhEROrchestrator, VALID_MODES


def scan_available_cameras(max_tested: int = 4) -> List[int]:
    """Scan and return indices of active local webcam devices."""
    available = []
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


def draw_hud(
    frame: np.ndarray,
    mode: str,
    fps: float,
    latency_ms: float,
    risk: str,
    hazards_count: int,
    status_text: str = ""
) -> np.ndarray:
    """Draw a clean, resolution-adaptive HUD overlay on the live frame."""
    h, w = frame.shape[:2]
    vis = frame.copy()
    scale = max(0.5, min(2.5, w / 960.0))

    hud_h = int(52 * scale)
    bottom_h = int(32 * scale)
    font_scale = max(0.42, 0.52 * scale)
    font_thick = max(1, int(round(1.8 * scale)))

    # Semi-transparent top and bottom bars
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 15, 20), -1)
    cv2.rectangle(overlay, (0, h - bottom_h), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)

    # Color code for overall risk
    if "HIGH" in risk or "CRITICAL" in risk:
        risk_color = (30, 30, 255)   # Red
    elif "MODERATE" in risk:
        risk_color = (0, 165, 255)  # Orange
    else:
        risk_color = (0, 255, 100)  # Bright green

    # Header telemetry
    t_y1 = int(24 * scale)
    t_y2 = int(44 * scale)
    cv2.putText(vis, "CYPhER MULTI-PERCEPTION", (int(12 * scale), t_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, (0, 255, 255), font_thick, cv2.LINE_AA)
    cv2.putText(vis, f"MODE: {mode.upper()}", (int(12 * scale), t_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (220, 220, 220), 1, cv2.LINE_AA)

    # Center Risk & Hazards
    risk_text = f"RISK: {risk} | HAZARDS: {hazards_count}"
    (r_w, _), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, font_thick)
    cv2.putText(vis, risk_text, (max(int(220 * scale), w // 2 - r_w // 2), int(33 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, risk_color, font_thick, cv2.LINE_AA)

    # Top Right Performance
    perf_str = f"{fps:.1f} FPS | {latency_ms:.0f} ms"
    (p_w, _), _ = cv2.getTextSize(perf_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    perf_color = (0, 255, 100) if fps >= 15 else (0, 180, 255)
    cv2.putText(vis, perf_str, (w - p_w - int(12 * scale), int(33 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, perf_color, font_thick, cv2.LINE_AA)

    # Bottom helper bar
    hotkeys = "[M] Mode  |  [S] Save Snapshot  |  [P] Pause  |  [Q] Exit"
    if status_text:
        hotkeys = f"{status_text}   {hotkeys}"
    cv2.putText(vis, hotkeys, (int(12 * scale), h - int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def open_camera_stream(source: str):
    """
    Open camera stream from either:
    - Device index (e.g. 0, 1)
    - IP URL (e.g. http://192.168.0.213:8080/video or rtsp://...)
    """
    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to connect to camera stream at: '{source}'")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def run_live_stream(
    source: str = "0",
    mode: str = "all",
    output_dir: str = "outputs/stream_captures",
    skip_frames: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None
):
    """Run real-time vision inference loop on live camera feed."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  CYPhER REAL-TIME LIVE CAMERA PERCEPTION ENGINE")
    print("=" * 72)
    print(f"Connecting to source: {source}")
    print(f"Active vision mode:   {mode}")
    print(f"Snapshot directory:   {output_dir}")
    print("=" * 72)

    try:
        cap = open_camera_stream(source)
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        cams = scan_available_cameras()
        if cams:
            print(f"[INFO] Found local camera devices at indices: {cams}")
            print(f"[INFO] Try running: python stream.py --source {cams[0]}")
        else:
            print("[INFO] No local webcam indices detected.")
            print("[INFO] If using a phone, install 'IP Webcam' on Android and pass:")
            print("       python stream.py --source http://<phone-ip>:8080/video")
        return

    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("[INFO] Stream initialized successfully. Loading CYPhER Orchestrator...")
    orchestrator = CYPhEROrchestrator()
    print("[INFO] Vision engine ready! Press 'Q' on the video window to stop.")

    window_name = "CYPhER Live Road Perception"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    # Preset mode toggle list with comprehensive multi-hazard support
    mode_presets = [
        "all",
        "vehicles,damage,water,accident",
        "vehicles",
        "damage",
        "water",
        "accident",
        "surface",
        "traffic_signals"
    ]
    current_mode_idx = 0
    if mode in mode_presets:
        current_mode_idx = mode_presets.index(mode)
    current_mode = mode

    frame_counter = 0
    last_annotated_frame = None
    last_risk = "LOW_RISK"
    last_hazards = 0
    last_latency = 0.0
    paused = False
    status_msg = ""
    status_timer = 0

    fps_history = []
    t_prev = time.perf_counter()

    try:
        while True:
            t_loop_start = time.perf_counter()

            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\n[WARNING] Frame drop or stream disconnected. Retrying...")
                    time.sleep(0.05)
                    continue

                frame_counter += 1

                # Perform model inference on specified frame cadence
                if (skip_frames == 0) or (frame_counter % (skip_frames + 1) == 1):
                    t_infer_start = time.perf_counter()

                    # Run CYPhER inference on frame with video tracking context
                    result = orchestrator.run(
                        source=frame,
                        mode=current_mode,
                        save_visualizations=False,
                        is_video=True
                    )

                    last_latency = (time.perf_counter() - t_infer_start) * 1000.0
                    clean_caps = result.get("capabilities", {})
                    fusion = result.get("fusion", {})

                    last_risk = fusion.get("overall_road_safety_risk", "LOW_RISK")
                    last_hazards = fusion.get("detected_hazards_count", 0)

                    # Render bold, distinct multi-class overlays
                    vis = orchestrator._render_composite_overlay(frame, clean_caps, fusion)
                    last_annotated_frame = vis
                else:
                    # Maintain display frame
                    vis = frame.copy()
            else:
                vis = last_annotated_frame.copy() if last_annotated_frame is not None else frame.copy()

            # Calculate FPS
            t_now = time.perf_counter()
            instant_fps = 1.0 / max(1e-4, (t_now - t_prev))
            t_prev = t_now
            fps_history.append(instant_fps)
            if len(fps_history) > 15:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Clear temporary status message after 3 seconds
            if status_msg and time.time() > status_timer:
                status_msg = ""

            # Draw HUD
            display_frame = draw_hud(
                last_annotated_frame if last_annotated_frame is not None else vis,
                mode=current_mode,
                fps=avg_fps,
                latency_ms=last_latency,
                risk=last_risk,
                hazards_count=last_hazards,
                status_text=status_msg
            )

            cv2.imshow(window_name, display_frame)

            # Process key events (wait 1ms)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n[INFO] Exiting live stream...")
                break
            elif key == ord('s'):  # Save snapshot
                snap_id = int(time.time())
                snap_path = os.path.join(output_dir, f"snapshot_{snap_id}.jpg")
                cv2.imwrite(snap_path, display_frame)
                status_msg = f"[SAVED: {os.path.basename(snap_path)}]"
                status_timer = time.time() + 3.0
                print(f"[INFO] Snapshot saved to {snap_path}")
            elif key == ord('m'):  # Cycle mode preset
                current_mode_idx = (current_mode_idx + 1) % len(mode_presets)
                current_mode = mode_presets[current_mode_idx]
                status_msg = f"[MODE: {current_mode.upper()}]"
                status_timer = time.time() + 3.0
                print(f"[INFO] Perception mode switched to: {current_mode}")
            elif key == ord('p'):  # Pause / Unpause
                paused = not paused
                status_msg = "[PAUSED]" if paused else "[RESUMED]"
                status_timer = time.time() + 2.0

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera stream closed.")


def main():
    parser = argparse.ArgumentParser(
        description="CYPhER Live Camera Feed & Real-Time Multi-Model Perception"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera device index (e.g. '0', '1') or phone IP stream URL (e.g. 'http://192.168.0.213:8080/video')"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Perception mode: 'all', 'vehicles,damage,water,accident', 'vehicles', 'damage', 'water', 'accident', etc."
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Inference frame stride (0 = infer every frame, 1 = every 2nd frame for higher FPS)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stream_captures",
        help="Directory to save snapshots"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional capture frame width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional capture frame height"
    )

    args = parser.parse_args()
    run_live_stream(
        source=args.source,
        mode=args.mode,
        output_dir=args.output_dir,
        skip_frames=args.skip_frames,
        width=args.width,
        height=args.height
    )


if __name__ == "__main__":
    main()
