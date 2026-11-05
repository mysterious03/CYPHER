"""
CYPhER Web Application & Minimalist Perception Frontend
FastAPI + Uvicorn backend serving a clean, responsive single-page dashboard.
Supports:
- Drag & Drop photo analysis with side-by-side original/annotated view.
- Video file upload and keyframe/temporal tracking analysis.
- Real-time live camera streaming (IP Webcam Wi-Fi / USB / Webcam).
- Comprehensive safety scoring, hazard breakdown cards, and latency telemetry.

Usage:
    python app.py
    # Then open http://127.0.0.1:8000 in your browser
"""

import os
import sys
import io
import time
import base64
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path and is current working directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from road_cv.core.orchestrator import CYPhEROrchestrator, VALID_MODES

app = FastAPI(title="CYPhER Web Dashboard", description="Minimalist Road Perception & Hazard Intelligence UI")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance (lazily loaded on startup)
orchestrator: Optional[CYPhEROrchestrator] = None

# Active live stream state
stream_active = False
stream_telemetry = {
    "fusion": {"overall_road_safety_risk": "LOW_RISK", "fusion_confidence": 0.95, "detected_hazards_count": 0, "hazards": []},
    "performance": {"total_pipeline_latency_ms": 0, "fps": 0},
    "capabilities": {}
}


def get_orchestrator() -> CYPhEROrchestrator:
    global orchestrator
    if orchestrator is None:
        print("[CYPhER Web] Initializing CYPhER Orchestrator...")
        orchestrator = CYPhEROrchestrator()
    return orchestrator


def encode_image_base64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image into JPEG base64 string."""
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the minimalist frontend dashboard."""
    html_file = PROJECT_ROOT / "web" / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    with open(html_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/simulator", response_class=HTMLResponse)
async def serve_simulator():
    """Serve the full 3D physics simulation world (Three.js + Cannon-es)."""
    sim_file = PROJECT_ROOT / "web" / "sim_world.html"
    if not sim_file.exists():
        raise HTTPException(status_code=404, detail="sim_world.html not found.")
    with open(sim_file, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    mode: str = Form("all")
):
    """Analyze an uploaded road photo."""
    orch = get_orchestrator()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported image format.")

    # Run CYPhER inference
    results = orch.run(source=frame, mode=mode, save_visualizations=False)
    clean_caps = results.get("capabilities", {})
    fusion = results.get("fusion", {})

    # Render high-contrast composite overlay
    overlay_img = orch._render_composite_overlay(frame, clean_caps, fusion)

    # Clean non-serializable objects from output
    clean_schema = dict(results)
    clean_schema["overlay_image_base64"] = encode_image_base64(overlay_img)
    clean_schema["original_image_base64"] = encode_image_base64(frame)

    return JSONResponse(content=clean_schema)


@app.get("/api/sample")
async def analyze_sample(mode: str = Query("all")):
    """Run analysis on sample Road.jpg file."""
    orch = get_orchestrator()
    sample_path = PROJECT_ROOT / "Road.jpg"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample Road.jpg not found on disk.")

    frame = cv2.imread(str(sample_path))
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to read sample image.")

    results = orch.run(source=frame, mode=mode, save_visualizations=False)
    clean_caps = results.get("capabilities", {})
    fusion = results.get("fusion", {})
    overlay_img = orch._render_composite_overlay(frame, clean_caps, fusion)

    clean_schema = dict(results)
    clean_schema["overlay_image_base64"] = encode_image_base64(overlay_img)
    clean_schema["original_image_base64"] = encode_image_base64(frame)

    return JSONResponse(content=clean_schema)


@app.post("/api/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    mode: str = Form("vehicles")
):
    """Analyze key frames from an uploaded dashcam video."""
    orch = get_orchestrator()
    
    # Save temporary uploaded video file
    temp_dir = PROJECT_ROOT / "outputs" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to decode uploaded video stream.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Grab middle representative frame
    target_frame_idx = max(0, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()
    cap.release()

    try:
        temp_path.unlink() # Cleanup temp file
    except Exception:
        pass

    if not ret or frame is None:
        raise HTTPException(status_code=400, detail="Failed to extract frames from video.")

    results = orch.run(source=frame, mode=mode, save_visualizations=False, is_video=True)
    clean_caps = results.get("capabilities", {})
    fusion = results.get("fusion", {})
    overlay_img = orch._render_composite_overlay(frame, clean_caps, fusion)

    clean_schema = dict(results)
    clean_schema["video_info"] = {
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "analyzed_frame_index": target_frame_idx
    }
    clean_schema["overlay_image_base64"] = encode_image_base64(overlay_img)
    clean_schema["original_image_base64"] = encode_image_base64(frame)

    return JSONResponse(content=clean_schema)


@app.get("/api/stream")
def stream_feed(
    source: str = Query("0"),
    mode: str = Query("all")
):
    """Continuous MJPEG streaming endpoint for phone IP webcam or USB webcam."""
    global stream_active, stream_telemetry
    stream_active = True
    orch = get_orchestrator()

    # Determine input capture source & normalize IP webcam URLs
    target_src = source.strip()
    if target_src.isdigit():
        cap = cv2.VideoCapture(int(target_src), cv2.CAP_DSHOW)
    else:
        # Prepend http:// if missing
        if not target_src.startswith("http://") and not target_src.startswith("https://"):
            target_src = "http://" + target_src
        # Append /video if user typed IP:port without endpoint
        if target_src.startswith("http://") or target_src.startswith("https://"):
            if not target_src.endswith("/video") and not target_src.endswith(".mjpg") and not target_src.endswith("/shot.jpg") and not target_src.endswith(".mp4"):
                target_src = target_src.rstrip("/") + "/video"
        
        cap = cv2.VideoCapture(target_src)
        if not cap.isOpened():
            # Fallback to literal source parameter
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Cannot connect to camera at '{source}'. Attempted: '{target_src}'.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def frame_generator():
        global stream_active, stream_telemetry
        t_prev = time.perf_counter()
        fps_hist = []

        try:
            while stream_active:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.04)
                    continue

                # Optimize high-res phone captures (e.g. 1440p or 1080p) for real-time GPU processing
                h, w = frame.shape[:2]
                if max(h, w) > 960:
                    scale = 960.0 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                # Run inference on frame
                t0 = time.perf_counter()
                results = orch.run(source=frame, mode=mode, save_visualizations=False, is_video=True)
                latency = (time.perf_counter() - t0) * 1000.0

                clean_caps = results.get("capabilities", {})
                fusion = results.get("fusion", {})

                # Update stream telemetry
                t_now = time.perf_counter()
                inst_fps = 1.0 / max(1e-4, t_now - t_prev)
                t_prev = t_now
                fps_hist.append(inst_fps)
                if len(fps_hist) > 10:
                    fps_hist.pop(0)

                stream_telemetry = {
                    "fusion": fusion,
                    "performance": {
                        "total_pipeline_latency_ms": round(latency, 1),
                        "fps": round(sum(fps_hist) / len(fps_hist), 1)
                    },
                    "capabilities": clean_caps
                }

                # Render composite visual
                vis = orch._render_composite_overlay(frame, clean_caps, fusion)

                # JPEG encode
                ret_enc, jpeg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret_enc:
                    continue

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

        finally:
            cap.release()

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/stream-telemetry")
async def get_stream_telemetry():
    """Poll latest stream telemetry while video is streaming."""
    return JSONResponse(content=stream_telemetry)


@app.get("/api/stop-stream")
async def stop_stream():
    """Stop live camera stream."""
    global stream_active
    stream_active = False
    return {"status": "stopped"}


def main():
    print("=" * 72)
    print("  CYPhER: MINIMALIST ROAD PERCEPTION & HAZARD DASHBOARD")
    print("=" * 72)
    print("  Local URL: http://127.0.0.1:8000")
    print("  Press Ctrl+C in this terminal to shut down.")
    print("=" * 72)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()

