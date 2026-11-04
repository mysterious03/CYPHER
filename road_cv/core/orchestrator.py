"""
CYPhER Master Orchestration Engine (Phase 5 & Phase 6)
Coordinates 15 perception modes, manages sequential GPU execution within 6 GB VRAM,
formats unified JSON output, and invokes explainable multi-model hazard fusion.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import cv2
import numpy as np
import torch

from road_cv.adapters import ADAPTER_MAP
from road_cv.core.fusion import CYPhERHazardFusion
from road_cv.scheduling.resource_scheduler import GPUResourceScheduler

VALID_MODES = [
    "all", "damage", "vehicles", "vehicle_tracking", "road_scene", "water", "depth", "surface",
    "accident", "weather", "lanes", "traffic_signals", "obstacles",
    "friction", "roughness", "pothole_geometry"
]

class CYPhEROrchestrator:
    """
    Unified application engine for the CYPhER Multi-Model Road Perception System.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = GPUResourceScheduler(max_vram_gb=6.0, safety_headroom_gb=1.0)
        self.adapters = {}

    def get_adapter(self, mode: str):
        """Lazy loads requested adapter instance."""
        if mode not in self.adapters:
            adapter_cls = ADAPTER_MAP.get(mode)
            if adapter_cls is None:
                raise ValueError(f"Unknown mode: {mode}. Must be one of {VALID_MODES}")
            self.adapters[mode] = adapter_cls()
        return self.adapters[mode]

    def run(
        self,
        source: Union[str, np.ndarray, int],
        mode: str = "all",
        save_visualizations: bool = True,
        output_dir: str = "outputs",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes perception across one, multiple, or all modes.
        """
        t_start = time.perf_counter()
        os.makedirs(output_dir, exist_ok=True)

        # 1. Image / Frame Ingestion
        is_video = False
        frame = None
        source_name = str(source)

        if isinstance(source, np.ndarray):
            frame = source
            source_name = "numpy_array"
        elif isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # Camera device input
            cap = cv2.VideoCapture(int(source))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError(f"Failed to capture frame from camera index: {source}")
            source_name = f"webcam_{source}"
        elif isinstance(source, str):
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"Input source path does not exist: {source}")

            # Check if video
            if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                is_video = True
                cap = cv2.VideoCapture(str(p))
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    raise RuntimeError(f"Failed to read first frame from video: {source}")
            else:
                frame = cv2.imread(str(p))
                if frame is None:
                    raise RuntimeError(f"Failed to decode image from path: {source}")

        h, w = frame.shape[:2]
        self.scheduler.reset_peak_memory()

        # 2. Determine Modes to Execute
        if mode == "all":
            # Execute all compatible vision modes
            selected_modes = [
                "road_scene", "weather", "surface", "damage", "vehicles",
                "water", "depth", "lanes", "traffic_signals", "obstacles",
                "accident", "pothole_geometry", "friction", "roughness"
            ]
        elif "," in mode:
            selected_modes = [m.strip() for m in mode.split(",") if m.strip() in VALID_MODES]
        else:
            if mode not in VALID_MODES:
                raise ValueError(f"Invalid mode '{mode}'. Choose from: {VALID_MODES}")
            selected_modes = [mode]

        # 3. Sequential Execution with Cross-Modal Context Sharing
        capabilities_out = {}
        warnings = []
        shared_context = {
            "road_mask": None,
            "depth_map": None,
            "damage": [],
            "vehicles": [],
            "surface": {},
            "is_video": is_video
        }

        for m in selected_modes:
            try:
                adapter = self.get_adapter(m)

                # Prepare shared context inputs
                call_kwargs = dict(kwargs)
                call_kwargs["is_video"] = is_video

                if m in ["weather", "damage", "lanes", "obstacles"]:
                    call_kwargs["road_mask"] = shared_context["road_mask"]
                if m == "water":
                    call_kwargs["road_mask"] = shared_context["road_mask"]
                    call_kwargs["detected_potholes"] = shared_context["damage"]
                if m == "pothole_geometry":
                    call_kwargs["damage"] = shared_context["damage"]
                    call_kwargs["depth_map"] = shared_context["depth_map"]
                if m in ["friction", "roughness"]:
                    call_kwargs["surface"] = shared_context["surface"]
                if m == "accident":
                    call_kwargs["vehicles"] = shared_context["vehicles"]

                # Run prediction
                res = adapter.predict(frame, **call_kwargs)
                capabilities_out[m] = res

                # Update shared context for downstream models
                if m == "road_scene" and "road_mask" in res:
                    shared_context["road_mask"] = res["road_mask"]
                if m == "depth" and "depth_map" in res:
                    shared_context["depth_map"] = res["depth_map"]
                if m == "damage":
                    shared_context["damage"] = res.get("predictions", [])
                if m == "vehicles":
                    shared_context["vehicles"] = res.get("predictions", [])
                if m == "surface":
                    shared_context["surface"] = res.get("predictions", {})

                # Sequential memory cleanup if GPU allocated
                if torch.cuda.is_available() and self.scheduler.get_memory_status()["allocated_gb"] > 4.0:
                    adapter.unload()

            except Exception as e:
                warnings.append(f"Module '{m}' encountered an error: {str(e)}")
                capabilities_out[m] = {
                    "mode": m,
                    "status": "failed",
                    "error": str(e),
                    "predictions": []
                }

        # 4. Multi-Model Hazard Fusion (Phase 7)
        fusion_summary = CYPhERHazardFusion.fuse(capabilities_out)

        # 5. Measure System Performance
        total_latency = (time.perf_counter() - t_start) * 1000.0
        peak_vram = self.scheduler.get_memory_status()["peak_allocated_gb"] * 1024.0

        # Remove raw non-serializable ndarrays from JSON output
        clean_capabilities = {}
        for k, v in capabilities_out.items():
            clean_v = dict(v)
            clean_v.pop("road_mask", None)
            clean_v.pop("depth_map", None)
            clean_capabilities[k] = clean_v

        # 6. Build Standard JSON Output Schema (Phase 6)
        output_schema = {
            "source": source_name,
            "timestamp": datetime.now().isoformat(),
            "image_size": {
                "width": int(w),
                "height": int(h)
            },
            "selected_mode": mode,
            "executed_modes_count": len(selected_modes),
            "capabilities": clean_capabilities,
            "fusion": fusion_summary,
            "warnings": warnings,
            "performance": {
                "total_pipeline_latency_ms": round(total_latency, 2),
                "fps": round(1000.0 / max(1.0, total_latency), 2),
                "peak_gpu_memory_mb": round(peak_vram, 2),
                "device": self.device
            }
        }

        # 7. Render Composite Output Visualization
        if save_visualizations and frame is not None:
            vis_img = self._render_composite_overlay(frame, clean_capabilities, fusion_summary)
            out_vis_path = os.path.join(output_dir, f"cypher_{Path(source_name).stem}_overlay.jpg")
            cv2.imwrite(out_vis_path, vis_img)
            output_schema["visualization_path"] = out_vis_path

        return output_schema

    def _render_composite_overlay(
        self,
        frame: np.ndarray,
        capabilities: Dict[str, Any],
        fusion: Dict[str, Any]
    ) -> np.ndarray:
        """Draws annotations and HUD telemetry."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw damage boxes (Yellow)
        for d in capabilities.get("damage", {}).get("predictions", []):
            box = d.get("box", [0, 0, 0, 0])
            cname = d.get("class_name", "defect")
            cf = d.get("confidence", 0)
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
            cv2.putText(vis, f"{cname} {cf:.2f}", (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw vehicles (Green)
        for v in capabilities.get("vehicles", {}).get("predictions", []):
            box = v.get("box", [0, 0, 0, 0])
            tid = v.get("track_id", "")
            cname = v.get("class_name", "vehicle")
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(vis, f"ID:{tid} {cname}", (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw top HUD banner
        risk = fusion.get("overall_road_safety_risk", "LOW_RISK")
        risk_color = (0, 255, 0) if risk == "LOW_RISK" else ((0, 165, 255) if "MODERATE" in risk else (0, 0, 255))

        cv2.rectangle(vis, (0, 0), (w, 50), (20, 20, 20), -1)
        hud_text = f"CYPhER ROAD PERCEPTION | RISK: {risk} | HAZARDS: {fusion.get('detected_hazards_count', 0)}"
        cv2.putText(vis, hud_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)

        return vis

