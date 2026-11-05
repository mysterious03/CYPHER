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
        """
        Draws high-contrast, bold bounding boxes with solid filled label badges,
        distinct colors for every vehicle type and defect, water/puddle tinting,
        traffic signals, accident warnings, and real-time road condition telemetry.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Resolution-adaptive scaling for high-res inputs (e.g. 1080p, 1440p)
        scale = max(0.5, min(2.5, w / 960.0))
        box_thick = max(3, int(round(3.5 * scale)))
        font_scale = max(0.45, 0.55 * scale)
        font_thick = max(1, int(round(2.0 * scale)))

        def draw_badge_box(img, box, label, color_bgr, text_color=(255, 255, 255)):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                return

            # Thick bounding rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, box_thick)

            # High-tech corner highlights
            c_len = min(int(15 * scale), (x2 - x1) // 3, (y2 - y1) // 3)
            if c_len > 4:
                cv2.line(img, (x1, y1), (x1 + c_len, y1), color_bgr, box_thick + 2)
                cv2.line(img, (x1, y1), (x1, y1 + c_len), color_bgr, box_thick + 2)
                cv2.line(img, (x2, y1), (x2 - c_len, y1), color_bgr, box_thick + 2)
                cv2.line(img, (x2, y1), (x2, y1 + c_len), color_bgr, box_thick + 2)
                cv2.line(img, (x1, y2), (x1 + c_len, y2), color_bgr, box_thick + 2)
                cv2.line(img, (x1, y2), (x1, y2 - c_len), color_bgr, box_thick + 2)
                cv2.line(img, (x2, y2), (x2 - c_len, y2), color_bgr, box_thick + 2)
                cv2.line(img, (x2, y2), (x2, y2 - c_len), color_bgr, box_thick + 2)

            # Solid filled badge background for clear readability
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            badge_h = th + int(8 * scale)
            badge_y1 = max(0, y1 - badge_h)
            badge_y2 = y1 if y1 >= badge_h else y1 + badge_h
            badge_x2 = min(w, x1 + tw + int(10 * scale))

            cv2.rectangle(img, (x1, badge_y1), (badge_x2, badge_y2), color_bgr, -1)
            text_y = badge_y2 - int(4 * scale)
            cv2.putText(
                img, label, (x1 + int(4 * scale), text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thick, cv2.LINE_AA
            )

        # 1. WATER & PUDDLES
        water_cap = capabilities.get("water", {})
        if water_cap.get("status") == "success":
            preds = water_cap.get("predictions", {})
            puddles = preds.get("puddles", [])
            for p in puddles:
                pbox = p.get("box", [0, 0, 0, 0])
                pconf = p.get("confidence", 0)
                draw_badge_box(vis, pbox, f"PUDDLE {pconf:.2f}", (255, 210, 0), (10, 10, 10))

            if preds.get("water_detected", False) or preds.get("water_severity", "NONE") != "NONE":
                pct = preds.get("pct_road_submerged", preds.get("coverage_ratio", 0) * 100)
                water_banner = f"WATERLOGGED ROAD: {pct:.1f}% SUBMERGED"
                (w_w, w_h), _ = cv2.getTextSize(water_banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, font_thick)
                cv2.rectangle(vis, (15, h - int(75 * scale)), (25 + w_w, h - int(45 * scale)), (255, 160, 0), -1)
                cv2.putText(vis, water_banner, (20, h - int(52 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, (20, 20, 20), font_thick, cv2.LINE_AA)

        # 2. ROAD DAMAGE (Cracks & Potholes with distinct hazard colors)
        DAMAGE_COLORS = {
            "pothole": (30, 30, 255),           # Crimson Red
            "d40": (30, 30, 255),
            "alligator crack": (0, 100, 255),   # Deep Orange
            "d20": (0, 100, 255),
            "longitudinal crack": (0, 180, 255),# Vivid Amber
            "d00": (0, 180, 255),
            "transverse crack": (0, 220, 255),  # Bright Gold
            "d10": (0, 220, 255),
            "defect": (0, 165, 255)
        }
        for d in capabilities.get("damage", {}).get("predictions", []):
            box = d.get("box", [0, 0, 0, 0])
            cname = str(d.get("class_name", "defect")).lower()
            cf = d.get("confidence", 0)
            col = DAMAGE_COLORS.get(cname, (0, 165, 255))
            draw_badge_box(vis, box, f"{cname.upper()} {cf:.2f}", col, (255, 255, 255))

        # 3. VEHICLES & ROAD USERS (Distinct colors per vehicle class)
        VEHICLE_COLORS = {
            "car": (255, 190, 0),           # Bright Cyan-Blue
            "truck": (0, 130, 255),         # Deep Orange
            "bus": (220, 20, 220),          # Neon Purple / Magenta
            "motorcycle": (0, 230, 255),    # Electric Yellow
            "bicycle": (80, 240, 100),      # Spring Green
            "person": (120, 80, 255),       # Coral Pink
            "pedestrian": (120, 80, 255),
            "auto_rickshaw": (0, 190, 255), # Indian Rickshaw Gold
            "vehicle": (0, 255, 120)        # Emerald
        }
        for v in capabilities.get("vehicles", {}).get("predictions", []):
            box = v.get("box", [0, 0, 0, 0])
            tid = v.get("track_id", "")
            cname = str(v.get("class_name", "vehicle")).lower()
            cf = v.get("confidence", 0)
            col = VEHICLE_COLORS.get(cname, (0, 255, 120))
            tag = f"{cname.upper()} #{tid} ({cf:.2f})" if tid else f"{cname.upper()} ({cf:.2f})"
            is_light_bg = col in [(0, 230, 255), (80, 240, 100), (0, 255, 120), (255, 190, 0)]
            draw_badge_box(vis, box, tag, col, (10, 10, 10) if is_light_bg else (255, 255, 255))

        # 4. TRAFFIC SIGNALS
        sig_data = capabilities.get("traffic_signals", {}).get("predictions", {})
        sig_list = sig_data.get("signals", []) if isinstance(sig_data, dict) else (sig_data if isinstance(sig_data, list) else [])
        for s in sig_list:
            if not isinstance(s, dict):
                continue
            box = s.get("box", [0, 0, 0, 0])
            sname = str(s.get("signal_state", s.get("class_name", "signal"))).lower()
            scf = s.get("state_confidence", s.get("detection_conf", 0))
            scol = (0, 0, 255) if "red" in sname else ((0, 255, 0) if "green" in sname else (0, 220, 255))
            draw_badge_box(vis, box, f"SIGNAL: {sname.upper()} {scf:.2f}", scol, (255, 255, 255))

        # 5. OBSTACLES & DEBRIS
        obs_data = capabilities.get("obstacles", {}).get("predictions", {})
        obs_list = obs_data.get("obstacles", []) if isinstance(obs_data, dict) else (obs_data if isinstance(obs_data, list) else [])
        for o in obs_list:
            if not isinstance(o, dict):
                continue
            box = o.get("box", [0, 0, 0, 0])
            oname = str(o.get("category", o.get("class_name", "obstacle"))).upper()
            ocf = o.get("confidence", 0)
            draw_badge_box(vis, box, f"OBSTACLE: {oname} {ocf:.2f}", (180, 100, 240), (255, 255, 255))

        # 6. ACCIDENT / COLLISION HAZARDS
        acc = capabilities.get("accident", {}).get("predictions", {})
        is_accident = (
            acc.get("accident_detected", False)
            or len(acc.get("incidents", [])) > 0
            or acc.get("event_class", "normal_traffic") in [
                "vehicle_collision_detected",
                "vehicle_rollover_or_jackknife",
                "pedestrian_collision_hazard"
            ]
        )
        if is_accident:
            incidents = acc.get("incidents", acc.get("details", []))
            for inc in incidents:
                boxes_to_draw = inc.get("boxes", inc.get("vehicle_pair", []))
                if not boxes_to_draw and "vehicle_box" in inc:
                    boxes_to_draw = [inc["vehicle_box"]]
                for abox in boxes_to_draw:
                    draw_badge_box(vis, abox, "COLLISION / CRASH", (0, 0, 255), (255, 255, 255))

            acc_banner = "(!) CRITICAL ACCIDENT / COLLISION DETECTED"
            (ab_w, ab_h), _ = cv2.getTextSize(acc_banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.15, font_thick + 1)
            cv2.rectangle(vis, (max(10, w // 2 - ab_w // 2 - 15), int(55 * scale)), (min(w - 10, w // 2 + ab_w // 2 + 15), int(55 * scale) + ab_h + 16), (0, 0, 240), -1)
            cv2.putText(vis, acc_banner, (max(20, w // 2 - ab_w // 2), int(55 * scale) + ab_h + 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.15, (255, 255, 255), font_thick + 1, cv2.LINE_AA)
        elif acc.get("has_proximity_warning", False):
            # Subtle amber indicator for close vehicles without false collision alarms
            prox_events = acc.get("proximity_events", [])
            for pe in prox_events[:2]:
                for pbox in pe.get("boxes", []):
                    draw_badge_box(vis, pbox, "PROXIMITY (CLOSE)", (0, 165, 255), (0, 0, 0))

        # 7. TOP HUD BANNER (Risk, Mode Telemetry)
        risk = fusion.get("overall_road_safety_risk", "LOW_RISK")
        risk_color = (0, 255, 100) if "LOW" in risk else ((0, 165, 255) if "MODERATE" in risk else (30, 30, 255))

        hud_h = int(48 * scale)
        cv2.rectangle(vis, (0, 0), (w, hud_h), (18, 18, 22), -1)
        hud_text = f"CYPhER | RISK: {risk} | HAZARDS: {fusion.get('detected_hazards_count', 0)}"

        # Add Surface info if available
        surf = capabilities.get("surface", {}).get("predictions", {})
        if surf:
            mat = surf.get("material", "")
            cond = surf.get("visual_condition", "")
            if mat:
                hud_text += f" | {mat.upper()}"
            if cond:
                hud_text += f" ({cond.upper()})"

        cv2.putText(vis, hud_text, (int(12 * scale), int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, risk_color, font_thick, cv2.LINE_AA)

        return vis

