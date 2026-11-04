"""
Trajectory Analyzer for CYPhER Vehicle Tracking.
Calculates vehicle kinematics, road contact points, heading vectors, and overlay rendering.
"""
import math
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional

class TrajectoryAnalyzer:
    """
    Analyzes spatial trajectories of tracked vehicles over temporal sequences.
    """
    def __init__(self, max_history: int = 60, min_track_len: int = 3):
        self.max_history = max_history
        self.min_track_len = min_track_len
        # Maps track_id -> list of (x, y, timestamp)
        self.histories: Dict[int, List[Tuple[float, float, float]]] = {}

    def update(self, track_id: int, bottom_center: Tuple[float, float], timestamp: float):
        """Record bottom-center road contact point."""
        if track_id not in self.histories:
            self.histories[track_id] = []
        self.histories[track_id].append((bottom_center[0], bottom_center[1], timestamp))
        if len(self.histories[track_id]) > self.max_history:
            self.histories[track_id].pop(0)

    def get_kinematics(self, track_id: int) -> Dict[str, Any]:
        """
        Compute velocity, speed, heading, and curvature for a track.
        """
        if track_id not in self.histories or len(self.histories[track_id]) < self.min_track_len:
            return {
                "speed_px": 0.0,
                "velocity": (0.0, 0.0),
                "heading_deg": 0.0,
                "curvature": 0.0,
                "history_length": len(self.histories.get(track_id, []))
            }

        history = self.histories[track_id]
        p_curr = np.array(history[-1][:2])
        p_prev = np.array(history[-2][:2])
        t_curr = history[-1][2]
        t_prev = history[-2][2]
        dt = max(1e-3, t_curr - t_prev)

        # Instantaneous velocity
        vx = (p_curr[0] - p_prev[0]) / dt
        vy = (p_curr[1] - p_prev[1]) / dt
        speed = math.hypot(vx, vy)
        heading = math.degrees(math.atan2(vy, vx)) % 360.0

        # Curvature estimate using last 3 points
        curvature = 0.0
        if len(history) >= 3:
            p_older = np.array(history[-3][:2])
            v_prev = p_prev - p_older
            v_curr = p_curr - p_prev
            angle_prev = math.atan2(v_prev[1], v_prev[0])
            angle_curr = math.atan2(v_curr[1], v_curr[0])
            angle_diff = (angle_curr - angle_prev + math.pi) % (2 * math.pi) - math.pi
            curvature = abs(angle_diff)

        return {
            "speed_px": round(speed, 2),
            "velocity": (round(vx, 2), round(vy, 2)),
            "heading_deg": round(heading, 1),
            "curvature": round(curvature, 3),
            "history_length": len(history)
        }

    def clear(self):
        """Clear all tracking histories."""
        self.histories.clear()

    def draw_overlays(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        draw_boxes: bool = True,
        draw_trajectories: bool = True,
        draw_arrows: bool = True,
        draw_contact_points: bool = True
    ) -> np.ndarray:
        """
        Render tracking visualization onto a video frame.
        """
        out = frame.copy()
        for trk in tracks:
            track_id = trk.get("track_id", -1)
            bbox = trk.get("bbox", [])
            cls_name = trk.get("class_name", "vehicle")
            conf = trk.get("conf", 1.0)
            contact = trk.get("road_contact_point", (0.0, 0.0))
            kinematics = trk.get("kinematics", {})

            # Deterministic color by track ID
            np.random.seed(track_id * 101 % 2**32)
            color = tuple(int(c) for c in np.random.randint(50, 255, size=3))

            # 1. Bounding box
            if draw_boxes and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} {cls_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out, (x1, max(0, y1 - 20)), (x1 + w, max(20, y1)), color, -1)
                cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 2. Road contact point (bottom-center of bounding box)
            if draw_contact_points:
                cx, cy = int(contact[0]), int(contact[1])
                cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(out, (cx, cy), 7, (255, 255, 255), 1)

            # 3. Trajectory history polyline
            if draw_trajectories and track_id in self.histories:
                pts = [(int(p[0]), int(p[1])) for p in self.histories[track_id]]
                if len(pts) > 1:
                    for i in range(1, len(pts)):
                        thickness = max(1, int(i / len(pts) * 3))
                        cv2.line(out, pts[i - 1], pts[i], color, thickness)

            # 4. Motion arrow
            if draw_arrows and kinematics.get("history_length", 0) >= self.min_track_len:
                vx, vy = kinematics.get("velocity", (0.0, 0.0))
                cx, cy = int(contact[0]), int(contact[1])
                # Scale velocity vector for visualization
                norm = math.hypot(vx, vy)
                if norm > 1.0:
                    scale = min(40.0, norm * 0.5)
                    end_x = int(cx + (vx / norm) * scale)
                    end_y = int(cy + (vy / norm) * scale)
                    cv2.arrowedLine(out, (cx, cy), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

        return out

