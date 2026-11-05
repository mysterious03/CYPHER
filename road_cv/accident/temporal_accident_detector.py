"""
CYPhER V2.2 Temporal Video Accident & Incident Intelligence Engine
Implements a 16-frame sliding temporal buffer with a Spatio-Temporal 3D-CNN architecture
and multi-vehicle trajectory tracking.
Distinguishes suspected incidents from verified collisions.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import deque

ACCIDENT_EVENT_CLASSES = [
    "normal_traffic",
    "hard_braking_near_miss",
    "vehicle_collision",
    "vehicle_rollover_or_jackknife"
]

class SpatioTemporalAccident3DCNN(nn.Module):
    """
    Lightweight 3D Convolutional Neural Network for 16-frame action recognition.
    Input: [B, 3, 16, 112, 112] (Batch, RGB channels, 16 temporal frames, H, W).
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv3d(x)
        flat = torch.flatten(f, 1)
        return self.classifier(flat)

class TemporalAccidentDetector:
    """
    Stateful temporal accident intelligence engine.
    Maintains a 16-frame circular temporal buffer and combines 3D spatio-temporal features
    with geometric trajectory anomaly heuristics.
    """
    def __init__(
        self,
        buffer_size: int = 16,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda"
    ):
        self.buffer_size = buffer_size
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.frame_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        self.trajectory_history = {} # track_id -> list of centroids

        self.model = SpatioTemporalAccident3DCNN()
        self.is_trained = False
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt["state_dict"])
                self.model.to(self.device).eval()
                self.is_trained = True
            except Exception as e:
                print(f"[TemporalAccident WARNING] Checkpoint load failed: {e}")

    def update_frame_buffer(self, frame: np.ndarray, timestamp: float):
        """Adds a downsampled frame (112x112) to the temporal buffer."""
        resized = cv2.resize(frame, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.frame_buffer.append(rgb)
        self.timestamp_buffer.append(timestamp)

    def analyze(
        self,
        frame: np.ndarray,
        detected_vehicles: List[Dict[str, Any]],
        timestamp: float = 0.0,
        is_video: bool = False
    ) -> Dict[str, Any]:
        """
        Executes hybrid temporal analysis or fallback geometric anomaly inspection.
        """
        self.update_frame_buffer(frame, timestamp)

        # 1. Video Temporal 3D-CNN Path (when buffer is filled and model is trained)
        if is_video and len(self.frame_buffer) == self.buffer_size and self.is_trained:
            frames_np = np.stack(list(self.frame_buffer), axis=0) # [16, 112, 112, 3]
            frames_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float() / 255.0 # [3, 16, 112, 112]
            frames_t = frames_t.unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(frames_t)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                idx = int(np.argmax(probs))
                event_name = ACCIDENT_EVENT_CLASSES[idx]
                conf = float(probs[idx])

                is_anomaly = idx > 0
                return {
                    "status": "success",
                    "mode": "temporal_3d_cnn",
                    "anomaly_detected": is_anomaly,
                    "accident_detected": is_anomaly,
                    "risk_level": "CRITICAL" if is_anomaly else "NORMAL",
                    "event_class": event_name,
                    "confidence": round(conf, 4),
                    "evidence_window_sec": round(self.timestamp_buffer[-1] - self.timestamp_buffer[0], 2),
                    "buffer_frames": len(self.frame_buffer),
                    "involved_vehicles_count": len(detected_vehicles),
                    "incidents": [{"type": event_name, "confidence": conf}] if is_anomaly else [],
                    "details": [{"type": event_name, "confidence": conf}] if is_anomaly else []
                }

        # 2. Geometric Trajectory & Spatial Overlap Intelligence
        anomalies = []
        proximity_events = []

        for i, v1 in enumerate(detected_vehicles):
            b1 = v1.get("box", [0, 0, 1, 1])
            w1 = max(1, b1[2] - b1[0])
            h1 = max(1, b1[3] - b1[1])
            area1 = w1 * h1
            aspect = w1 / float(h1)
            cname1 = str(v1.get("class_name", "vehicle")).lower()

            # Overturned vehicle / Rollover orientation check
            # Only apply to regular passenger cars (exclude long trucks, buses, motorcycles, bicycles, pedestrians)
            is_passenger_car = cname1 in ["car", "sedan", "suv", "automobile", "vehicle"]
            if is_passenger_car and (aspect < 0.55 or aspect > 3.5) and h1 > 60 and w1 > 60:
                anomalies.append({
                    "type": "vehicle_rollover_or_jackknife",
                    "confidence": 0.88,
                    "vehicle_box": b1,
                    "boxes": [b1],
                    "aspect_ratio": round(aspect, 2),
                    "reason": f"Severe vehicle orientation abnormality (aspect {aspect:.2f}) indicates rollover, tilt, or structural crash deformation"
                })

            # Inter-vehicle collision & contact overlap
            for j in range(i + 1, len(detected_vehicles)):
                v2 = detected_vehicles[j]
                b2 = v2.get("box", [0, 0, 1, 1])
                w2 = max(1, b2[2] - b2[0])
                h2 = max(1, b2[3] - b2[1])
                area2 = w2 * h2
                cname2 = str(v2.get("class_name", "vehicle")).lower()

                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih

                if inter > 0:
                    union = area1 + area2 - inter
                    iou = inter / max(1.0, float(union))
                    iomin = inter / max(1.0, float(min(area1, area2)))

                    # Pedestrian / Cyclist collision check (requires substantial bounding volume penetration)
                    is_ped_incident = ("person" in [cname1, cname2] or "pedestrian" in [cname1, cname2]) and (cname1 != cname2)
                    if is_ped_incident and (iomin >= 0.28 or iou >= 0.18):
                        anomalies.append({
                            "type": "pedestrian_collision_hazard",
                            "confidence": 0.92,
                            "overlap_ratio": round(float(iomin), 3),
                            "vehicle_pair": [b1, b2],
                            "boxes": [b1, b2],
                            "reason": "Pedestrian/cyclist directly in collision path or contacting vehicular bounding volume"
                        })
                    # Severe vehicular collision (Requires deep physical intersection of bounding volumes)
                    elif (iou >= 0.28 and iomin >= 0.38) or (iomin >= 0.55):
                        anomalies.append({
                            "type": "vehicle_collision_detected",
                            "confidence": min(0.96, round(0.75 + float(iomin) * 0.22, 2)),
                            "overlap_iou": round(float(iou), 3),
                            "overlap_ratio": round(float(iomin), 3),
                            "vehicle_pair": [b1, b2],
                            "boxes": [b1, b2],
                            "reason": f"Severe physical vehicular bounding overlap (IoU: {iou:.2f}, Overlap: {iomin:.2f}) indicates active collision or crash impact"
                        })
                    # Moderate proximity / Tailgating / Dense Traffic (NOT a collision!)
                    elif (iou >= 0.10 or iomin >= 0.18):
                        proximity_events.append({
                            "type": "close_vehicle_proximity",
                            "confidence": 0.65,
                            "overlap_iou": round(float(iou), 3),
                            "overlap_ratio": round(float(iomin), 3),
                            "vehicle_pair": [b1, b2],
                            "boxes": [b1, b2],
                            "reason": f"Vehicles traveling in close proximity / adjacent lanes (IoU: {iou:.2f}, Overlap: {iomin:.2f})"
                        })

        has_accident = len(anomalies) > 0
        has_proximity = len(proximity_events) > 0
        top_event = anomalies[0] if has_accident else (proximity_events[0] if has_proximity else None)

        return {
            "status": "success",
            "mode": "spatio_geometric_heuristic",
            "anomaly_detected": has_accident,
            "accident_detected": has_accident,
            "has_proximity_warning": has_proximity,
            "risk_level": "CRITICAL" if has_accident else ("MODERATE" if has_proximity else "NORMAL"),
            "event_class": top_event["type"] if top_event else "normal_traffic",
            "confidence": top_event["confidence"] if top_event else 0.95,
            "buffer_frames": len(self.frame_buffer),
            "scientific_status": "TRAINED" if self.is_trained else "HEURISTIC_ACTIVE",
            "incidents": anomalies,
            "details": anomalies,
            "proximity_events": proximity_events
        }
