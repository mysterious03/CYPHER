"""
Temporal Consistency and Video Verification Module
Tracks road-damage defect candidates across sequential frames with camera-motion
compensation, evidence aggregation, and transient flicker rejection.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

class CameraMotionCompensator:
    """
    Estimates global camera motion / vehicle ego-motion between consecutive frames
    using affine transformation on road texture feature points.
    """

    def __init__(self, max_features: int = 150):
        self.max_features = max_features
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

    def estimate_motion(self, current_bgr: np.ndarray) -> np.ndarray:
        """
        Computes 2x3 Affine transformation matrix M translating previous frame coords
        to current frame coords. If estimation fails or on first frame, returns identity.
        """
        curr_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                curr_gray,
                maxCorners=self.max_features,
                qualityLevel=0.01,
                minDistance=15
            )
            return identity

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=self.max_features,
                qualityLevel=0.01,
                minDistance=15
            )
            if self.prev_pts is None or len(self.prev_pts) < 10:
                self.prev_gray = curr_gray
                return identity

        # Lucas-Kanade optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None
        )

        valid_prev = self.prev_pts[status == 1]
        valid_curr = curr_pts[status == 1]

        if len(valid_prev) >= 8:
            M, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
            if M is None:
                M = identity
        else:
            M = identity

        # Update state
        self.prev_gray = curr_gray
        self.prev_pts = cv2.goodFeaturesToTrack(
            curr_gray,
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=15
        )
        return M


class DefectTrack:
    """
    Represents an ongoing multi-frame observation of a physical defect candidate.
    """

    def __init__(self, track_id: int, initial_box: List[float], initial_conf: float, class_name: str, frame_idx: int):
        self.track_id = track_id
        self.box = list(initial_box) # [x1, y1, x2, y2]
        self.confs = [initial_conf]
        self.class_votes = {class_name: 1}
        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1
        self.misses = 0
        self.state = "provisional" # "provisional", "confirmed", "rejected_flicker"

    def update(self, box: List[float], conf: float, class_name: str, frame_idx: int, smoothing: float = 0.65):
        # Exponential moving average for coordinate smoothing
        for i in range(4):
            self.box[i] = smoothing * self.box[i] + (1.0 - smoothing) * box[i]
        self.confs.append(conf)
        self.class_votes[class_name] = self.class_votes.get(class_name, 0) + 1
        self.last_frame = frame_idx
        self.hits += 1
        self.misses = 0

    @property
    def majority_class(self) -> str:
        return max(self.class_votes, key=self.class_votes.get)

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confs))


class TemporalVideoVerifier:
    """
    Tracks defects over sequential frames to filter transient false alarms
    and confirm persisting road defects.
    """

    def __init__(
        self,
        min_hits_to_confirm: int = 3,
        max_misses_to_drop: int = 2,
        iou_track_threshold: float = 0.30
    ):
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_misses_to_drop = max_misses_to_drop
        self.iou_track_threshold = iou_track_threshold
        
        self.motion_compensator = CameraMotionCompensator()
        self.active_tracks: List[DefectTrack] = []
        self.archived_tracks: List[Dict[str, Any]] = []
        self.next_track_id = 1
        self.frame_counter = 0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingests detections for the current frame, compensates previous tracks for ego-motion,
        matches candidates, updates states, and returns validated results.
        """
        self.frame_counter += 1
        M = self.motion_compensator.estimate_motion(frame_bgr)

        # Warp active track positions according to estimated ego-motion
        for trk in self.active_tracks:
            # Transform top-left and bottom-right corners
            pts = np.array([
                [trk.box[0], trk.box[1], 1.0],
                [trk.box[2], trk.box[3], 1.0]
            ]).T
            warped = np.dot(M, pts)
            trk.box = [float(warped[0, 0]), float(warped[1, 0]), float(warped[0, 1]), float(warped[1, 1])]

        matched_tracks = set()
        matched_dets = set()

        # Bipartite matching by IoU
        for d_idx, det in enumerate(detections):
            d_box = det["box"]
            best_iou = 0.0
            best_t_idx = -1

            for t_idx, trk in enumerate(self.active_tracks):
                if t_idx in matched_tracks:
                    continue
                iou = self._iou(d_box, trk.box)
                if iou > best_iou:
                    best_iou = iou
                    best_t_idx = t_idx

            if best_iou >= self.iou_track_threshold and best_t_idx >= 0:
                self.active_tracks[best_t_idx].update(
                    box=d_box,
                    conf=det.get("confidence", 0.5),
                    class_name=det.get("class_name", "defect"),
                    frame_idx=self.frame_counter
                )
                matched_tracks.add(best_t_idx)
                matched_dets.add(d_idx)

        # Create new provisional tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = DefectTrack(
                    track_id=self.next_track_id,
                    initial_box=det["box"],
                    initial_conf=det.get("confidence", 0.5),
                    class_name=det.get("class_name", "defect"),
                    frame_idx=self.frame_counter
                )
                self.next_track_id += 1
                self.active_tracks.append(new_track)

        # Update misses and state transitions
        surviving_tracks = []
        confirmed_output = []
        provisional_output = []

        for t_idx, trk in enumerate(self.active_tracks):
            if t_idx not in matched_tracks and trk.last_frame != self.frame_counter:
                trk.misses += 1

            # State transition
            if trk.hits >= self.min_hits_to_confirm:
                trk.state = "confirmed"
            elif trk.misses >= self.max_misses_to_drop and trk.hits < self.min_hits_to_confirm:
                trk.state = "rejected_flicker"

            if trk.misses <= self.max_misses_to_drop:
                surviving_tracks.append(trk)
                out_item = {
                    "track_id": trk.track_id,
                    "box": [round(c, 1) for c in trk.box],
                    "class_name": trk.majority_class,
                    "confidence": round(trk.mean_confidence, 3),
                    "temporal_hits": trk.hits,
                    "state": trk.state
                }
                if trk.state == "confirmed":
                    confirmed_output.append(out_item)
                else:
                    provisional_output.append(out_item)
            else:
                self.archived_tracks.append({
                    "track_id": trk.track_id,
                    "class_name": trk.majority_class,
                    "hits": trk.hits,
                    "final_state": trk.state
                })

        self.active_tracks = surviving_tracks

        return {
            "frame_index": self.frame_counter,
            "confirmed_defects": confirmed_output,
            "provisional_defects": provisional_output,
            "active_track_count": len(self.active_tracks),
            "rejected_flicker_count": sum(1 for t in self.archived_tracks if t["final_state"] == "rejected_flicker")
        }

    @staticmethod
    def _iou(boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        denom = areaA + areaB - inter
        return inter / denom if denom > 0 else 0.0

