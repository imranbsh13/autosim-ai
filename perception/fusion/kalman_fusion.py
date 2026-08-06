import numpy as np
from collections import OrderedDict


class TrackedObject:
    """
    Single tracked object with pure numpy Kalman filter.
    State: [cx, cy, w, h, vx, vy]
    """
    _next_id = 0

    def __init__(self, class_id, class_name, bbox,
                 frame_width, frame_height):
        self.track_id   = TrackedObject._next_id
        TrackedObject._next_id += 1

        self.class_id   = class_id
        self.class_name = class_name
        self.age        = 0
        self.frames_since_detection = 0
        self.hit_count  = 1
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.estimated_distance = 0.0
        self.velocity   = 0.0
        self.risk_score = 0.0

        # ── KALMAN MATRICES (pure numpy) ──────────────────

        # State transition: constant velocity model
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix: observe cx,cy,w,h only
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=float)

        # Measurement noise covariance
        self.R = np.diag([0.01, 0.01, 0.005, 0.005])

        # Process noise covariance
        self.Q = np.diag([
            0.001, 0.001, 0.0005, 0.0005, 0.01, 0.01])

        # State covariance
        self.P = np.diag([0.1, 0.1, 0.05, 0.05, 1.0, 1.0])

        # Initial state
        self.x = np.array([
            bbox["cx"], bbox["cy"],
            bbox["w"],  bbox["h"],
            0.0, 0.0
        ], dtype=float).reshape(6, 1)

        self.estimated_distance = \
            self._estimate_distance(bbox["h"])

    def predict(self):
        """Predict state forward one frame."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.frames_since_detection += 1

    def update(self, bbox, confidence):
        """Update with new matched detection."""
        z = np.array([
            bbox["cx"], bbox["cy"],
            bbox["w"],  bbox["h"],
        ], dtype=float).reshape(4, 1)

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        self.frames_since_detection = 0
        self.hit_count += 1

        h = float(self.x[3])
        self.estimated_distance = self._estimate_distance(h)

        vx = float(self.x[4])
        vy = float(self.x[5])
        self.velocity = np.sqrt(vx**2 + vy**2)

        distance_factor = max(
            0.0, 1.0 - self.estimated_distance / 50.0)
        self.risk_score = distance_factor * (
            1.0 + self.velocity * 10.0)

    def get_state(self):
        """Return current smoothed state as dict."""
        s = self.x.flatten()
        return {
            "track_id":   self.track_id,
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "cx": float(s[0]),
            "cy": float(s[1]),
            "w":  float(s[2]),
            "h":  float(s[3]),
            "vx": float(s[4]),
            "vy": float(s[5]),
            "velocity":   round(self.velocity, 4),
            "distance_m": round(self.estimated_distance, 2),
            "risk_score": round(self.risk_score, 3),
            "age":        self.age,
            "hit_count":  self.hit_count,
        }

    def _estimate_distance(self, bbox_height_norm):
        if bbox_height_norm < 0.001:
            return 999.0
        if self.class_id == 0:
            # Vehicle: bbox_height 0.15 ≈ 10m, 0.05 ≈ 30m
            reference = 1.5
        else:
            # Pedestrian
            reference = 0.8
        distance = reference / bbox_height_norm
        return max(2.0, min(100.0, distance))


class KalmanFusion:
    """
    Multi-object tracker using pure numpy Kalman filters.
    Associates detections to tracks using IoU matching.
    """

    def __init__(self, max_age=5, min_hits=2,
                 iou_threshold=0.25):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.tracks        = OrderedDict()
        self.frame_count   = 0
        self.vehicle_speed    = 0.0
        self.vehicle_steering = 0.0

    def update(self, detections, telemetry):
        """
        Main update. Called every frame.
        Returns confirmed tracks and scene state.
        """
        self.frame_count += 1
        self.vehicle_speed    = telemetry.get("speed", 0.0)
        self.vehicle_steering = telemetry.get("steering", 0.0)

        # Predict all existing tracks forward
        for track in self.tracks.values():
            track.predict()

        # Match detections to tracks
        matched, unmatched_dets, _ = \
            self._associate(detections)

        # Update matched tracks
        for det_idx, trk_id in matched:
            self.tracks[trk_id].update(
                detections[det_idx]["bbox"],
                detections[det_idx]["confidence"])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            t = TrackedObject(
                class_id=det["class_id"],
                class_name=det["class_name"],
                bbox=det["bbox"],
                frame_width=640,
                frame_height=360,
            )
            self.tracks[t.track_id] = t

        # Remove stale tracks
        dead = [tid for tid, t in self.tracks.items()
                if t.frames_since_detection > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        # Collect confirmed tracks
        confirmed = [
            t.get_state()
            for t in self.tracks.values()
            if t.hit_count >= self.min_hits
        ]

        scene_state = self._compute_scene_state(confirmed)
        return confirmed, scene_state

    def _associate(self, detections):
        """Greedy IoU matching of detections to tracks."""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), \
                   list(self.tracks.keys())

        track_ids = list(self.tracks.keys())
        tracks    = [self.tracks[tid] for tid in track_ids]

        # Build IoU matrix
        iou_mat = np.zeros(
            (len(detections), len(tracks)), dtype=float)
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                s = trk.x.flatten()
                trk_bbox = {
                    "cx": s[0], "cy": s[1],
                    "w":  s[2], "h":  s[3]}
                iou_mat[d, t] = self._iou(
                    det["bbox"], trk_bbox)

        # Sort pairs by IoU descending, greedily assign
        pairs = [(iou_mat[d, t], d, t)
                 for d in range(len(detections))
                 for t in range(len(tracks))]
        pairs.sort(key=lambda x: x[0], reverse=True)

        matched    = []
        used_dets  = set()
        used_trks  = set()

        for iou_val, d, t in pairs:
            if iou_val < self.iou_threshold:
                break
            if d in used_dets or t in used_trks:
                continue
            matched.append((d, track_ids[t]))
            used_dets.add(d)
            used_trks.add(t)

        unmatched_dets = [d for d in range(len(detections))
                          if d not in used_dets]
        unmatched_trks = [track_ids[t]
                          for t in range(len(tracks))
                          if t not in used_trks]

        return matched, unmatched_dets, unmatched_trks

    def _iou(self, a, b):
        """IoU between two cx,cy,w,h boxes."""
        ax1, ay1 = a["cx"] - a["w"]/2, a["cy"] - a["h"]/2
        ax2, ay2 = a["cx"] + a["w"]/2, a["cy"] + a["h"]/2
        bx1, by1 = b["cx"] - b["w"]/2, b["cy"] - b["h"]/2
        bx2, by2 = b["cx"] + b["w"]/2, b["cy"] + b["h"]/2

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a["w"]*a["h"] + b["w"]*b["h"] - inter
        return inter / union if union > 0 else 0.0

    def _compute_scene_state(self, tracks):
        """Compute high-level scene summary."""
        vehicles    = [t for t in tracks if t["class_id"] == 0]
        pedestrians = [t for t in tracks if t["class_id"] == 1]

        ahead = [v for v in vehicles if v["cy"] > 0.4]
        closest_dist = 999.0
        if ahead:
            closest_dist = min(v["distance_m"] for v in ahead)

        highest_risk = (max(t["risk_score"] for t in tracks)
                        if tracks else 0.0)

        emergency_brake = any(
            t["distance_m"] < 8.0 and t["risk_score"] > 0.7
            for t in tracks)

        return {
            "vehicle_count":    len(vehicles),
            "pedestrian_count": len(pedestrians),
            "total_tracked":    len(tracks),
            "closest_vehicle_distance": round(closest_dist, 2),
            "highest_risk_score": round(highest_risk, 3),
            "emergency_brake":  emergency_brake,
            "collision_warning": closest_dist < 20.0,
            "ego_speed":        self.vehicle_speed,
            "ego_steering":     self.vehicle_steering,
            "tracks":           tracks,
        }