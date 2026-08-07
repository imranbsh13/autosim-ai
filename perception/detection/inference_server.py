import socket
import struct
import time
import cv2
import numpy as np
import json
import sys
import os

# Add perception/ and perception/detection/ to path
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(
    os.path.abspath(__file__)))

from ultralytics import YOLO
from lane_detector import LaneDetector
from fusion.kalman_fusion import KalmanFusion
from autonomous.autonomous_controller import AutonomousController
from stereo_depth import StereoDepth
from bev_projection import BEVProjection

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH     = r"C:\Projects\autosim-ai\models\autosim_v1\weights\best.pt"
RECEIVE_PORT   = 5006
SEND_PORT      = 5008
STEREO_PORT    = 5009
UNITY_IP       = "127.0.0.1"
BUFFER_SIZE    = 65535
CONF_THRESHOLD = 0.47
PREVIEW_PATH   = r"C:\Projects\autosim-ai\perception\preview.jpg"
# ─────────────────────────────────────────────


class PerceptionServer:
    """
    Real-time perception server — Phase 4.5 with stereo + BEV.

    Pipeline per frame:
    1.  Receive RGB frame + telemetry (port 5006)
    2.  Receive stereo frame (port 5009)
    3.  YOLOv8 object detection
    4.  OpenCV lane detection
    5.  Stereo depth estimation
    6.  Kalman filter sensor fusion
    7.  Autonomous controller
    8.  Send payload to Unity (port 5008)
    9.  Save annotated preview every 30 frames
    """

    def __init__(self):
        print("[PerceptionServer] Initialising...")

        # ── MODEL ─────────────────────────────────────────
        print(f"[PerceptionServer] Loading model...")
        self.model = YOLO(MODEL_PATH)
        self.model.to('cuda')
        print("[PerceptionServer] Model loaded on GPU")

        self.class_names = {
            0: "vehicle",      1: "pedestrian",
            2: "traffic_sign", 3: "traffic_light"
        }
        self.class_colors = {
            0: (0, 255, 0),    # vehicle — green
            1: (0, 0, 255),    # pedestrian — red
            2: (0, 255, 255),  # traffic_sign — yellow
            3: (255, 0, 0),    # traffic_light — blue
        }

        # ── PERCEPTION MODULES ────────────────────────────
        self.lane_detector = LaneDetector()
        print("[PerceptionServer] Lane detector initialised")

        self.fusion = KalmanFusion(
            max_age=5, min_hits=1, iou_threshold=0.25)
        print("[PerceptionServer] Kalman fusion initialised")

        self.stereo = StereoDepth(
            baseline=0.06, fov_degrees=60,
            image_width=640)
        print("[PerceptionServer] Stereo depth initialised")

        self.bev = BEVProjection(
            image_width=640, image_height=360,
            bev_width=280, bev_height=360)
        print("[PerceptionServer] BEV projection initialised")

        # ── NETWORK ───────────────────────────────────────
        # Main receive
        self.recv_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.bind(("0.0.0.0", RECEIVE_PORT))
        self.recv_socket.settimeout(5.0)
        print(f"[PerceptionServer] Listening on "
              f"port {RECEIVE_PORT}")

        # Send
        self.send_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[PerceptionServer] Sending to "
              f"{UNITY_IP}:{SEND_PORT}")

        # Stereo receive
        self.stereo_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        self.stereo_socket.bind(("0.0.0.0", STEREO_PORT))
        self.stereo_socket.settimeout(0.01)
        print(f"[PerceptionServer] Stereo on "
              f"port {STEREO_PORT}")

        # ── AUTONOMOUS CONTROLLER ─────────────────────────
        self.controller = AutonomousController()
        self.controller.set_mode(
            AutonomousController.MODE_AUTONOMOUS)
        print("[PerceptionServer] Controller initialised")

        # ── STATE ─────────────────────────────────────────
        self.running              = False
        self.frame_count          = 0
        self.total_inference_time = 0.0
        self.stereo_frame         = None

        print("[PerceptionServer] Ready")

    def decode_frame(self, data):
        """
        Decode UDP packet → (frame, telemetry).
        Packet: [4B length][telemetry JSON][JPEG bytes]
        """
        try:
            telem_len  = struct.unpack('>I', data[:4])[0]
            if telem_len <= 0 or telem_len > 10000:
                return None, {}

            telem_json = data[4:4 + telem_len].decode('utf-8')
            telem_json = telem_json.replace('Infinity', '999')
            telem_json = telem_json.replace('NaN', '0')
            telemetry  = json.loads(telem_json)

            jpeg_data = data[4 + telem_len:]
            np_arr    = np.frombuffer(
                jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            return frame, telemetry

        except Exception as e:
            print(f"[ERROR] decode_frame: {e}")
            return None, {}

    def run_detection(self, frame):
        """Run YOLOv8 inference. Returns detections + ms."""
        t_start = time.perf_counter()

        results = self.model(
            frame, conf=CONF_THRESHOLD,
            verbose=False, device='cuda')

        inference_ms = (time.perf_counter() - t_start) * 1000
        detections   = []
        result       = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id   = int(box.cls[0])

                cx = ((x1 + x2) / 2) / frame.shape[1]
                cy = ((y1 + y2) / 2) / frame.shape[0]
                w  = (x2 - x1) / frame.shape[1]
                h  = (y2 - y1) / frame.shape[0]

                detections.append({
                    "class_id":   class_id,
                    "class_name": self.class_names.get(
                        class_id, "unknown"),
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                        "cx": round(cx, 4),
                        "cy": round(cy, 4),
                        "w":  round(w, 4),
                        "h":  round(h, 4),
                    }
                })

        return detections, inference_ms

    def draw_detections(self, frame, detections):
        """Draw YOLOv8 bounding boxes."""
        for det in detections:
            bbox     = det["bbox"]
            class_id = det["class_id"]
            color    = self.class_colors.get(
                class_id, (255, 255, 255))

            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])

            cv2.rectangle(frame, (x1, y1),
                          (x2, y2), color, 2)
            label = (f"{det['class_name']} "
                     f"{det['confidence']:.2f}")
            sz = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, 1)[0]
            cv2.rectangle(frame,
                (x1, y1 - sz[1] - 8),
                (x1 + sz[0], y1), color, -1)
            cv2.putText(frame, label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)
        return frame

    def draw_overlay(self, frame, detections, telemetry,
                     inference_ms, current_fps,
                     lane_offset, lane_detected, control):
        """Draw telemetry and stats overlay."""
        speed    = telemetry.get("speed", 0)
        steering = telemetry.get("steering", 0)
        gear     = telemetry.get("gear", 0)

        cv2.putText(frame,
            f"FPS:{current_fps:.1f} | "
            f"Inf:{inference_ms:.1f}ms",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
        cv2.putText(frame,
            f"Speed:{speed:.1f}m/s | "
            f"Steer:{steering:.2f} | Gear:{gear}",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
        cv2.putText(frame,
            f"Det:{len(detections)} | "
            f"Frame:{self.frame_count}",
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
        cv2.putText(frame,
            f"Lane:{lane_offset:.3f} | "
            f"{'OK' if lane_detected else 'NONE'}",
            (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0) if lane_detected
            else (0, 0, 255), 2)
        cv2.putText(frame,
            f"AI: Steer:{control['steering']:+.3f} | "
            f"Thr:{control['throttle']:.2f} | "
            f"Brk:{control['brake']:.2f}",
            (20, 180), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2)
        return frame

    def send_payload(self, detections, telemetry,
                     inference_ms, lane_offset,
                     lane_detected, scene_state, control):
        """Send single JSON payload to Unity."""
        payload = {
            "frame_id":        self.frame_count,
            "inference_ms":    round(inference_ms, 2),
            "detection_count": len(detections),
            "detections":      detections,
            "lane_offset":     round(lane_offset, 4),
            "lane_detected":   lane_detected,
            "scene_state":     scene_state or {},
            "control": {
                "steering":      control["steering"],
                "throttle":      control["throttle"],
                "brake":         control["brake"],
                "mode":          control["mode"],
                "emergency_brake":
                    control["emergency_brake"],
            },
            "telemetry_echo":  telemetry,
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            self.send_socket.sendto(
                data, (UNITY_IP, SEND_PORT))
        except Exception as e:
            print(f"[ERROR] send_payload: {e}")

    def run(self):
        """Main server loop."""
        self.running = True
        print("\n[PerceptionServer] Server running.")
        print(f"[PerceptionServer] Preview: {PREVIEW_PATH}")
        print("[PerceptionServer] Press Ctrl+C to stop.\n")

        fps_timer       = time.time()
        fps_frame_count = 0
        current_fps     = 0.0
        session_start   = time.time()

        while self.running:
            try:
                # ── RECEIVE STEREO FRAME ────────────────────
                try:
                    stereo_data, _ = \
                        self.stereo_socket.recvfrom(
                            BUFFER_SIZE)
                    stereo_arr = np.frombuffer(
                        stereo_data, dtype=np.uint8)
                    self.stereo_frame = cv2.imdecode(
                        stereo_arr, cv2.IMREAD_COLOR)
                except (socket.timeout,
                        TimeoutError, OSError):
                    pass

                # ── RECEIVE MAIN FRAME ──────────────────────
                data, addr = self.recv_socket.recvfrom(
                    BUFFER_SIZE)

                # ── DECODE ─────────────────────────────────
                frame, telemetry = self.decode_frame(data)
                if frame is None:
                    continue

                # ── YOLO DETECTION ─────────────────────────
                detections, inference_ms = \
                    self.run_detection(frame)

                # ── LANE DETECTION ─────────────────────────
                lane_frame, lane_offset, lane_detected = \
                    self.lane_detector.detect(frame.copy())

                # ── STEREO DEPTH ───────────────────────────
                depth_map = None
                depth_vis = None
                if self.stereo_frame is not None:
                    depth_map, depth_vis, _ = \
                        self.stereo.compute(
                            frame, self.stereo_frame)

                    if (self.frame_count % 30 == 0
                            and detections):
                        det_depths = \
                            self.stereo.get_detection_depths(
                                depth_map, detections)
                        for det, d in det_depths[:3]:
                            if d:
                                print(
                                    f"  Stereo: "
                                    f"{det['class_name']}"
                                    f" @ {d:.1f}m")

                # ── SENSOR FUSION ──────────────────────────
                confirmed_tracks, scene_state = \
                    self.fusion.update(detections, telemetry)

                # ── AUTONOMOUS CONTROL ─────────────────────
                control = self.controller.compute(
                    scene_state, lane_offset, lane_detected)

                # ── BEV MINIMAP ────────────────────────────
                bev_map = self.bev.create_minimap(
                    detections, confirmed_tracks, frame)

                # ── SEND TO UNITY ──────────────────────────
                self.send_payload(
                    detections, telemetry, inference_ms,
                    lane_offset, lane_detected,
                    scene_state, control)

                # ── COUNTERS ───────────────────────────────
                self.frame_count += 1
                self.total_inference_time += inference_ms
                fps_frame_count += 1

                # ── FPS ────────────────────────────────────
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    current_fps     = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_timer       = time.time()

                # ── PREVIEW EVERY 30 FRAMES ────────────────
                if self.frame_count % 30 == 0:
                    vis = lane_frame.copy()
                    vis = self.draw_detections(
                        vis, detections)
                    vis = self.draw_overlay(
                        vis, detections, telemetry,
                        inference_ms, current_fps,
                        lane_offset, lane_detected,
                        control)

                    # BEV minimap — bottom right
                    h, w = vis.shape[:2]
                    bev_r = cv2.resize(bev_map, (200, 260))
                    bh, bw = bev_r.shape[:2]
                    vis[h-bh-10:h-10,
                        w-bw-10:w-10] = bev_r

                    # Depth map — bottom left
                    if depth_vis is not None:
                        dv = cv2.resize(
                            depth_vis, (200, 113))
                        dh, dw = dv.shape[:2]
                        vis[h-dh-10:h-10,
                            10:10+dw] = dv

                    cv2.imwrite(PREVIEW_PATH,
                        cv2.resize(vis, (960, 540)))

                # ── CONSOLE LOG EVERY 30 FRAMES ────────────
                if self.frame_count % 30 == 0:
                    avg_inf = (self.total_inference_time
                               / self.frame_count)
                    speed   = telemetry.get("speed", 0)
                    closest = scene_state.get(
                        "closest_vehicle_distance", 999)
                    stereo_str = (
                        "OK" if self.stereo_frame
                        is not None else "NO")
                    print(
                        f"[Frame {self.frame_count:05d}] "
                        f"FPS:{current_fps:.1f} | "
                        f"Inf:{inference_ms:.1f}ms"
                        f"(avg:{avg_inf:.1f}) | "
                        f"Det:{len(detections)} | "
                        f"Tracks:{scene_state.get('total_tracked', 0)} | "
                        f"Closest:{closest:.1f}m | "
                        f"Steer:{control['steering']:+.3f} | "
                        f"Thr:{control['throttle']:.2f} | "
                        f"Brk:{control['brake']:.2f} | "
                        f"Stereo:{stereo_str} | "
                        f"Speed:{speed:.1f}m/s")

            except (socket.timeout,
                    TimeoutError, OSError):
                print("[PerceptionServer] "
                      "Waiting for Unity connection...")
                continue

            except KeyboardInterrupt:
                print(
                    "\n[PerceptionServer] Shutting down...")
                self.running = False

        # ── CLEANUP ────────────────────────────────────────
        self.recv_socket.close()
        self.send_socket.close()
        self.stereo_socket.close()

        total_time = time.time() - session_start
        avg_inf    = (self.total_inference_time
                      / self.frame_count
                      if self.frame_count > 0 else 0)

        print(f"\n[PerceptionServer] Session complete:")
        print(f"  Total frames:  {self.frame_count}")
        print(f"  Total time:    {total_time:.1f}s")
        print(f"  Avg FPS:       "
              f"{self.frame_count / total_time:.1f}")
        print(f"  Avg inference: {avg_inf:.1f}ms")


if __name__ == "__main__":
    server = PerceptionServer()
    server.run()