import socket
import struct
import time
import cv2
import numpy as np
import json
from ultralytics import YOLO
from lane_detector import LaneDetector

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH      = r"C:\Projects\autosim-ai\models\autosim_v1\weights\best.pt"
RECEIVE_PORT    = 5006
SEND_PORT       = 5007
UNITY_IP        = "127.0.0.1"
BUFFER_SIZE     = 65535
CONF_THRESHOLD  = 0.47
PREVIEW_PATH    = r"C:\Projects\autosim-ai\perception\preview.jpg"
# ─────────────────────────────────────────────


class PerceptionServer:
    """
    Real-time perception server.
    Receives JPEG frames + telemetry from Unity over UDP port 5006.
    Runs YOLOv8 detection + lane detection.
    Sends detections + lane data back to Unity over UDP port 5007.
    Saves annotated preview image to disk every 30 frames.
    """

    def __init__(self):
        print("[PerceptionServer] Initialising...")

        # ── YOLO MODEL ────────────────────────────────────
        print(f"[PerceptionServer] Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        self.model.to('cuda')
        print("[PerceptionServer] Model loaded on GPU")

        # ── LANE DETECTOR ─────────────────────────────────
        self.lane_detector = LaneDetector()
        print("[PerceptionServer] Lane detector initialised")

        # ── CLASS DEFINITIONS ─────────────────────────────
        self.class_names = {
            0: "vehicle",
            1: "pedestrian",
            2: "traffic_sign",
            3: "traffic_light"
        }
        self.class_colors = {
            0: (0, 255, 0),    # vehicle — green
            1: (0, 0, 255),    # pedestrian — red
            2: (0, 255, 255),  # traffic_sign — yellow
            3: (255, 0, 0),    # traffic_light — blue
        }

        # ── NETWORK ───────────────────────────────────────
        self.recv_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.bind(("0.0.0.0", RECEIVE_PORT))
        self.recv_socket.settimeout(5.0)
        print(f"[PerceptionServer] Listening on port {RECEIVE_PORT}")

        self.send_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[PerceptionServer] Sending to {UNITY_IP}:{SEND_PORT}")

        # ── STATE ─────────────────────────────────────────
        self.running = False
        self.frame_count = 0
        self.total_inference_time = 0.0

        print("[PerceptionServer] Ready")

    def decode_frame(self, data):
        """
        Decode UDP packet into frame and telemetry.

        Packet format:
        [4 bytes: telemetry JSON length (big-endian uint32)]
        [N bytes: telemetry JSON]
        [remaining: JPEG bytes]
        """
        try:
            telem_len = struct.unpack('>I', data[:4])[0]
            telem_json = data[4:4 + telem_len].decode('utf-8')
            telemetry = json.loads(telem_json)
            jpeg_data = data[4 + telem_len:]
            np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame, telemetry
        except Exception as e:
            print(f"[ERROR] decode_frame: {e}")
            return None, {}

    def run_detection(self, frame):
        """
        Run YOLOv8 inference on a single frame.
        Returns list of detection dicts and inference time in ms.
        """
        t_start = time.perf_counter()

        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            verbose=False,
            device='cuda'
        )

        inference_ms = (time.perf_counter() - t_start) * 1000
        detections = []
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

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
        """Draw YOLOv8 bounding boxes on frame."""
        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            color = self.class_colors.get(class_id, (255, 255, 255))

            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0], y1),
                color, -1)
            cv2.putText(frame, label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    def draw_overlay(self, frame, detections, telemetry,
                     inference_ms, current_fps,
                     lane_offset, lane_detected):
        """Draw telemetry, FPS, and lane stats overlay."""
        speed    = telemetry.get("speed", 0)
        steering = telemetry.get("steering", 0)
        gear     = telemetry.get("gear", 0)

        cv2.putText(frame,
            f"FPS: {current_fps:.1f} | Inference: {inference_ms:.1f}ms",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame,
            f"Speed: {speed:.1f} m/s | Steer: {steering:.2f} | Gear: {gear}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame,
            f"Detections: {len(detections)} | Frame: {self.frame_count}",
            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame,
            f"Lane offset: {lane_offset:.3f} | "
            f"{'DETECTED' if lane_detected else 'NOT DETECTED'}",
            (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0) if lane_detected else (0, 0, 255), 2)

        return frame

    def send_detections(self, detections, telemetry, inference_ms,
                        lane_offset=0.0, lane_detected=False):
        """Send detection + lane results to Unity as JSON over UDP."""
        payload = {
            "frame_id":        self.frame_count,
            "inference_ms":    round(inference_ms, 2),
            "detection_count": len(detections),
            "detections":      detections,
            "lane_offset":     round(lane_offset, 4),
            "lane_detected":   lane_detected,
            "telemetry_echo":  telemetry,
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            self.send_socket.sendto(data, (UNITY_IP, SEND_PORT))
        except Exception as e:
            print(f"[ERROR] send_detections: {e}")

    def run(self):
        """
        Main server loop.
        Receive → Decode → Detect → Lane → Send → Preview
        """
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
                # ── RECEIVE ────────────────────────────────
                data, addr = self.recv_socket.recvfrom(BUFFER_SIZE)

                # ── DECODE ─────────────────────────────────
                frame, telemetry = self.decode_frame(data)
                if frame is None:
                    continue

                # ── YOLO DETECTION ─────────────────────────
                detections, inference_ms = self.run_detection(frame)

                # ── LANE DETECTION ─────────────────────────
                lane_frame, lane_offset, lane_detected = \
                    self.lane_detector.detect(frame.copy())

                # ── SEND TO UNITY ──────────────────────────
                self.send_detections(
                    detections, telemetry, inference_ms,
                    lane_offset, lane_detected)

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

                # ── SAVE PREVIEW EVERY 30 FRAMES ───────────
                if self.frame_count % 30 == 0:
                    vis = lane_frame.copy()
                    vis = self.draw_detections(vis, detections)
                    vis = self.draw_overlay(
                        vis, detections, telemetry,
                        inference_ms, current_fps,
                        lane_offset, lane_detected)
                    cv2.imwrite(PREVIEW_PATH,
                                cv2.resize(vis, (960, 540)))

                # ── CONSOLE LOG EVERY 30 FRAMES ────────────
                if self.frame_count % 30 == 0:
                    avg_inf = (self.total_inference_time
                               / self.frame_count)
                    speed = telemetry.get("speed", 0)
                    lane_str = (f"offset={lane_offset:.3f}"
                                if lane_detected else "no lane")
                    print(f"[Frame {self.frame_count:05d}] "
                          f"FPS: {current_fps:.1f} | "
                          f"Inf: {inference_ms:.1f}ms "
                          f"(avg: {avg_inf:.1f}ms) | "
                          f"Det: {len(detections)} | "
                          f"Lane: {lane_str} | "
                          f"Speed: {speed:.1f} m/s")

            except socket.timeout:
                print("[PerceptionServer] Waiting for Unity connection...")
                continue

            except KeyboardInterrupt:
                print("\n[PerceptionServer] Shutting down...")
                self.running = False

        # ── CLEANUP ────────────────────────────────────────
        self.recv_socket.close()
        self.send_socket.close()

        total_time = time.time() - session_start
        avg_inf = (self.total_inference_time / self.frame_count
                   if self.frame_count > 0 else 0)

        print(f"\n[PerceptionServer] Session complete:")
        print(f"  Total frames:  {self.frame_count}")
        print(f"  Total time:    {total_time:.1f}s")
        print(f"  Avg FPS:       {self.frame_count / total_time:.1f}")
        print(f"  Avg inference: {avg_inf:.1f}ms")


if __name__ == "__main__":
    server = PerceptionServer()
    server.run()