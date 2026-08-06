import socket
import struct
import threading
import time
import cv2
import numpy as np
import json
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH      = r"C:\Projects\autosim-ai\models\autosim_v1\weights\best.pt"
RECEIVE_PORT    = 5006      # Unity sends frames TO this port
SEND_PORT       = 5007      # Python sends detections TO Unity on this port
UNITY_IP        = "127.0.0.1"
BUFFER_SIZE     = 65535     # Max UDP packet size
CONF_THRESHOLD  = 0.47      # Optimal from F1 curve
IMG_WIDTH       = 1920
IMG_HEIGHT      = 1080
PREVIEW_PATH    = r"C:\Projects\autosim-ai\perception\preview.jpg"
# ─────────────────────────────────────────────

class PerceptionServer:
    """
    Real-time perception server.
    Receives JPEG frames from Unity over UDP,
    runs YOLOv8 inference, sends detections back.
    Preview frames saved to disk every 30 frames.
    """

    def __init__(self):
        print("[PerceptionServer] Initialising...")

        # Load YOLOv8 model
        print(f"[PerceptionServer] Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        self.model.to('cuda')
        print("[PerceptionServer] Model loaded on GPU")

        # Class names matching training config
        self.class_names = {
            0: "vehicle",
            1: "pedestrian",
            2: "traffic_sign",
            3: "traffic_light"
        }

        # Class colours for visualisation (BGR)
        self.class_colors = {
            0: (0, 255, 0),    # vehicle — green
            1: (0, 0, 255),    # pedestrian — red
            2: (0, 255, 255),  # traffic_sign — yellow
            3: (255, 0, 0),    # traffic_light — blue
        }

        # Receive socket — listens for incoming frames from Unity
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.bind(("0.0.0.0", RECEIVE_PORT))
        self.recv_socket.settimeout(5.0)
        print(f"[PerceptionServer] Listening on port {RECEIVE_PORT}")

        # Send socket — sends detections back to Unity
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[PerceptionServer] Sending to {UNITY_IP}:{SEND_PORT}")

        # State
        self.running = False
        self.frame_count = 0
        self.total_inference_time = 0.0

        print("[PerceptionServer] Ready")

    def decode_frame(self, data):
        """
        Decode a received UDP packet into an image and telemetry.

        Packet format:
        [4 bytes: telemetry JSON length (big-endian uint32)]
        [N bytes: telemetry JSON string]
        [remaining bytes: JPEG image data]
        """
        try:
            # Read telemetry length (first 4 bytes)
            telem_len = struct.unpack('>I', data[:4])[0]

            # Read telemetry JSON
            telem_json = data[4:4 + telem_len].decode('utf-8')
            telemetry = json.loads(telem_json)

            # Read JPEG image data
            jpeg_data = data[4 + telem_len:]

            # Decode JPEG bytes to numpy array
            np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            return frame, telemetry

        except Exception as e:
            print(f"[ERROR] decode_frame failed: {e}")
            return None, {}

    def run_detection(self, frame):
        """
        Run YOLOv8 inference on a single frame.
        Returns a list of detection dictionaries and inference time in ms.
        """
        t_start = time.perf_counter()

        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            verbose=False,
            device='cuda'
        )

        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000

        detections = []
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # Normalised centre, width, height
                cx = ((x1 + x2) / 2) / frame.shape[1]
                cy = ((y1 + y2) / 2) / frame.shape[0]
                w  = (x2 - x1) / frame.shape[1]
                h  = (y2 - y1) / frame.shape[0]

                detections.append({
                    "class_id":   class_id,
                    "class_name": self.class_names.get(class_id, "unknown"),
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
        """
        Draw bounding boxes and labels on the frame.
        Used for preview image saved to disk.
        """
        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            color = self.class_colors.get(class_id, (255, 255, 255))

            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{det['class_name']} {det['confidence']:.2f}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0], y1),
                color, -1)

            # Label text
            cv2.putText(frame, label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)

        return frame

    def draw_overlay(self, frame, detections, telemetry,
                     inference_ms, current_fps):
        """
        Draw telemetry and stats overlay on the frame.
        """
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

        return frame

    def send_detections(self, detections, telemetry, inference_ms):
        """
        Send detection results back to Unity over UDP as JSON.
        """
        payload = {
            "frame_id":        self.frame_count,
            "inference_ms":    round(inference_ms, 2),
            "detection_count": len(detections),
            "detections":      detections,
            "telemetry_echo":  telemetry,
        }

        json_str = json.dumps(payload)
        data = json_str.encode('utf-8')

        try:
            self.send_socket.sendto(data, (UNITY_IP, SEND_PORT))
        except Exception as e:
            print(f"[ERROR] send_detections failed: {e}")

    def run(self):
        """
        Main server loop.
        Receives frames, runs inference, sends results.
        Saves annotated preview image to disk every 30 frames.
        """
        self.running = True
        print("\n[PerceptionServer] Server running.")
        print("[PerceptionServer] Preview saved to:", PREVIEW_PATH)
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

                # ── INFERENCE ──────────────────────────────
                detections, inference_ms = self.run_detection(frame)

                # ── SEND RESULTS TO UNITY ──────────────────
                self.send_detections(detections, telemetry, inference_ms)

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
                    vis = frame.copy()
                    vis = self.draw_detections(vis, detections)
                    vis = self.draw_overlay(
                        vis, detections, telemetry,
                        inference_ms, current_fps)
                    preview = cv2.resize(vis, (960, 540))
                    cv2.imwrite(PREVIEW_PATH, preview)

                # ── CONSOLE LOG EVERY 30 FRAMES ────────────
                if self.frame_count % 30 == 0:
                    avg_inf = (self.total_inference_time
                               / self.frame_count)
                    speed = telemetry.get("speed", 0)
                    print(f"[Frame {self.frame_count:05d}] "
                          f"FPS: {current_fps:.1f} | "
                          f"Inf: {inference_ms:.1f}ms "
                          f"(avg: {avg_inf:.1f}ms) | "
                          f"Det: {len(detections)} | "
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
        print(f"  Total frames:       {self.frame_count}")
        print(f"  Total time:         {total_time:.1f}s")
        print(f"  Average FPS:        {self.frame_count / total_time:.1f}")
        print(f"  Avg inference:      {avg_inf:.1f}ms")


if __name__ == "__main__":
    server = PerceptionServer()
    server.run()