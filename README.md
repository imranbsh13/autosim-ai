# AutoSim AI

**AI-enabled autonomous driving perception and VR simulation platform**

Built as an independent portfolio project targeting Computer Vision, Perception Engineering, and AI Engineer roles. Demonstrates a complete end-to-end pipeline — from synthetic training data generation through deep learning model training, real-time sensor fusion, autonomous control logic, and on-device VR deployment on Oculus Quest 2.

> **Live demo:** Oculus Quest 2 APK running YOLOv8 at 8.4ms on-device via Unity Sentis — no Python server, no cloud, fully standalone.

---

## What this project demonstrates

| Skill area | Implementation |
|---|---|
| Synthetic data generation | Custom Unity capture system — automated YOLO annotation, no manual labelling |
| Deep learning | YOLOv8n fine-tuned on 5,747 synthetic images — mAP@0.5 0.681 |
| Classical computer vision | OpenCV lane detection — HSV masking, Canny, Hough transform |
| Sensor fusion | Pure numpy Kalman filter — camera detections + vehicle telemetry |
| Autonomous driving logic | Rule-based PID controller — lane keep, adaptive cruise, collision avoidance |
| Model deployment | ONNX export → Unity Sentis on-device inference at 8.4ms |
| VR/XR integration | Oculus Quest 2 standalone APK with real-time AI overlay |
| Simulation engineering | Unity 6 URP — city scene, vehicle physics, AI traffic, pedestrians |

---

## Tech stack

| Layer | Technologies |
|---|---|
| Simulation | Unity 6 (URP), Realistic Car Controller, Fantastic City Generator, Urban Traffic System |
| Data pipeline | C# CaptureManager, Python, Albumentations, YOLO format |
| CV & deep learning | YOLOv8n (Ultralytics), OpenCV, PyTorch, CUDA |
| Sensor fusion | Pure numpy Kalman filter, multi-object tracking with IoU matching |
| Inference & deployment | ONNX Runtime, Unity Sentis 2.6.1 |
| Networking | UDP socket bridge (Python ↔ Unity), ports 5006/5008 |
| VR | Oculus Quest 2, OpenXR, Unity XR Toolkit |
| MLOps | Git, versioned dataset, training metrics tracked per epoch |

---

## Project phases

- [x] **Phase 1** — Unity 6 simulation environment (city, vehicles, AI traffic, pedestrians)
- [x] **Phase 2** — Synthetic data generation pipeline (3,000 samples → 5,747 augmented)
- [x] **Phase 3** — Python AI perception pipeline (YOLOv8 + lane detection + Kalman + autonomous controller)
- [x] **Phase 4** — Oculus Quest 2 VR deployment (Unity Sentis on-device inference)
- [ ] **Phase 4.5** — Stereo vision depth estimation + Bird's Eye View projection *(in progress)*

---

## Results

| Metric | Value |
|---|---|
| YOLOv8 mAP@0.5 | **0.681** |
| Vehicle detection mAP@0.5 | 0.770 |
| Pedestrian detection mAP@0.5 | 0.591 |
| Precision | 0.874 |
| Recall | 0.600 |
| Python inference (RTX 4070) | **8–12ms** per frame |
| Sentis on-device (Quest 2) | **7.9–8.4ms** per frame |
| Training images | 5,747 (augmented from 2,414 raw) |
| Training classes | vehicle, pedestrian, traffic_sign, traffic_light |
| Pipeline throughput | 10–12 FPS end-to-end |
| Unity editor FPS | 60+ FPS |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — Unity 6 Simulation                           │
│  City scene · RCC vehicle · FCG traffic · UTS people    │
└──────────────────┬──────────────────────────────────────┘
                   │ RGB frames + telemetry (UDP 5006)
┌──────────────────▼──────────────────────────────────────┐
│  Layer 2 — Python AI Perception Pipeline                │
│  YOLOv8 detection · OpenCV lane detection               │
│  Kalman filter fusion · PID autonomous controller       │
└──────────────────┬──────────────────────────────────────┘
                   │ Control signals (UDP 5008)
┌──────────────────▼──────────────────────────────────────┐
│  Layer 3 — Unity Vehicle Control                        │
│  RCC external controller · steering · throttle · brake  │
└─────────────────────────────────────────────────────────┘

Quest 2 Standalone (Phase 4):
┌─────────────────────────────────────────────────────────┐
│  Unity Sentis · YOLOv8 ONNX · AI overlay · Hand HMI    │
│  No Python · No network · Fully on-device               │
└─────────────────────────────────────────────────────────┘
```

---

## Repository structure

```
autosim-ai/
├── unity-project/AutoSimAI/
│   └── Assets/
│       ├── Models/             ← best.onnx (Sentis model)
│       ├── Scenes/             ← MainDriving.unity
│       └── Scripts/
│           ├── DataCapture/    ← CaptureManager.cs, LabelledObject.cs
│           └── Perception/     ← PerceptionBridge.cs, SentisDetector.cs
├── perception/
│   ├── detection/
│   │   ├── inference_server.py ← real-time YOLOv8 UDP server
│   │   ├── lane_detector.py    ← OpenCV lane detection pipeline
│   │   ├── train_yolov8.py     ← YOLOv8 fine-tuning
│   │   ├── prepare_dataset.py  ← augmentation + YOLO format conversion
│   │   └── export_onnx.py      ← ONNX export for Sentis
│   ├── fusion/
│   │   └── kalman_fusion.py    ← pure numpy Kalman filter tracker
│   └── autonomous/
│       └── autonomous_controller.py ← PID + ACC + collision avoidance
├── data/                       ← dataset (gitignored, tracked separately)
├── models/                     ← trained weights (gitignored)
└── docs/                       ← architecture diagrams, reports
```

---

## How to run

### Prerequisites

- Unity 6000.0.77f1 with Android Build Support
- Python 3.12 with CUDA (pytorch_env conda environment)
- NVIDIA GPU (tested on RTX 4070 Laptop)
- Oculus Quest 2 with developer mode enabled

### Phase 3 — Live AI pipeline (PC)

```bash
# 1. Activate environment
conda activate pytorch_env

# 2. Start Python perception server
cd C:\Projects\autosim-ai
python perception/detection/inference_server.py

# 3. Open Unity, press Play
# The server connects automatically and begins inference
```

### Phase 4 — Quest 2 VR

1. Open Unity → File → Build Settings → Android
2. Connect Quest 2 via USB
3. Enable developer mode on Quest 2
4. Build and Run

No Python server needed — Sentis runs the model on-device.

### Training (optional — retrain on your own data)

```bash
# Collect data: press Play in Unity with CaptureManager enabled
# Then run:

python perception/detection/prepare_dataset.py
python perception/detection/train_yolov8.py
python perception/detection/export_onnx.py
```

---

## Key technical decisions

**Why synthetic data?**
Real-world dashcam annotation is expensive and slow. Unity generates perfect ground truth labels automatically — bounding boxes, segmentation masks, depth maps. 3,000 labelled frames collected in 10 minutes of driving.

**Why YOLOv8n (nano)?**
Fits within Quest 2's Snapdragon XR2 memory budget after ONNX export. Achieves 8.4ms inference on-device — sufficient for real-time detection at 10+ FPS. Larger variants improve mAP but won't run on mobile VR hardware.

**Why Kalman filter over a neural tracker?**
Interpretable, debuggable, zero training data required. The Kalman filter's predict/update cycle handles YOLOv8's frame-to-frame detection noise and provides velocity and distance estimates that feed the autonomous controller. Implemented from scratch in pure numpy — no filterpy dependency.

**Why UDP over TCP?**
For real-time 60 FPS driving, TCP's handshake overhead introduces unacceptable latency. UDP fires packets without waiting for acknowledgement — dropped frames are skipped rather than queuing.

**Why URP over HDRP?**
Quest 2's Snapdragon XR2 cannot run HDRP. URP supports all three asset packages (RCC, FCG, UTS) out of the box and is fully compatible with OpenXR and Unity Sentis.

---

## Known limitations and next steps

| Limitation | Planned fix |
|---|---|
| Pedestrian recall 0.60 | Collect closer-range pedestrian footage |
| Traffic sign/light detection | Label signs in Unity scene, retrain |
| Distance estimation empirical | Phase 4.5: stereo vision for geometric depth |
| No BEV map | Phase 4.5: homography projection minimap |
| Lane detection fails on curves | Upgrade to segmentation-based lane model |

---

## About

Built by **Imran Momin** — VR/XR Engineer at Bentley Motors, transitioning into Computer Vision and AI Engineering.

- 7+ years XR development (Unity, Varjo XR-4, Oculus Quest 2)
- Previous: ANSYS Sheffield (2018–2019)
- Notable: Mixed Reality demo at F1 British Grand Prix, Silverstone (July 2023)

**Contact:** imranbsh13@gmail.com  
**LinkedIn:** linkedin.com/in/momin-imran  
**GitHub:** github.com/imranbsh13