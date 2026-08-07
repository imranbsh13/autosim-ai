from ultralytics import YOLO
import torch
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_YAML   = r"C:\Projects\autosim-ai\data\data.yaml"
MODEL       = "yolov8n.pt"       # nano — fastest, fits Quest 2
EPOCHS      = 50
BATCH_SIZE  = 16
IMAGE_SIZE  = 640
PROJECT_DIR = r"C:\Projects\autosim-ai\models"
RUN_NAME    = "autosim_v1"
# ─────────────────────────────────────────────

def main():
    # Verify GPU is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Load pretrained YOLOv8n
    # yolov8n.pt = nano variant, pretrained on COCO 80 classes
    # Downloads automatically on first run (~6MB)
    model = YOLO(MODEL)

    print(f"Starting training on: {DATA_YAML}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Image size: {IMAGE_SIZE}")
    print()

    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        device=0,                # 0 = first GPU (RTX 4070)
        workers=4,               # data loading threads
        patience=20,             # stop early if no improvement for 20 epochs
        save=True,               # save best and last checkpoints
        save_period=10,          # also save every 10 epochs
        cache=False,             # don't cache images in RAM
        pretrained=True,         # use COCO pretrained weights
        optimizer="AdamW",       # AdamW optimizer
        lr0=0.001,               # initial learning rate
        lrf=0.01,                # final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,         # gradually increase lr for first 3 epochs
        warmup_momentum=0.8,
        box=7.5,                 # box loss weight
        cls=0.5,                 # classification loss weight
        dfl=1.5,                 # distribution focal loss weight
        augment=True,            # YOLOv8's own built-in augmentation
        degrees=5.0,             # rotation augmentation
        translate=0.1,           # translation augmentation
        scale=0.5,               # scale augmentation
        flipud=0.0,              # vertical flip (0 = off, cars don't appear upside down)
        fliplr=0.5,              # horizontal flip
        mosaic=1.0,              # mosaic augmentation (combines 4 images)
        mixup=0.1,               # mixup augmentation
        verbose=True,
    )

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best model saved to: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    print(f"Results: {PROJECT_DIR}/{RUN_NAME}/")
    print("="*50)

    # Quick validation on val set
    print("\nRunning validation on best model...")
    best_model = YOLO(f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    metrics = best_model.val(data=DATA_YAML, device=0)

    print(f"\nFinal Metrics:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()