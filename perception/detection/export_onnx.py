from ultralytics import YOLO
import os

# ─────────────────────────────────────────────
MODEL_PATH  = r"C:\Projects\autosim-ai\models\autosim_v1\weights\best.pt"
OUTPUT_DIR  = r"C:\Projects\autosim-ai\models"
# ─────────────────────────────────────────────

def export_onnx():
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Exporting to ONNX...")
    # opset=12 is required for Unity Sentis compatibility
    # dynamic=False fixes input shape for Quest 2
    # simplify=True reduces model complexity
    path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
    )

    print(f"ONNX model saved to: {path}")

    # Verify file size
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    print("Done. Copy this .onnx file to Unity Assets/Models/")

if __name__ == "__main__":
    export_onnx()