import os
import shutil
import random
import cv2
import albumentations as A
from tqdm import tqdm
import yaml

# ─────────────────────────────────────────────
# CONFIGURATION — update these paths
# ─────────────────────────────────────────────
RAW_RGB_DIR = r"C:\Projects\autosim-ai\unity-project\AutoSimAI\CaptureData"
OUTPUT_DIR  = r"C:\Projects\autosim-ai\data"

CLASS_NAMES = ["vehicle", "pedestrian", "traffic_sign", "traffic_light"]

TRAIN_SPLIT = 0.8          # 80% train, 20% val
AUGMENT_MULTIPLIER = 2     # Each image gets 2 augmented copies
SEED = 42
# ─────────────────────────────────────────────

random.seed(SEED)

def find_latest_session(base_dir):
    """Find the most recent capture session folder."""
    sessions = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not sessions:
        raise FileNotFoundError(f"No session folders found in {base_dir}")
    latest = sorted(sessions)[-1]
    print(f"[INFO] Using session: {latest}")
    return os.path.join(base_dir, latest)

def load_samples(session_dir):
    """Load all valid image/label pairs."""
    rgb_dir   = os.path.join(session_dir, "rgb")
    label_dir = os.path.join(session_dir, "labels")

    samples = []
    skipped = 0

    image_files = sorted([
        f for f in os.listdir(rgb_dir) if f.endswith(".jpg")
    ])

    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        img_path   = os.path.join(rgb_dir, img_file)
        label_path = os.path.join(label_dir, f"{base}.txt")

        if not os.path.exists(label_path):
            skipped += 1
            continue

        # Read label file
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        # Skip empty label files (no objects visible)
        if len(lines) == 0:
            skipped += 1
            continue

        samples.append((img_path, label_path, lines))

    print(f"[INFO] Loaded {len(samples)} valid samples, skipped {skipped}")
    return samples

def get_augmentation_pipeline():
    """
    Albumentations augmentation pipeline.
    
    What each transform does and why:
    
    RandomBrightnessContrast — simulates different lighting conditions,
    dawn/dusk/overcast. Real cameras see huge brightness variation.
    
    GaussNoise — simulates camera sensor noise, especially in low light.
    Real camera sensors add random noise to every pixel.
    
    MotionBlur — simulates camera movement at speed. At 60mph a dashcam
    image has motion blur from both the car and moving objects.
    
    RandomRain — overlays rain streaks. Our Unity scene has clear weather
    so this bridges the domain gap for wet conditions.
    
    HorizontalFlip — mirrors the image. Doubles dataset size and teaches
    the model that cars on the right look the same as cars on the left.
    
    CLAHE (Contrast Limited Adaptive Histogram Equalization) — improves
    local contrast. Helps detection in shadowed or low-contrast areas.
    
    RandomFog — simulates fog/mist conditions Unity doesn't generate.
    
    ImageCompression — simulates JPEG compression artefacts from real
    dashcam footage which is always compressed.
    """
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.GaussNoise(
            std_range=(0.01, 0.05),
            p=0.5
        ),
        A.MotionBlur(
            blur_limit=(3, 7),
            p=0.4
        ),
        A.RandomRain(
            slant_lower=-10,
            slant_upper=10,
            drop_length=15,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.85,
            rain_type='drizzle',
            p=0.3
        ),
        A.HorizontalFlip(p=0.5),
        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=0.4
        ),
        A.RandomFog(
            fog_coef_lower=0.1,
            fog_coef_upper=0.3,
            alpha_coef=0.1,
            p=0.2
        ),
        A.ImageCompression(
            quality_range=(70, 95),
            p=0.4
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))

def flip_yolo_labels(lines):
    """Flip bounding box x coordinates for horizontal flip."""
    flipped = []
    for line in lines:
        parts = line.split()
        cls = parts[0]
        cx, cy, w, h = float(parts[1]), float(parts[2]), \
                       float(parts[3]), float(parts[4])
        # Horizontal flip: new_cx = 1 - cx
        flipped.append(f"{cls} {1-cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return flipped

def parse_yolo_labels(lines):
    """Parse YOLO label lines into class_labels and bboxes lists."""
    class_labels = []
    bboxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), \
                       float(parts[3]), float(parts[4])
        # Clamp to valid range
        cx = max(0.001, min(1.0, cx))
        cy = max(0.001, min(1.0, cy))
        w  = max(0.001, min(1.0, w))
        h  = max(0.001, min(1.0, h))
        class_labels.append(cls)
        bboxes.append([cx, cy, w, h])
    return class_labels, bboxes

def save_yolo_labels(path, class_labels, bboxes):
    """Save class_labels and bboxes back to YOLO format text file."""
    lines = []
    for cls, bbox in zip(class_labels, bboxes):
        cx, cy, w, h = bbox
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))

def build_dataset(samples, output_dir, augment_pipeline):
    """Build final dataset with augmentation and train/val split."""

    # Create output folders
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Shuffle and split
    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_samples = samples[:split_idx]
    val_samples   = samples[split_idx:]

    print(f"[INFO] Train: {len(train_samples)} | Val: {len(val_samples)}")

    # Process training samples with augmentation
    print("[INFO] Processing training samples with augmentation...")
    train_count = 0
    for img_path, label_path, lines in tqdm(train_samples, desc="Train"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        class_labels, bboxes = parse_yolo_labels(lines)
        if not class_labels:
            continue

        # Save original
        base_name = f"train_{train_count:06d}"
        cv2.imwrite(
            os.path.join(output_dir, "images", "train", f"{base_name}.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        save_yolo_labels(
            os.path.join(output_dir, "labels", "train", f"{base_name}.txt"),
            class_labels, bboxes
        )
        train_count += 1

        # Save augmented copies
        for aug_idx in range(AUGMENT_MULTIPLIER):
            try:
                result = augment_pipeline(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                aug_img    = result["image"]
                aug_bboxes = result["bboxes"]
                aug_labels = result["class_labels"]

                if not aug_labels:
                    continue

                aug_name = f"train_{train_count:06d}"
                cv2.imwrite(
                    os.path.join(output_dir, "images", "train",
                                 f"{aug_name}.jpg"),
                    cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                )
                save_yolo_labels(
                    os.path.join(output_dir, "labels", "train",
                                 f"{aug_name}.txt"),
                    aug_labels, list(aug_bboxes)
                )
                train_count += 1

            except Exception as e:
                print(f"[WARN] Augmentation failed for {img_path}: {e}")
                continue

    # Process validation samples — NO augmentation
    # Val set must reflect real distribution to give honest metrics
    print("[INFO] Processing validation samples (no augmentation)...")
    val_count = 0
    for img_path, label_path, lines in tqdm(val_samples, desc="Val"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        class_labels, bboxes = parse_yolo_labels(lines)
        if not class_labels:
            continue

        base_name = f"val_{val_count:06d}"
        cv2.imwrite(
            os.path.join(output_dir, "images", "val", f"{base_name}.jpg"),
            img
        )
        save_yolo_labels(
            os.path.join(output_dir, "labels", "val", f"{base_name}.txt"),
            class_labels, bboxes
        )
        val_count += 1

    print(f"[INFO] Final dataset: {train_count} train | {val_count} val")
    return train_count, val_count

def create_data_yaml(output_dir, train_count, val_count):
    """Create data.yaml file for YOLOv8 training."""
    data = {
        "path": output_dir.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[INFO] data.yaml saved to: {yaml_path}")
    return yaml_path

def print_class_distribution(samples):
    """Print how many annotations per class."""
    counts = {}
    for _, _, lines in samples:
        for line in lines:
            cls = int(line.split()[0])
            counts[cls] = counts.get(cls, 0) + 1
    print("\n[INFO] Class distribution in raw data:")
    for cls_id, count in sorted(counts.items()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"  Class {cls_id} ({name}): {count} annotations")
    print()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("AutoSim AI — Dataset Preparation Pipeline")
    print("=" * 50)

    # Find latest capture session
    session_dir = find_latest_session(RAW_RGB_DIR)

    # Load all valid samples
    samples = load_samples(session_dir)

    # Show class distribution
    print_class_distribution(samples)

    # Build augmentation pipeline
    aug_pipeline = get_augmentation_pipeline()

    # Build dataset
    train_count, val_count = build_dataset(samples, OUTPUT_DIR, aug_pipeline)

    # Create data.yaml
    yaml_path = create_data_yaml(OUTPUT_DIR, train_count, val_count)

    print("\n" + "=" * 50)
    print("Dataset preparation complete!")
    print(f"Total training images: {train_count}")
    print(f"Total validation images: {val_count}")
    print(f"data.yaml: {yaml_path}")
    print("=" * 50)