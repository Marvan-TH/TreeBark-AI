from pathlib import Path
import shutil
import random

# ==============================
# CONFIGURATION
# ==============================

SOURCE_DIR = Path("processed_resized_data")
OUTPUT_DIR = Path("processed_prepared_data")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

CLASSES = ["cinnamon", "neem", "none"]

# ==============================
# CHECK SOURCE
# ==============================

if not SOURCE_DIR.exists():
    print("ERROR: processed_resized_data folder not found!")
    exit()

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        folder = OUTPUT_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)

# ==============================
# SPLIT DATASET
# ==============================

print("\n========================================")
print("PROCESSING PREPARED TREE BARK DATASET")
print("========================================\n")

for class_name in CLASSES:

    source_folder = SOURCE_DIR / class_name

    if not source_folder.exists():
        print(f"WARNING: {source_folder} not found")
        continue

    images = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend(source_folder.glob(ext))

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    print(f"{class_name.upper()}: {total} images")
    print(f"  Train: {len(train_images)}")
    print(f"  Val:   {len(val_images)}")
    print(f"  Test:  {len(test_images)}")

    # Copy images
    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / class_name / image.name
        )

    for image in val_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "val" / class_name / image.name
        )

    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / class_name / image.name
        )

print("\n========================================")
print("DATASET SPLITTING COMPLETE")
print("========================================")
print(f"\nOutput saved to: {OUTPUT_DIR}")