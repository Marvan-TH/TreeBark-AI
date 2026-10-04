from pathlib import Path
import shutil
import random
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

# NEW cleaned dataset
SOURCE_DIR = Path("dataset")

# NEW output folder
OUTPUT_DIR = Path("prepared_data_new")

CLASSES = ["cinnamon", "neem", "none"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(RANDOM_SEED)


# ============================================================
# CHECK RATIOS
# ============================================================

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 0.001


# ============================================================
# REMOVE OLD NEW-SPLIT FOLDER IF IT EXISTS
# ============================================================

if OUTPUT_DIR.exists():
    print("Removing previous prepared_data_new folder...")
    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for class_name in CLASSES:

    for split in ["train", "val", "test"]:

        folder = OUTPUT_DIR / split / class_name

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS DATASET
# ============================================================

total_images = 0

print("=" * 60)
print("TREE BARK DATASET PREPARATION")
print("=" * 60)

print(f"Source : {SOURCE_DIR}")
print(f"Output : {OUTPUT_DIR}")


for class_name in CLASSES:

    source_folder = SOURCE_DIR / class_name

    if not source_folder.exists():

        print(
            f"\nWARNING: {source_folder} not found"
        )

        continue


    # --------------------------------------------------------
    # GET VALID IMAGE FILES
    # --------------------------------------------------------

    images = [

        file

        for file in source_folder.iterdir()

        if file.is_file()

        and file.suffix.lower() in VALID_EXTENSIONS

    ]


    # --------------------------------------------------------
    # VERIFY IMAGES
    # --------------------------------------------------------

    valid_images = []

    for file in images:

        try:

            with Image.open(file) as img:
                img.verify()

            valid_images.append(file)

        except Exception as e:

            print(
                f"Skipping corrupted image: {file.name}"
            )


    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    random.shuffle(valid_images)


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    total = len(valid_images)

    train_end = int(total * TRAIN_RATIO)

    val_end = (
        train_end
        + int(total * VAL_RATIO)
    )


    train_images = valid_images[:train_end]

    val_images = valid_images[
        train_end:val_end
    ]

    test_images = valid_images[
        val_end:
    ]


    splits = {

        "train": train_images,

        "val": val_images,

        "test": test_images

    }


    # --------------------------------------------------------
    # PRINT COUNTS
    # --------------------------------------------------------

    print("\n" + class_name.upper())

    print("-" * 30)

    print(f"Total : {total}")

    print(
        f"Train : {len(train_images)}"
    )

    print(
        f"Val   : {len(val_images)}"
    )

    print(
        f"Test  : {len(test_images)}"
    )


    # --------------------------------------------------------
    # COPY FILES
    # --------------------------------------------------------

    for split_name, split_images in splits.items():

        destination = (
            OUTPUT_DIR
            / split_name
            / class_name
        )

        for image in split_images:

            shutil.copy2(
                image,
                destination / image.name
            )


    total_images += total


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(
    f"Total images prepared: {total_images}"
)

print("\nOutput structure:")

print("prepared_data_new/")

print("├── train/")

print("│   ├── cinnamon/")

print("│   ├── neem/")

print("│   └── none/")

print("├── val/")

print("│   ├── cinnamon/")

print("│   ├── neem/")

print("│   └── none/")

print("└── test/")

print("    ├── cinnamon/")

print("    ├── neem/")

print("    └── none/")