from pathlib import Path
from rembg import remove
from PIL import Image
import shutil

# ============================================================
# TREE BARK DATASET - BACKGROUND REMOVAL
# ============================================================

BASE_DIR = Path("prepared_data_new")
OUTPUT_DIR = Path("processed_data_new")

CLASSES = ["cinnamon", "neem", "none"]
SPLITS = ["train", "val", "test"]

# Minimum percentage of image that should contain
# detected foreground
MIN_FOREGROUND_RATIO = 0.02

# Padding around detected foreground
PADDING = 20

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

print("=" * 60)
print("TREE BARK DATASET - BACKGROUND REMOVAL")
print("=" * 60)

print(f"Input  : {BASE_DIR}")
print(f"Output : {OUTPUT_DIR}")


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if OUTPUT_DIR.exists():

    print("\nRemoving previous processed_data_new...")

    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for split in SPLITS:

    for class_name in CLASSES:

        folder = (
            OUTPUT_DIR
            / split
            / class_name
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(image_path, output_path):

    with Image.open(image_path) as img:

        # ----------------------------------------------------
        # CONVERT TO RGB
        # ----------------------------------------------------

        img = img.convert("RGB")

        # ----------------------------------------------------
        # BACKGROUND REMOVAL
        # ----------------------------------------------------

        result = remove(img)

        result = result.convert("RGBA")

        # ----------------------------------------------------
        # FIND FOREGROUND
        # ----------------------------------------------------

        alpha = result.getchannel("A")

        bbox = alpha.getbbox()

        # ----------------------------------------------------
        # IF NO FOREGROUND IS DETECTED
        # ----------------------------------------------------

        if bbox is None:

            print(
                "    WARNING: No foreground detected - "
                "keeping original image"
            )

            final_img = img

        else:

            left, top, right, bottom = bbox

            width, height = img.size

            bbox_area = (
                (right - left)
                * (bottom - top)
            )

            image_area = width * height

            foreground_ratio = (
                bbox_area / image_area
            )

            # ------------------------------------------------
            # VERY SMALL DETECTION
            # ------------------------------------------------

            if foreground_ratio < MIN_FOREGROUND_RATIO:

                print(
                    f"    WARNING: Small foreground "
                    f"({foreground_ratio * 100:.2f}%)"
                )

                # Keep original image
                final_img = img

            else:

                # ------------------------------------------------
                # ADD PADDING
                # ------------------------------------------------

                left = max(
                    0,
                    left - PADDING
                )

                top = max(
                    0,
                    top - PADDING
                )

                right = min(
                    width,
                    right + PADDING
                )

                bottom = min(
                    height,
                    bottom + PADDING
                )

                # ------------------------------------------------
                # CROP FOREGROUND
                # ------------------------------------------------

                cropped = result.crop(
                    (
                        left,
                        top,
                        right,
                        bottom
                    )
                )

                # ------------------------------------------------
                # CREATE WHITE BACKGROUND
                # ------------------------------------------------

                white_background = Image.new(
                    "RGB",
                    cropped.size,
                    (255, 255, 255)
                )

                # ------------------------------------------------
                # PASTE FOREGROUND ON WHITE BACKGROUND
                # ------------------------------------------------

                white_background.paste(
                    cropped,
                    mask=cropped.getchannel("A")
                )

                final_img = white_background

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        final_img.save(
            output_path,
            "PNG"
        )


# ============================================================
# PROCESS DATASET
# ============================================================

total_processed = 0
total_failed = 0


for split in SPLITS:

    print("\n" + "=" * 60)
    print(f"PROCESSING: {split.upper()}")
    print("=" * 60)

    for class_name in CLASSES:

        source_folder = (
            BASE_DIR
            / split
            / class_name
        )

        output_folder = (
            OUTPUT_DIR
            / split
            / class_name
        )

        if not source_folder.exists():

            print(
                f"\nWARNING: {source_folder} not found"
            )

            continue

        images = [

            file

            for file in source_folder.iterdir()

            if file.is_file()
            and file.suffix.lower()
            in VALID_EXTENSIONS

        ]

        print(
            f"\n{class_name.upper()}: "
            f"{len(images)} images"
        )

        # ====================================================
        # PROCESS ALL CLASSES
        # ====================================================

        for i, image_path in enumerate(
            images,
            1
        ):

            try:

                output_path = (
                    output_folder
                    / f"{image_path.stem}.png"
                )

                process_image(
                    image_path,
                    output_path
                )

                total_processed += 1

                print(
                    f"  [{i}/{len(images)}] "
                    f"{image_path.name}"
                )

            except Exception as e:

                print(
                    f"ERROR: "
                    f"{image_path.name} -> {e}"
                )

                total_failed += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BACKGROUND REMOVAL COMPLETE")
print("=" * 60)

print(
    f"Processed images : "
    f"{total_processed}"
)

print(
    f"Failed           : "
    f"{total_failed}"
)

print(
    f"\nOutput saved to: "
    f"{OUTPUT_DIR}"
)

print("\nOutput structure:")

print("processed_data_new/")
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

print("\nAll three classes have undergone background removal.")
print("=" * 60)