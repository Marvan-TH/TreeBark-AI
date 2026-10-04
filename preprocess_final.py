from pathlib import Path
from PIL import Image
from rembg import remove

# ==========================================
# FINAL TREE BARK DATASET PREPROCESSING
# ==========================================

SOURCE_DIR = Path("dataset")
OUTPUT_DIR = Path("processed_resized_data")

CLASSES = ["cinnamon", "neem", "none"]

IMAGE_SIZE = 224

print("=" * 60)
print("TREE BARK DATASET - FINAL PREPROCESSING")
print("=" * 60)

for class_name in CLASSES:

    source_folder = SOURCE_DIR / class_name
    output_folder = OUTPUT_DIR / class_name

    output_folder.mkdir(parents=True, exist_ok=True)

    images = (
        list(source_folder.glob("*.jpg"))
        + list(source_folder.glob("*.jpeg"))
        + list(source_folder.glob("*.png"))
    )

    print(f"\n{class_name.upper()}: {len(images)} images")

    for i, image_path in enumerate(images, 1):

        try:
            with Image.open(image_path) as img:

                # --------------------------------------
                # CINNAMON + NEEM
                # --------------------------------------
                if class_name in ["cinnamon", "neem"]:

                    img = img.convert("RGB")

                    # Remove background
                    result = remove(img)

                    # Create white background
                    background = Image.new(
                        "RGB",
                        result.size,
                        (255, 255, 255)
                    )

                    # Composite bark onto white background
                    if result.mode == "RGBA":
                        background.paste(
                            result,
                            mask=result.getchannel("A")
                        )
                        img = background
                    else:
                        img = result.convert("RGB")

                # --------------------------------------
                # NONE
                # --------------------------------------
                else:

                    # DO NOT remove background
                    img = img.convert("RGB")

                # --------------------------------------
                # RESIZE WITHOUT DISTORTION
                # --------------------------------------

                # Keep original aspect ratio
                img.thumbnail(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    Image.Resampling.LANCZOS
                )

                # White 224x224 canvas
                final_img = Image.new(
                    "RGB",
                    (IMAGE_SIZE, IMAGE_SIZE),
                    (255, 255, 255)
                )

                # Center the image
                x = (IMAGE_SIZE - img.width) // 2
                y = (IMAGE_SIZE - img.height) // 2

                final_img.paste(img, (x, y))

                # Save
                output_path = (
                    output_folder /
                    f"{image_path.stem}.jpg"
                )

                final_img.save(
                    output_path,
                    "JPEG",
                    quality=95
                )

            print(f"[{i}/{len(images)}] {image_path.name}")

        except Exception as e:
            print(f"ERROR: {image_path.name} -> {e}")

print("\n" + "=" * 60)
print("FINAL PREPROCESSING COMPLETE")
print("=" * 60)

print(f"\nOutput saved to:")
print(OUTPUT_DIR)