from pathlib import Path
from PIL import Image
import pillow_heif
import shutil

SOURCE_DIR = Path("dataset")
OUTPUT_DIR = Path("dataset_converted")

pillow_heif.register_heif_opener()

classes = ["cinnamon", "neem", "none"]

converted = 0
copied = 0
failed = 0

for class_name in classes:

    source_folder = SOURCE_DIR / class_name
    output_folder = OUTPUT_DIR / class_name

    output_folder.mkdir(parents=True, exist_ok=True)

    for file in source_folder.iterdir():

        if not file.is_file():
            continue

        extension = file.suffix.lower()

        try:

            # HEIC / HEIF → JPG
            if extension in [".heic", ".heif"]:

                print(f"Converting: {file.name}")

                image = Image.open(file)
                image = image.convert("RGB")

                output_file = output_folder / f"{file.stem}.jpg"

                image.save(
                    output_file,
                    "JPEG",
                    quality=95
                )

                converted += 1

            # Existing JPG/JPEG → copy
            elif extension in [".jpg", ".jpeg"]:

                output_file = output_folder / file.name

                shutil.copy2(file, output_file)

                copied += 1

            else:
                print(f"Skipped unsupported file: {file.name}")

        except Exception as e:

            failed += 1

            print(f"FAILED: {file.name}")
            print(e)


print("\n==============================")
print("CONVERSION COMPLETE")
print("==============================")
print(f"HEIC converted : {converted}")
print(f"JPG/JPEG copied: {copied}")
print(f"Failed         : {failed}")
print(f"Output         : {OUTPUT_DIR}")
print("==============================")