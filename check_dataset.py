from pathlib import Path
from PIL import Image

DATASET_DIR = Path("dataset")

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

classes = ["cinnamon", "neem", "none"]

print("=" * 60)
print("TREE BARK DATASET CHECK")
print("=" * 60)

total_valid = 0
total_invalid = 0

for class_name in classes:

    folder = DATASET_DIR / class_name

    if not folder.exists():
        print(f"\n{class_name.upper()}: folder not found")
        continue

    valid = 0
    invalid = 0
    dimensions = {}

    for file in folder.iterdir():

        if file.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            with Image.open(file) as img:

                # Verify that the image can actually be read
                img.verify()

            # Open again because verify() closes the file
            with Image.open(file) as img:
                width, height = img.size

            valid += 1
            total_valid += 1

            dimensions[(width, height)] = dimensions.get(
                (width, height), 0
            ) + 1

        except Exception as e:
            invalid += 1
            total_invalid += 1

            print(f"INVALID: {file}")

    print(f"\n{class_name.upper()}")
    print("-" * 30)
    print(f"Valid images   : {valid}")
    print(f"Invalid images : {invalid}")

    print("\nImage dimensions:")

    for dimension, count in sorted(
        dimensions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]:

        print(f"  {dimension[0]} × {dimension[1]} → {count} images")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total valid images   : {total_valid}")
print(f"Total invalid images : {total_invalid}")