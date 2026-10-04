import os
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image
import imagehash


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset_converted"
REVIEW_DIR = "duplicate_review"

CLASSES = ["cinnamon", "neem", "none"]

EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# CREATE REVIEW FOLDER
# ============================================================

os.makedirs(REVIEW_DIR, exist_ok=True)


# ============================================================
# EXACT DUPLICATE HASH
# ============================================================

def get_file_hash(filepath):

    hasher = hashlib.sha256()

    with open(filepath, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            hasher.update(data)

    return hasher.hexdigest()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def get_perceptual_hash(filepath):

    image = Image.open(filepath).convert("RGB")

    return imagehash.phash(image)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TREE BARK DATASET - DUPLICATE CHECK")
print("=" * 70)

print(f"Dataset: {DATASET_DIR}")


total_images = 0

exact_hashes = defaultdict(list)

image_hashes = []


# ============================================================
# SCAN DATASET
# ============================================================

for class_name in CLASSES:

    class_dir = Path(DATASET_DIR) / class_name

    if not class_dir.exists():

        print(f"\nWARNING: {class_dir} not found")
        continue


    files = [
        file
        for file in class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in EXTENSIONS
    ]


    print(f"\n{class_name.upper()}")
    print("-" * 50)

    print(f"Images: {len(files)}")

    total_images += len(files)


    for filepath in files:

        # ----------------------------------------------------
        # Exact hash
        # ----------------------------------------------------

        try:

            file_hash = get_file_hash(filepath)

            exact_hashes[file_hash].append(
                str(filepath)
            )

        except Exception as e:

            print(f"Could not hash: {filepath}")
            print(e)

            continue


        # ----------------------------------------------------
        # Perceptual hash
        # ----------------------------------------------------

        try:

            phash = get_perceptual_hash(filepath)

            image_hashes.append(
                (
                    str(filepath),
                    class_name,
                    phash
                )
            )

        except Exception as e:

            print(f"Could not process: {filepath}")
            print(e)


# ============================================================
# EXACT DUPLICATES
# ============================================================

exact_duplicates = []

for file_hash, files in exact_hashes.items():

    if len(files) > 1:

        exact_duplicates.append(files)


# ============================================================
# NEAR DUPLICATES
# ============================================================

near_duplicates = []

DISTANCE_THRESHOLD = 5


for i in range(len(image_hashes)):

    file1, class1, hash1 = image_hashes[i]

    for j in range(i + 1, len(image_hashes)):

        file2, class2, hash2 = image_hashes[j]

        # Skip comparing an image with itself
        if file1 == file2:
            continue

        distance = hash1 - hash2

        if distance <= DISTANCE_THRESHOLD:

            near_duplicates.append(
                (
                    file1,
                    class1,
                    file2,
                    class2,
                    distance
                )
            )


# ============================================================
# SAVE REPORT
# ============================================================

report_file = Path(REVIEW_DIR) / "duplicate_report.txt"


with open(report_file, "w", encoding="utf-8") as report:

    report.write("=" * 70 + "\n")
    report.write("TREE BARK DATASET - DUPLICATE REPORT\n")
    report.write("=" * 70 + "\n\n")

    report.write(
        f"Total images scanned: {total_images}\n"
    )

    report.write(
        f"Exact duplicate groups: "
        f"{len(exact_duplicates)}\n"
    )

    report.write(
        f"Near duplicate pairs: "
        f"{len(near_duplicates)}\n"
    )


    # --------------------------------------------------------
    # EXACT DUPLICATES
    # --------------------------------------------------------

    report.write("\n\n")
    report.write("=" * 70 + "\n")
    report.write("EXACT DUPLICATES\n")
    report.write("=" * 70 + "\n")


    if exact_duplicates:

        for i, group in enumerate(
            exact_duplicates,
            start=1
        ):

            report.write(
                f"\nDuplicate group {i}\n"
            )

            for filepath in group:

                report.write(
                    f"  {filepath}\n"
                )

    else:

        report.write(
            "\nNo exact duplicates found.\n"
        )


    # --------------------------------------------------------
    # NEAR DUPLICATES
    # --------------------------------------------------------

    report.write("\n\n")
    report.write("=" * 70 + "\n")
    report.write("NEAR DUPLICATES\n")
    report.write("=" * 70 + "\n")


    if near_duplicates:

        for i, pair in enumerate(
            near_duplicates,
            start=1
        ):

            file1, class1, file2, class2, distance = pair

            report.write(
                f"\nPair {i} | Distance: {distance}\n"
            )

            report.write(
                f"  [{class1}] {file1}\n"
            )

            report.write(
                f"  [{class2}] {file2}\n"
            )

    else:

        report.write(
            "\nNo near duplicates detected.\n"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Total images scanned: {total_images}"
)

print(
    f"Exact duplicate groups: "
    f"{len(exact_duplicates)}"
)

print(
    f"Near duplicate pairs: "
    f"{len(near_duplicates)}"
)

print(
    f"\nReport saved to: {report_file}"
)

print("\nDuplicate check complete.")