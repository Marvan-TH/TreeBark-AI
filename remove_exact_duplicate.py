import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

DATASET_DIR = Path("dataset_converted")
QUARANTINE_DIR = Path("duplicate_review") / "removed_exact"

CLASSES = ["cinnamon", "neem", "none"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)


def get_file_hash(filepath):
    hasher = hashlib.sha256()

    with open(filepath, "rb") as f:
        while True:
            data = f.read(1024 * 1024)

            if not data:
                break

            hasher.update(data)

    return hasher.hexdigest()


# ------------------------------------------------------------
# Find exact duplicates
# ------------------------------------------------------------

hashes = defaultdict(list)

for class_name in CLASSES:

    folder = DATASET_DIR / class_name

    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in EXTENSIONS:
            continue

        file_hash = get_file_hash(file)

        hashes[file_hash].append(file)


# ------------------------------------------------------------
# Move duplicate copies
# ------------------------------------------------------------

moved = 0
groups = 0

print("=" * 60)
print("EXACT DUPLICATE CLEANUP")
print("=" * 60)

for file_hash, files in hashes.items():

    if len(files) <= 1:
        continue

    groups += 1

    # Keep the first image
    keep = files[0]

    print(f"\nDuplicate group {groups}")
    print(f"KEEP: {keep}")

    # Move remaining copies
    for duplicate in files[1:]:

        destination = QUARANTINE_DIR / duplicate.name

        # Avoid filename collision
        counter = 1

        while destination.exists():

            destination = (
                QUARANTINE_DIR
                / f"{duplicate.stem}_{counter}{duplicate.suffix}"
            )

            counter += 1

        shutil.move(str(duplicate), str(destination))

        print(f"MOVED: {duplicate}")
        print(f"   -> {destination}")

        moved += 1


print("\n" + "=" * 60)
print("CLEANUP COMPLETE")
print("=" * 60)

print(f"Duplicate groups : {groups}")
print(f"Files moved      : {moved}")

print(f"\nQuarantine folder:")
print(QUARANTINE_DIR)

print("=" * 60)