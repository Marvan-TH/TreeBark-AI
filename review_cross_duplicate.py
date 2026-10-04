from pathlib import Path
from PIL import Image
import hashlib
import shutil

DATASET = Path("dataset")
REVIEW = Path("duplicate_review")

CINNAMON = DATASET / "cinnamon"
NEEM = DATASET / "neem"

REVIEW.mkdir(exist_ok=True)


def image_hash(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return None


def get_images(folder):
    images = []

    for path in folder.iterdir():
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".heic"]:
            h = image_hash(path)

            if h:
                images.append((h, path))

    return images


print("=" * 60)
print("CINNAMON ↔ NEEM DUPLICATE REVIEW")
print("=" * 60)

cinnamon_images = get_images(CINNAMON)
neem_images = get_images(NEEM)

neem_hashes = {}

for h, path in neem_images:
    neem_hashes.setdefault(h, []).append(path)

count = 0

for h, cinnamon_path in cinnamon_images:

    if h not in neem_hashes:
        continue

    for neem_path in neem_hashes[h]:

        count += 1

        folder = REVIEW / f"pair_{count:02d}"
        folder.mkdir(exist_ok=True)

        shutil.copy2(
            cinnamon_path,
            folder / f"CINNAMON{cinnamon_path.suffix.lower()}"
        )

        shutil.copy2(
            neem_path,
            folder / f"NEEM{neem_path.suffix.lower()}"
        )

        with open(folder / "INFO.txt", "w") as f:
            f.write(f"Cinnamon: {cinnamon_path}\n")
            f.write(f"Neem:     {neem_path}\n")

        print(f"Pair {count}:")
        print(f"  Cinnamon: {cinnamon_path}")
        print(f"  Neem:     {neem_path}")

print()
print("=" * 60)
print(f"TOTAL CROSS-CLASS PAIRS: {count}")
print("=" * 60)
print()
print("Review folders created in:")
print(REVIEW.resolve())