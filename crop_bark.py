import os
import cv2


# ==========================================
# SETTINGS
# ==========================================

SOURCE_DIR = "processed_data_new"
OUTPUT_DIR = "cropped_data_new"

CLASSES = ["cinnamon", "neem", "none"]

PADDING = 25


# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

for split in ["train", "val", "test"]:

    for class_name in CLASSES:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                class_name
            ),
            exist_ok=True
        )


# ==========================================
# CROP BARK
# ==========================================

def crop_bark(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect darker pixels against white background
    mask = cv2.inRange(
        gray,
        0,
        245
    )

    # Remove small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    # Select largest detected object
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # Ignore extremely small detections
    if cv2.contourArea(largest_contour) < 100:
        return image

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # Add padding
    x1 = max(
        0,
        x - PADDING
    )

    y1 = max(
        0,
        y - PADDING
    )

    x2 = min(
        image.shape[1],
        x + w + PADDING
    )

    y2 = min(
        image.shape[0],
        y + h + PADDING
    )

    return image[
        y1:y2,
        x1:x2
    ]


# ==========================================
# PROCESS DATASET
# ==========================================

total = 0

for split in ["train", "val", "test"]:

    for class_name in CLASSES:

        input_dir = os.path.join(
            SOURCE_DIR,
            split,
            class_name
        )

        output_dir = os.path.join(
            OUTPUT_DIR,
            split,
            class_name
        )

        if not os.path.exists(input_dir):

            print(
                f"Skipping missing folder: "
                f"{input_dir}"
            )

            continue

        for filename in os.listdir(input_dir):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                continue

            input_path = os.path.join(
                input_dir,
                filename
            )

            output_path = os.path.join(
                output_dir,
                filename
            )

            image = cv2.imread(input_path)

            if image is None:

                print(
                    f"Could not read: {input_path}"
                )

                continue

            # Crop only bark images
            if class_name in ["cinnamon", "neem"]:

                image = crop_bark(image)

            # Keep NONE images unchanged
            cv2.imwrite(
                output_path,
                image
            )

            total += 1


# ==========================================
# DONE
# ==========================================

print("\n" + "=" * 60)
print("CROPPING COMPLETE")
print("=" * 60)

print(
    f"\nProcessed images: {total}"
)

print(
    f"Saved to: {OUTPUT_DIR}"
)

print(
    "\nOriginal dataset was NOT modified."
)