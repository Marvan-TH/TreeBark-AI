import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from rembg import remove
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_hybrid_cnn.pth"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE PATH
# ============================================================

IMAGE_PATH = input(
    "Enter image path: "
).strip().strip('"')


# ============================================================
# HYBRID CNN
# ============================================================

class HybridCNN(nn.Module):

    def __init__(self):

        super(HybridCNN, self).__init__()

        # ====================================================
        # BRANCH A - 3x3 CONVOLUTIONS
        # ====================================================

        self.branch_a = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1))
        )


        # ====================================================
        # BRANCH B - 5x5 + 3x3 CONVOLUTIONS
        # ====================================================

        self.branch_b = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                32, 64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1))
        )


        # ====================================================
        # FEATURE FUSION
        # ====================================================

        self.fc1 = nn.Linear(
            256 + 128,
            128
        )

        self.dropout = nn.Dropout(0.4)

        self.fc2 = nn.Linear(
            128,
            3
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        features_a = self.branch_a(x)

        features_a = torch.flatten(
            features_a,
            1
        )


        features_b = self.branch_b(x)

        features_b = torch.flatten(
            features_b,
            1
        )


        # ====================================================
        # FEATURE FUSION
        # ====================================================

        features = torch.cat(
            [
                features_a,
                features_b
            ],
            dim=1
        )


        x = self.fc1(features)

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# BACKGROUND REMOVAL
# SAME METHOD USED FOR DATASET
# ============================================================

def remove_background(image):

    print("\nRemoving background...")

    # --------------------------------------------------------
    # Remove background using rembg
    # --------------------------------------------------------

    result = remove(image)

    result = result.convert("RGBA")

    # --------------------------------------------------------
    # Get foreground mask
    # --------------------------------------------------------

    alpha = result.getchannel("A")

    bbox = alpha.getbbox()

    # --------------------------------------------------------
    # If no foreground detected
    # --------------------------------------------------------

    if bbox is None:

        print("⚠ Background removal failed.")
        print("Using original image.")

        return image


    left, top, right, bottom = bbox

    width, height = image.size


    # --------------------------------------------------------
    # Calculate foreground ratio
    # SAME AS DATASET
    # --------------------------------------------------------

    bbox_area = (
        (right - left) *
        (bottom - top)
    )

    image_area = (
        width *
        height
    )

    foreground_ratio = (
        bbox_area / image_area
    )


    # --------------------------------------------------------
    # Small foreground
    # SAME THRESHOLD AS DATASET
    # --------------------------------------------------------

    if foreground_ratio < 0.02:

        print(
            f"⚠ Small foreground detected "
            f"({foreground_ratio * 100:.2f}%)"
        )

        print("Using original image.")

        return image


    # --------------------------------------------------------
    # Padding
    # SAME AS DATASET
    # --------------------------------------------------------

    padding = 20

    left = max(
        0,
        left - padding
    )

    top = max(
        0,
        top - padding
    )

    right = min(
        width,
        right + padding
    )

    bottom = min(
        height,
        bottom + padding
    )


    # --------------------------------------------------------
    # Crop foreground
    # --------------------------------------------------------

    cropped = result.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )


    # --------------------------------------------------------
    # Create white background
    # SAME AS DATASET
    # --------------------------------------------------------

    white_background = Image.new(
        "RGB",
        cropped.size,
        (255, 255, 255)
    )


    white_background.paste(
        cropped,
        mask=cropped.getchannel("A")
    )


    print("✓ Background removed")

    return white_background


# ============================================================
# BARK CROPPING
# SAME BASIC METHOD USED FOR DATASET
# ============================================================

def crop_bark(image):

    print("\nDetecting bark...")


    # --------------------------------------------------------
    # PIL → OpenCV
    # --------------------------------------------------------

    image_cv = np.array(image)

    image_cv = cv2.cvtColor(
        image_cv,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Same threshold used in dataset crop
    # --------------------------------------------------------

    _, mask = cv2.threshold(
        gray,
        245,
        255,
        cv2.THRESH_BINARY_INV
    )


    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # --------------------------------------------------------
    # No contour
    # --------------------------------------------------------

    if not contours:

        print("⚠ Could not detect bark.")
        print("Using background-removed image.")

        return image


    # --------------------------------------------------------
    # Largest contour
    # --------------------------------------------------------

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


    # --------------------------------------------------------
    # Bounding rectangle
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )


    # --------------------------------------------------------
    # Check bark size
    # --------------------------------------------------------

    image_area = (
        image_cv.shape[0] *
        image_cv.shape[1]
    )

    bark_area = (
        w * h
    )


    if bark_area < image_area * 0.02:

        print("⚠ Bark detection uncertain.")
        print("Using background-removed image.")

        return image


    # --------------------------------------------------------
    # Padding
    # SAME AS DATASET CROP
    # --------------------------------------------------------

    padding = 20

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        image_cv.shape[1],
        x + w + padding
    )

    y2 = min(
        image_cv.shape[0],
        y + h + padding
    )


    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    cropped = image_cv[
        y1:y2,
        x1:x2
    ]


    # --------------------------------------------------------
    # OpenCV BGR → RGB
    # --------------------------------------------------------

    cropped = cv2.cvtColor(
        cropped,
        cv2.COLOR_BGR2RGB
    )


    print("✓ Bark crop completed")

    return Image.fromarray(
        cropped
    )


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):

    print("\n❌ Image not found:")
    print(IMAGE_PATH)

    exit()


# ============================================================
# START
# ============================================================

print("\n" + "=" * 60)

print(
    "TREE BARK IMAGE PREDICTION"
)

print("=" * 60)

print(
    f"Device: {DEVICE}"
)


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print("✓ Image loaded")


# ============================================================
# BACKGROUND REMOVAL
# ============================================================

background_removed_image = (
    remove_background(image)
)


# ============================================================
# BARK CROPPING
# ============================================================

cropped_image = crop_bark(
    background_removed_image
)


# ============================================================
# SAVE PROCESSED IMAGE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

cropped_image.save(
    "outputs/prediction_cropped.jpg"
)

print(
    "\n✓ Final processed image saved to:"
)

print(
    "  outputs/prediction_cropped.jpg"
)


# ============================================================
# LOAD HYBRID CNN
# ============================================================

print("\nLoading Hybrid CNN...")

model = HybridCNN()


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)


model = model.to(
    DEVICE
)

model.eval()

print(
    "✓ Model loaded successfully"
)


# ============================================================
# PREPARE IMAGE
# ============================================================

input_tensor = transform(
    cropped_image
)


input_tensor = input_tensor.unsqueeze(
    0
)


input_tensor = input_tensor.to(
    DEVICE
)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(
        input_tensor
    )


    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]


# ============================================================
# GET PROBABILITIES
# ============================================================

cinnamon_probability = (
    probabilities[0].item()
    * 100
)

neem_probability = (
    probabilities[1].item()
    * 100
)

none_probability = (
    probabilities[2].item()
    * 100
)


# ============================================================
# FINAL PREDICTION
# ============================================================

class_names = [
    "CINNAMON",
    "NEEM",
    "NONE"
]


predicted_index = torch.argmax(
    probabilities
).item()


prediction = class_names[
    predicted_index
]


confidence = (
    probabilities[
        predicted_index
    ].item()
    * 100
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)

print(
    "PREDICTION RESULT"
)

print("=" * 60)


print(
    f"Cinnamon : "
    f"{cinnamon_probability:.2f}%"
)


print(
    f"Neem     : "
    f"{neem_probability:.2f}%"
)


print(
    f"None     : "
    f"{none_probability:.2f}%"
)


print("-" * 60)


print(
    f"Prediction : "
    f"{prediction}"
)


print(
    f"Confidence : "
    f"{confidence:.2f}%"
)


print("=" * 60)