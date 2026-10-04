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

CLASS_NAMES = [
    "CINNAMON",
    "NEEM",
    "NONE"
]


# ============================================================
# IMAGE PATH
# ============================================================

IMAGE_PATH = input(
    "Enter image path for Grad-CAM: "
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

        self.dropout = nn.Dropout(
            p=0.4
        )

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
# BACKGROUND REMOVAL + BARK CROPPING
# ============================================================

def preprocess_bark(image):

    print("\nRemoving background...")

    result = remove(image)

    result = result.convert("RGBA")

    alpha = result.getchannel("A")

    bbox = alpha.getbbox()

    if bbox is None:

        print("⚠ Background removal failed.")
        print("Using original image.")

        return image

    left, top, right, bottom = bbox

    width, height = image.size

    bbox_area = (
        (right - left)
        * (bottom - top)
    )

    image_area = width * height

    foreground_ratio = (
        bbox_area / image_area
    )

    if foreground_ratio < 0.02:

        print(
            "⚠ Foreground too small."
        )

        print(
            "Using original image."
        )

        return image

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

    cropped = result.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )

    white_background = Image.new(
        "RGB",
        cropped.size,
        (255, 255, 255)
    )

    white_background.paste(
        cropped,
        mask=cropped.getchannel("A")
    )

    bark_image = white_background

    print("✓ Background removed")
    print("✓ Bark crop completed")

    return bark_image


# ============================================================
# FEATURE LEVEL CONVERSION
# ============================================================

def feature_level(value):

    if value < 0.33:

        return "Low"

    elif value < 0.66:

        return "Moderate"

    else:

        return "High"


# ============================================================
# BAR DISPLAY
# ============================================================

def feature_bar(value):

    value = max(
        0.0,
        min(1.0, value)
    )

    blocks = int(
        round(value * 10)
    )

    return (
        "█" * blocks
        + "░" * (10 - blocks)
    )


# ============================================================
# ANALYZE IMPORTANT REGION
# ============================================================

def analyze_visual_features(
    image,
    heatmap
):

    image_np = np.array(
        image
    )

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

    # --------------------------------------------------------
    # CREATE IMPORTANT REGION MASK
    # --------------------------------------------------------

    threshold = np.percentile(
        heatmap,
        70
    )

    mask = heatmap >= threshold

    # If too few pixels are selected,
    # use a lower threshold.

    if np.sum(mask) < 100:

        threshold = np.percentile(
            heatmap,
            60
        )

        mask = heatmap >= threshold


    # --------------------------------------------------------
    # TEXTURE
    # --------------------------------------------------------

    local_std = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    texture_score = (
        1.0
        - np.exp(
            -local_std / 300.0
        )
    )


    # --------------------------------------------------------
    # SURFACE DETAIL
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    if np.sum(mask) > 0:

        masked_edges = edges[
            mask
        ]

        surface_detail_score = (
            np.mean(
                masked_edges > 0
            )
        )

    else:

        surface_detail_score = 0.0


    # --------------------------------------------------------
    # COLOR VARIATION
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1]

    if np.sum(mask) > 0:

        color_std = np.std(
            saturation[mask]
        )

    else:

        color_std = np.std(
            saturation
        )

    color_variation_score = (
        1.0
        - np.exp(
            -color_std / 40.0
        )
    )


    # --------------------------------------------------------
    # EDGE DENSITY
    # --------------------------------------------------------

    if np.sum(mask) > 0:

        edge_density = np.mean(
            edges[mask] > 0
        )

    else:

        edge_density = np.mean(
            edges > 0
        )

    edge_score = min(
        edge_density * 4.0,
        1.0
    )


    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    if np.sum(mask) > 0:

        brightness_value = np.mean(
            gray[mask]
        )

    else:

        brightness_value = np.mean(
            gray
        )

    brightness_score = (
        brightness_value / 255.0
    )


    # --------------------------------------------------------
    # PATTERN SCALE
    # --------------------------------------------------------

    # Compare fine and blurred image detail.

    blurred = cv2.GaussianBlur(
        gray,
        (15, 15),
        0
    )

    fine_detail = cv2.absdiff(
        gray,
        blurred
    )

    if np.sum(mask) > 0:

        fine_value = np.mean(
            fine_detail[mask]
        )

    else:

        fine_value = np.mean(
            fine_detail
        )

    pattern_scale_score = (
        1.0
        - np.exp(
            -fine_value / 25.0
        )
    )


    return {

        "Texture": min(
            texture_score,
            1.0
        ),

        "Surface detail": min(
            surface_detail_score * 4.0,
            1.0
        ),

        "Color variation": min(
            color_variation_score,
            1.0
        ),

        "Edge density": min(
            edge_score,
            1.0
        ),

        "Pattern scale": min(
            pattern_scale_score,
            1.0
        ),

        "Brightness": min(
            brightness_score,
            1.0
        )
    }


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(
    IMAGE_PATH
):

    print(
        "\n❌ Image not found:"
    )

    print(
        IMAGE_PATH
    )

    exit()


# ============================================================
# LOAD IMAGE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TREE BARK - GRAD-CAM EXPLAINABLE AI"
)

print(
    "=" * 60
)

print(
    f"Device: {DEVICE}"
)

print(
    "\nLoading image..."
)

original_image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    "✓ Image loaded"
)


# ============================================================
# PREPROCESS
# ============================================================

processed_image = preprocess_bark(
    original_image
)


# ============================================================
# SAVE PROCESSED IMAGE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

processed_path = (
    "outputs/gradcam_processed.jpg"
)

processed_image.save(
    processed_path
)

print(
    "\n✓ Processed image saved to:"
)

print(
    f"  {processed_path}"
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
# PREPARE INPUT
# ============================================================

input_tensor = transform(
    processed_image
)

input_tensor = input_tensor.unsqueeze(
    0
)

input_tensor = input_tensor.to(
    DEVICE
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading Hybrid CNN..."
)

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
# GRAD-CAM TARGET LAYER
# ============================================================

target_layer = model.branch_a[12]

print(
    "\nGrad-CAM target layer:"
)

print(
    "Branch A - Final Convolution"
)


# ============================================================
# STORAGE
# ============================================================

activations = None

gradients = None


# ============================================================
# FORWARD HOOK
# ============================================================

def forward_hook(
    module,
    input,
    output
):

    global activations

    activations = output


# ============================================================
# BACKWARD HOOK
# ============================================================

def backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


# ============================================================
# REGISTER HOOKS
# ============================================================

forward_handle = (
    target_layer.register_forward_hook(
        forward_hook
    )
)

backward_handle = (
    target_layer.register_full_backward_hook(
        backward_hook
    )
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nRunning prediction..."
)

outputs = model(
    input_tensor
)

probabilities = torch.softmax(
    outputs,
    dim=1
)[0]


# ============================================================
# GET PREDICTION
# ============================================================

predicted_index = torch.argmax(
    probabilities
).item()

prediction = CLASS_NAMES[
    predicted_index
]

confidence = (
    probabilities[
        predicted_index
    ].item()
    * 100
)


# ============================================================
# DISPLAY PREDICTION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "PREDICTION"
)

print(
    "=" * 60
)

print(
    f"Cinnamon : "
    f"{probabilities[0].item() * 100:.2f}%"
)

print(
    f"Neem     : "
    f"{probabilities[1].item() * 100:.2f}%"
)

print(
    f"None     : "
    f"{probabilities[2].item() * 100:.2f}%"
)

print(
    "-" * 60
)

print(
    f"Prediction : "
    f"{prediction}"
)

print(
    f"Confidence : "
    f"{confidence:.2f}%"
)


# ============================================================
# BACKWARD PASS
# ============================================================

model.zero_grad()

target_score = outputs[
    0,
    predicted_index
]

target_score.backward()


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()

backward_handle.remove()


# ============================================================
# GET ACTIVATIONS
# ============================================================

activations = activations.detach()

gradients = gradients.detach()


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

weights = torch.mean(
    gradients,
    dim=(2, 3),
    keepdim=True
)


# ============================================================
# CREATE GRAD-CAM
# ============================================================

cam = torch.sum(
    weights * activations,
    dim=1
)

cam = torch.relu(
    cam
)

cam = cam[0]

cam = cam.cpu().numpy()


# ============================================================
# NORMALIZE
# ============================================================

cam_min = cam.min()

cam_max = cam.max()

if cam_max - cam_min != 0:

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min
    )

else:

    cam = np.zeros_like(
        cam
    )


# ============================================================
# ORIGINAL PROCESSED IMAGE
# ============================================================

processed_np = np.array(
    processed_image
)

height, width = (
    processed_np.shape[:2]
)


# ============================================================
# RESIZE HEATMAP
# ============================================================

heatmap = cv2.resize(
    cam,
    (width, height)
)


# ============================================================
# ANALYZE VISUAL CHARACTERISTICS
# ============================================================

visual_features = analyze_visual_features(
    processed_image,
    heatmap
)


# ============================================================
# DISPLAY VISUAL CHARACTERISTICS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "VISUAL CHARACTERISTICS OF IMPORTANT REGION"
)

print(
    "=" * 60
)

for name, value in visual_features.items():

    print(
        f"{name:<17}"
        f"{feature_bar(value)}  "
        f"{feature_level(value)}"
    )

print(
    "\nThese characteristics are measured from the "
    "image region highlighted by Grad-CAM."
)

print(
    "They describe visual properties associated with "
    "the model's prediction."
)


# ============================================================
# CREATE COLOR HEATMAP
# ============================================================

heatmap_uint8 = np.uint8(
    255 * heatmap
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)


# ============================================================
# CONVERT IMAGE TO BGR
# ============================================================

processed_bgr = cv2.cvtColor(
    processed_np,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# OVERLAY
# ============================================================

overlay = cv2.addWeighted(
    processed_bgr,
    0.60,
    heatmap_color,
    0.40,
    0
)


# ============================================================
# SAVE GRAD-CAM
# ============================================================

output_path = (
    "outputs/gradcam_new_image.jpg"
)

cv2.imwrite(
    output_path,
    overlay
)


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "GRAD-CAM RESULT"
)

print(
    "=" * 60
)

print(
    f"Prediction : "
    f"{prediction}"
)

print(
    f"Confidence : "
    f"{confidence:.2f}%"
)

print(
    "\nGrad-CAM shows the regions that contributed "
    "most strongly to the prediction."
)

print(
    "\nGrad-CAM output:"
)

print(
    "outputs/gradcam_new_image.jpg"
)

print(
    "=" * 60
)