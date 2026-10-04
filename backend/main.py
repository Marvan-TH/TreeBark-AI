import os
import base64
import io

import cv2
import numpy as np

import torch
import torch.nn as nn

from PIL import Image
from rembg import remove
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_hybrid_cnn.pth"
)

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
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="TreeBark AI API",
    description="Hybrid CNN based dried tree bark classification API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


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
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            ),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            ),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            ),


            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            ),


            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # ====================================================
        # BRANCH B - 5x5 + 3x3 CONVOLUTIONS
        # ====================================================

        self.branch_b = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            ),


            nn.Conv2d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            ),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            ),


            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
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

        # Branch A

        features_a = self.branch_a(x)

        features_a = torch.flatten(
            features_a,
            1
        )


        # Branch B

        features_b = self.branch_b(x)

        features_b = torch.flatten(
            features_b,
            1
        )


        # Feature Fusion

        features = torch.cat(
            [
                features_a,
                features_b
            ],
            dim=1
        )


        # Classifier

        x = self.fc1(features)

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("TREE BARK AI - FASTAPI BACKEND")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

print(
    f"Model path: {MODEL_PATH}"
)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
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
    "✓ Hybrid CNN loaded successfully"
)

print(
    "✓ Model is ready"
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
# BACKGROUND REMOVAL + BARK CROPPING
# ============================================================

def preprocess_bark(image):

    result = remove(image)

    result = result.convert(
        "RGBA"
    )

    alpha = result.getchannel(
        "A"
    )

    bbox = alpha.getbbox()


    # --------------------------------------------------------
    # BACKGROUND REMOVAL FAILED
    # --------------------------------------------------------

    if bbox is None:

        return image


    left, top, right, bottom = bbox


    # --------------------------------------------------------
    # ORIGINAL IMAGE SIZE
    # --------------------------------------------------------

    width, height = image.size


    # --------------------------------------------------------
    # FOREGROUND RATIO
    # --------------------------------------------------------

    bbox_area = (
        (right - left)
        *
        (bottom - top)
    )

    image_area = (
        width
        *
        height
    )

    foreground_ratio = (
        bbox_area /
        image_area
    )


    # --------------------------------------------------------
    # FOREGROUND TOO SMALL
    # --------------------------------------------------------

    if foreground_ratio < 0.02:

        return image


    # --------------------------------------------------------
    # PADDING
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
    # CROP
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
    # WHITE BACKGROUND
    # --------------------------------------------------------

    white_background = Image.new(
        "RGB",
        cropped.size,
        (255, 255, 255)
    )


    white_background.paste(
        cropped,
        mask=cropped.getchannel(
            "A"
        )
    )


    return white_background


# ============================================================
# FEATURE LEVEL
# ============================================================

def feature_level(value):

    if value < 0.33:

        return "Low"

    elif value < 0.66:

        return "Moderate"

    return "High"


# ============================================================
# FEATURE BAR
# ============================================================

def feature_bar(value):

    value = max(
        0.0,
        min(
            1.0,
            value
        )
    )

    blocks = int(
        round(
            value * 10
        )
    )

    return (
        "█" * blocks
        +
        "░" * (10 - blocks)
    )


# ============================================================
# VISUAL FEATURE ANALYSIS
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
    # IMPORTANT REGION
    # --------------------------------------------------------

    threshold = np.percentile(
        heatmap,
        70
    )

    mask = (
        heatmap >= threshold
    )


    if np.sum(mask) < 100:

        threshold = np.percentile(
            heatmap,
            60
        )

        mask = (
            heatmap >= threshold
        )


    # --------------------------------------------------------
    # TEXTURE
    # --------------------------------------------------------

    local_std = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


    texture_score = (
        1.0
        -
        np.exp(
            -local_std / 300.0
        )
    )


    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )


    # --------------------------------------------------------
    # SURFACE DETAIL
    # --------------------------------------------------------

    if np.sum(mask) > 0:

        masked_edges = edges[
            mask
        ]

        surface_detail_score = np.mean(
            masked_edges > 0
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
        -
        np.exp(
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
        brightness_value /
        255.0
    )


    # --------------------------------------------------------
    # PATTERN SCALE
    # --------------------------------------------------------

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
        -
        np.exp(
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
# GRAD-CAM
# ============================================================

def generate_gradcam(
    input_tensor
):

    activations = None
    gradients = None


    # --------------------------------------------------------
    # TARGET LAYER
    # --------------------------------------------------------

    target_layer = (
        model.branch_a[12]
    )


    # --------------------------------------------------------
    # FORWARD HOOK
    # --------------------------------------------------------

    def forward_hook(
        module,
        input,
        output
    ):

        nonlocal activations

        activations = output


    # --------------------------------------------------------
    # BACKWARD HOOK
    # --------------------------------------------------------

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        nonlocal gradients

        gradients = grad_output[0]


    # --------------------------------------------------------
    # REGISTER HOOKS
    # --------------------------------------------------------

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


    try:

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        model.zero_grad()

        outputs = model(
            input_tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()


        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        target_score = outputs[
            0,
            predicted_index
        ]

        target_score.backward()


        # ----------------------------------------------------
        # ACTIVATIONS / GRADIENTS
        # ----------------------------------------------------

        activations = (
            activations.detach()
        )

        gradients = (
            gradients.detach()
        )


        # ----------------------------------------------------
        # WEIGHTS
        # ----------------------------------------------------

        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )


        # ----------------------------------------------------
        # CAM
        # ----------------------------------------------------

        cam = torch.sum(
            weights * activations,
            dim=1
        )


        cam = torch.relu(
            cam
        )


        cam = cam[0].cpu().numpy()


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        cam_min = cam.min()

        cam_max = cam.max()


        if (
            cam_max - cam_min
            != 0
        ):

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam
            )


        return (
            probabilities[0].detach().cpu().numpy(),
            cam
        )


    finally:

        forward_handle.remove()

        backward_handle.remove()


# ============================================================
# CONVERT IMAGE TO BASE64
# ============================================================

def image_to_base64(
    image
):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "TreeBark AI API is running",
        "model": "Hybrid CNN",
        "device": str(DEVICE),
        "classes": CLASS_NAMES
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "model_loaded": True,
        "device": str(DEVICE)
    }


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # READ UPLOADED IMAGE
        # ----------------------------------------------------

        contents = await file.read()


        if not contents:

            return {
                "success": False,
                "error": "Uploaded file is empty."
            }


        original_image = Image.open(
            io.BytesIO(contents)
        ).convert(
            "RGB"
        )


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed_image = preprocess_bark(
            original_image
        )


        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        input_tensor = transform(
            processed_image
        )


        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )


        # ----------------------------------------------------
        # NORMAL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]


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


        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        gradcam_probabilities, cam = (
            generate_gradcam(
                input_tensor
            )
        )


        # ----------------------------------------------------
        # RESIZE CAM
        # ----------------------------------------------------

        processed_np = np.array(
            processed_image
        )


        height, width = (
            processed_np.shape[:2]
        )


        heatmap = cv2.resize(
            cam,
            (width, height)
        )


        # ----------------------------------------------------
        # VISUAL FEATURES
        # ----------------------------------------------------

        visual_features = (
            analyze_visual_features(
                processed_image,
                heatmap
            )
        )


        # ----------------------------------------------------
        # FEATURE RESPONSE
        # ----------------------------------------------------

        feature_response = {}


        for name, value in visual_features.items():

            feature_response[name] = {

                "score": round(
                    float(value),
                    4
                ),

                "level": feature_level(
                    value
                ),

                "bar": feature_bar(
                    value
                )
            }


        # ----------------------------------------------------
        # GRAD-CAM HEATMAP
        # ----------------------------------------------------

        heatmap_uint8 = np.uint8(
            255 * heatmap
        )


        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )


        # ----------------------------------------------------
        # RGB → BGR
        # ----------------------------------------------------

        processed_bgr = cv2.cvtColor(
            processed_np,
            cv2.COLOR_RGB2BGR
        )


        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        overlay = cv2.addWeighted(
            processed_bgr,
            0.60,
            heatmap_color,
            0.40,
            0
        )


        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        overlay_rgb = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )


        gradcam_image = Image.fromarray(
            overlay_rgb
        )


        # ----------------------------------------------------
        # BASE64 GRAD-CAM
        # ----------------------------------------------------

        gradcam_base64 = image_to_base64(
            gradcam_image
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probability_response = {

            "CINNAMON": round(
                float(
                    probabilities[0].item()
                    * 100
                ),
                2
            ),

            "NEEM": round(
                float(
                    probabilities[1].item()
                    * 100
                ),
                2
            ),

            "NONE": round(
                float(
                    probabilities[2].item()
                    * 100
                ),
                2
            )
        }


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "success": True,

            "prediction": prediction,

            "confidence": round(
                confidence,
                2
            ),

            "probabilities":
                probability_response,

            "gradcam":
                gradcam_base64,

            "features":
                feature_response
        }


    except Exception as e:

        print(
            "\n❌ Prediction error:"
        )

        print(
            str(e)
        )


        return {

            "success": False,

            "error": str(e)
        }


# ============================================================
# RUN INFORMATION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print(
        "\nStarting TreeBark AI API..."
    )

    print(
        "API: http://127.0.0.1:8000"
    )

    print(
        "Docs: http://127.0.0.1:8000/docs"
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )