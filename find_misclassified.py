import os
import shutil
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "cropped_data_new"
MODEL_PATH = "models/best_hybrid_cnn.pth"

OUTPUT_DIR = "outputs/misclassified"

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================
# TRANSFORMS
# ==========================================

test_transforms = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    os.path.join(
        DATA_DIR,
        "test"
    ),
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes


print("=" * 60)
print("FINDING MISCLASSIFIED TEST IMAGES")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {class_names}")


# ==========================================
# HYBRID CNN
# ==========================================

class HybridCNN(nn.Module):

    def __init__(self):

        super(HybridCNN, self).__init__()


        # ======================================
        # BRANCH A
        # ======================================

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

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # ======================================
        # BRANCH B
        # ======================================

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

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # ======================================
        # FEATURE FUSION
        # ======================================

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


    # ==========================================
    # FORWARD PASS
    # ==========================================

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


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")

model = HybridCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("✓ Model loaded successfully")


# ==========================================
# FIND MISCLASSIFICATIONS
# ==========================================

misclassified_count = 0
correct_count = 0

# IMPORTANT:
# This tracks the actual position of the image
# inside test_dataset.samples.
global_index = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        for i in range(len(labels)):

            actual_index = labels[i].item()

            predicted_index = predictions[i].item()

            confidence = probabilities[
                i,
                predicted_index
            ].item() * 100


            # ==================================
            # GET CORRECT IMAGE PATH
            # ==================================

            original_path, _ = (
                test_dataset.samples[
                    global_index
                ]
            )

            global_index += 1


            # ==================================
            # CORRECT PREDICTION
            # ==================================

            if actual_index == predicted_index:

                correct_count += 1

                continue


            # ==================================
            # MISCLASSIFIED
            # ==================================

            misclassified_count += 1

            actual_class = class_names[
                actual_index
            ]

            predicted_class = class_names[
                predicted_index
            ]


            print("\nMISCLASSIFIED:")

            print(
                f"Actual     : "
                f"{actual_class}"
            )

            print(
                f"Predicted  : "
                f"{predicted_class}"
            )

            print(
                f"Confidence : "
                f"{confidence:.2f}%"
            )

            print(
                f"Image      : "
                f"{os.path.basename(original_path)}"
            )


            # ==================================
            # CREATE CLASS FOLDER
            # ==================================

            folder_name = (
                f"{actual_class}_as_{predicted_class}"
            )

            destination_folder = os.path.join(
                OUTPUT_DIR,
                folder_name
            )

            os.makedirs(
                destination_folder,
                exist_ok=True
            )


            # ==================================
            # COPY IMAGE
            # ==================================

            filename = os.path.basename(
                original_path
            )

            destination_path = os.path.join(
                destination_folder,
                filename
            )

            shutil.copy2(
                original_path,
                destination_path
            )


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("MISCLASSIFICATION ANALYSIS COMPLETE")
print("=" * 60)

print(
    f"Correct images       : "
    f"{correct_count}"
)

print(
    f"Misclassified images : "
    f"{misclassified_count}"
)

print(
    f"Total images         : "
    f"{len(test_dataset)}"
)


print(
    f"\nSaved to: "
    f"{OUTPUT_DIR}"
)


# ==========================================
# FOLDER SUMMARY
# ==========================================

print("\nFolders created:")

for folder in sorted(
    os.listdir(OUTPUT_DIR)
):

    folder_path = os.path.join(
        OUTPUT_DIR,
        folder
    )

    if os.path.isdir(folder_path):

        count = len(
            [
                f
                for f in os.listdir(
                    folder_path
                )
                if os.path.isfile(
                    os.path.join(
                        folder_path,
                        f
                    )
                )
            ]
        )

        print(
            f"  {folder}: "
            f"{count} images"
        )