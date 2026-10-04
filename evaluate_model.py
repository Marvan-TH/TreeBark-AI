import os
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "cropped_data_new"
MODEL_PATH = "models/best_hybrid_cnn.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# TRANSFORMS
# ==========================================

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

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
    os.path.join(DATA_DIR, "test"),
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("=" * 60)
print("HYBRID CNN - MODEL EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {test_dataset.class_to_idx}")


# ==========================================
# HYBRID CNN
# ==========================================

class HybridCNN(nn.Module):

    def __init__(self):

        super(HybridCNN, self).__init__()

        # ======================================
        # BRANCH A - 3x3
        # ======================================

        self.branch_a = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ======================================
        # BRANCH B - 5x5 + 3x3
        # ======================================

        self.branch_b = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ======================================
        # FEATURE FUSION
        # ======================================

        self.fc1 = nn.Linear(
            256 + 128,
            128
        )

        self.dropout = nn.Dropout(0.4)

        self.fc2 = nn.Linear(
            128,
            3
        )


    # ==========================================
    # FORWARD
    # ==========================================

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
            [features_a, features_b],
            dim=1
        )

        x = self.fc1(features)

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading Hybrid CNN...")

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
# PREDICTION
# ==========================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ==========================================
# CLASS NAMES
# ==========================================

class_names = test_dataset.classes


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ==========================================
# PLOT CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Hybrid CNN Confusion Matrix")

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()


# ==========================================
# FINAL ACCURACY
# ==========================================

correct = sum(
    1
    for true, pred
    in zip(
        all_labels,
        all_predictions
    )
    if true == pred
)

total = len(all_labels)

accuracy = (
    correct / total
) * 100


print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

print(
    f"Accuracy: {accuracy:.2f}%"
)

print(
    f"Correct: {correct}/{total}"
)

print(
    "\nConfusion matrix saved to:"
    " outputs/confusion_matrix.png"
)