import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================
# IMPORT YOUR CUSTOM CNN
# ==========================================

from models.custom_cnn import TreeBarkCNN


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "processed_prepared_data"
MODEL_PATH = "models/best_custom_cnn.pth"

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


# ==========================================
# LOAD CUSTOM CNN
# ==========================================

model = TreeBarkCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# ==========================================
# PREDICTIONS
# ==========================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
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


print("\n" + "=" * 60)
print("CUSTOM CNN EVALUATION")
print("=" * 60)

print("\nClasses:")
print(class_names)


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
# DISPLAY CONFUSION MATRIX
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    values_format="d"
)

plt.title(
    "Confusion Matrix - Custom CNN"
)

plt.tight_layout()

plt.savefig(
    "outputs/custom_cnn_confusion_matrix.png",
    dpi=300
)

plt.show()


print(
    "\nConfusion matrix saved to:"
    "\noutputs/custom_cnn_confusion_matrix.png"
)