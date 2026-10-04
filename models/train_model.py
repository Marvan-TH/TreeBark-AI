import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "cropped_data_new"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 0.0005

IMAGE_SIZE = 224

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("TREE BARK CLASSIFICATION - HYBRID CNN TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transforms = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# VALIDATION / TEST TRANSFORMS
# ============================================================

val_test_transforms = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_test_transforms
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=val_test_transforms
)


print("\nClasses:")
print(train_dataset.class_to_idx)

print(
    f"\nTraining images:   {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)

print(
    f"Testing images:    {len(test_dataset)}"
)


# ============================================================
# CHECK CLASS ORDER
# ============================================================

class_names = train_dataset.classes

print("\nClass names:")
for index, name in enumerate(class_names):
    print(f"{index} = {name}")


# ============================================================
# DATA LOADERS
# ============================================================

generator = torch.Generator()

generator.manual_seed(SEED)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# HYBRID CNN
# ============================================================

class HybridCNN(nn.Module):

    def __init__(self):

        super(HybridCNN, self).__init__()


        # ====================================================
        # BRANCH A
        # 3x3 CONVOLUTIONS
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
        # BRANCH B
        # 5x5 + 3x3 CONVOLUTIONS
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
# CREATE MODEL
# ============================================================

print("\nCreating Hybrid CNN...")

model = HybridCNN()

model = model.to(DEVICE)

print("✓ Hybrid CNN created")

print("✓ No pretrained model is being used")


# ============================================================
# CLASS WEIGHTS
# ============================================================

# ImageFolder class order should be:
#
# 0 = Cinnamon
# 1 = Neem
# 2 = None
#
# None has fewer training images, so we give it
# higher importance during training.

print("\nClass weights:")

class_weights = torch.tensor(
    [
        1.0,   # Cinnamon
        1.0,   # Neem
        2.0    # None
    ],
    dtype=torch.float32
).to(DEVICE)

print(
    "Cinnamon = 1.0"
)

print(
    "Neem     = 1.0"
)

print(
    "None     = 2.0"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=4,
    factor=0.5
)


# ============================================================
# BEST MODEL
# ============================================================

best_selection_score = -1.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

none_recall_history = []


# ============================================================
# TRAINING
# ============================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 70)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    ) * 100


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0

    none_correct = 0

    none_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


            # ------------------------------------------------
            # NONE CLASS RECALL
            # ------------------------------------------------

            none_mask = (
                labels == 2
            )

            none_total += (
                none_mask.sum().item()
            )

            none_correct += (
                (
                    predicted == labels
                ) & none_mask
            ).sum().item()


    val_loss = (
        val_loss / val_total
    )

    val_accuracy = (
        val_correct / val_total
    ) * 100


    if none_total > 0:

        none_recall = (
            none_correct /
            none_total
        ) * 100

    else:

        none_recall = 0.0


    # --------------------------------------------------------
    # STORE HISTORY
    # --------------------------------------------------------

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )

    none_recall_history.append(
        none_recall
    )


    # --------------------------------------------------------
    # LEARNING RATE UPDATE
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )


    # --------------------------------------------------------
    # MODEL SELECTION SCORE
    # --------------------------------------------------------

    selection_score = (
        0.7 * val_accuracy
        +
        0.3 * none_recall
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    print(
        f"None Recall: {none_recall:.2f}%"
    )

    print(
        f"Selection Score: {selection_score:.2f}"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if selection_score > best_selection_score:

        best_selection_score = (
            selection_score
        )

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )


        model_path = os.path.join(
            MODEL_DIR,
            "best_hybrid_cnn.pth"
        )


        torch.save(
            model.state_dict(),
            model_path
        )


        print(
            "✓ Best Hybrid CNN saved!"
        )

        print(
            f"✓ Selection Score: "
            f"{selection_score:.2f}"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)


# ============================================================
# FINAL TESTING
# ============================================================

print("\n" + "=" * 70)

print("FINAL TESTING")

print("=" * 70)


model.eval()

correct = 0

total = 0

none_correct = 0

none_total = 0


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

num_classes = 3

confusion_matrix = torch.zeros(
    num_classes,
    num_classes,
    dtype=torch.int64
)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        outputs = model(images)


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


        # None recall

        none_mask = (
            labels == 2
        )

        none_total += (
            none_mask.sum().item()
        )

        none_correct += (
            (
                predicted == labels
            ) & none_mask
        ).sum().item()


        # Confusion matrix

        for true_label, pred_label in zip(
            labels,
            predicted
        ):

            confusion_matrix[
                true_label,
                pred_label
            ] += 1


test_accuracy = (
    correct / total
) * 100


if none_total > 0:

    test_none_recall = (
        none_correct /
        none_total
    ) * 100

else:

    test_none_recall = 0.0


print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f"None Recall: "
    f"{test_none_recall:.2f}%"
)


# ============================================================
# PRINT CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    "                 Predicted"
)

print(
    "              Cinnamon  Neem  None"
)

for i, class_name in enumerate(
    class_names
):

    print(
        f"{class_name:<10} "
        f"{confusion_matrix[i, 0]:>8} "
        f"{confusion_matrix[i, 1]:>6} "
        f"{confusion_matrix[i, 2]:>6}"
    )


# ============================================================
# SAVE TRAINING / VALIDATION CURVES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


epochs = range(
    1,
    NUM_EPOCHS + 1
)


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs,
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


accuracy_path = os.path.join(
    OUTPUT_DIR,
    "training_validation_accuracy.png"
)


plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


loss_path = os.path.join(
    OUTPUT_DIR,
    "training_validation_loss.png"
)


plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# NONE RECALL CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs,
    none_recall_history,
    label="Validation None Recall"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "None Recall (%)"
)

plt.title(
    "Validation None-Class Recall"
)

plt.ylim(
    0,
    105
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


none_recall_path = os.path.join(
    OUTPUT_DIR,
    "none_recall_curve.png"
)


plt.savefig(
    none_recall_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(8, 7)
)


plt.imshow(
    confusion_matrix.numpy()
)


plt.title(
    "Hybrid CNN Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)


plt.xticks(
    range(num_classes),
    class_names
)

plt.yticks(
    range(num_classes),
    class_names
)


for i in range(num_classes):

    for j in range(num_classes):

        plt.text(
            j,
            i,
            confusion_matrix[i, j].item(),
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()


confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("TRAINING COMPLETE")

print("=" * 70)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f"None Recall: "
    f"{test_none_recall:.2f}%"
)

print(
    "Model saved to:"
)

print(
    "models/best_hybrid_cnn.pth"
)

print(
    "\nTraining accuracy curve:"
)

print(
    "outputs/training_validation_accuracy.png"
)

print(
    "Training loss curve:"
)

print(
    "outputs/training_validation_loss.png"
)

print(
    "None recall curve:"
)

print(
    "outputs/none_recall_curve.png"
)

print(
    "Confusion matrix:"
)

print(
    "outputs/confusion_matrix.png"
)

print("=" * 70)