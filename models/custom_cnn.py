import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "cropped_prepared_data"
MODEL_DIR = "models"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 0.0005

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("TREE BARK CLASSIFICATION - CUSTOM CNN")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================
# DATA AUGMENTATION
# ==========================================

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

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


# ==========================================
# VALIDATION / TEST
# ==========================================

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


# ==========================================
# LOAD DATASETS
# ==========================================

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
    f"\nTraining images:   "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(val_dataset)}"
)

print(
    f"Testing images:    "
    f"{len(test_dataset)}"
)


# ==========================================
# DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
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


# ==========================================
# CUSTOM CNN ARCHITECTURE - VERSION 2
# ==========================================

class TreeBarkCNN(nn.Module):

    def __init__(self):

        super(TreeBarkCNN, self).__init__()

        # --------------------------------------
        # CONVOLUTIONAL BLOCK 1
        # --------------------------------------

        self.conv1 = nn.Conv2d(
            3, 32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # --------------------------------------
        # CONVOLUTIONAL BLOCK 2
        # --------------------------------------

        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # --------------------------------------
        # CONVOLUTIONAL BLOCK 3
        # --------------------------------------

        self.conv3 = nn.Conv2d(
            64, 128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # --------------------------------------
        # CONVOLUTIONAL BLOCK 4
        # --------------------------------------

        self.conv4 = nn.Conv2d(
            128, 256,
            kernel_size=3,
            padding=1
        )

        self.bn4 = nn.BatchNorm2d(256)

        self.pool4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # --------------------------------------
        # GLOBAL AVERAGE POOLING
        # --------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        # --------------------------------------
        # CLASSIFIER
        # --------------------------------------

        self.fc1 = nn.Linear(
            256,
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

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)


        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)


        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)


        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool4(x)


        # Global Average Pooling
        x = self.avgpool(x)


        # Classifier
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

# ==========================================
# CREATE MODEL
# ==========================================

print("\nCreating custom CNN...")

model = TreeBarkCNN()

model = model.to(DEVICE)

print(model)


# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# LEARNING RATE SCHEDULER
# ==========================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=4,
    factor=0.5
)


# ==========================================
# TRAINING
# ==========================================

best_val_loss = float("inf")

best_model_weights = copy.deepcopy(
    model.state_dict()
)


for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print("=" * 60)


    # --------------------------------------
    # TRAINING
    # --------------------------------------

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


    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


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


    val_loss = (
        val_loss / val_total
    )

    val_accuracy = (
        val_correct / val_total
    ) * 100


    scheduler.step(val_loss)


    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


        model_path = os.path.join(
            MODEL_DIR,
            "best_custom_cnn.pth"
        )


        torch.save(
            model.state_dict(),
            model_path
        )


        print(
            "✓ Best custom CNN saved!"
        )


# ==========================================
# LOAD BEST MODEL
# ==========================================

model.load_state_dict(
    best_model_weights
)


# ==========================================
# FINAL TESTING
# ==========================================

print("\n" + "=" * 60)

print("FINAL TESTING")

print("=" * 60)


model.eval()

correct = 0

total = 0


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


test_accuracy = (
    correct / total
) * 100


print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f"Correct: "
    f"{correct}/{total}"
)


print("\nCustom CNN training complete!")

print(
    f"Model saved to: "
    f"{MODEL_DIR}/best_custom_cnn.pth"
)