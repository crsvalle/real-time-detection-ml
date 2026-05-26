import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from scipy.io import loadmat
import json

from dataset import CarDataset
from prepare_data import data

# -------------------------
# PATHS
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
img_dir = BASE_DIR / "data/raw/stanford-cars/cars_train/cars_train"
devkit_dir = BASE_DIR / "data/raw/stanford-cars/car_devkit/devkit"
checkpoint_dir = BASE_DIR / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

# -------------------------
# LOAD CLASS NAMES (brand + model + year)
# e.g. class_names[0] = "AM General Hummer SUV 2000"
# -------------------------
meta = loadmat(devkit_dir / "cars_meta.mat")
class_names = [str(c[0]) for c in meta["class_names"][0]]  # 196 entries, 1-indexed in MATLAB
# Save as JSON for inference use (DetectionCanvas / useVehicleIdentifier)
with open("car_classes.json", "w") as f:
    json.dump(class_names, f, indent=2)
print(f"Saved {len(class_names)} class names to car_classes.json")

# -------------------------
# TRANSFORMS
# ImageNet mean/std required — ResNet50 was pretrained on these
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------
# DATASET
# -------------------------
# We need separate transforms for train/val, so we split indices first
from torch.utils.data import Subset
import numpy as np

full_dataset = CarDataset(data, img_dir, transform=None)  # no transform yet
indices = np.arange(len(full_dataset))
np.random.seed(42)
np.random.shuffle(indices)

split = int(0.8 * len(indices))
train_indices, val_indices = indices[:split], indices[split:]

train_dataset = CarDataset(
    [data[i] for i in train_indices], img_dir, transform=train_transform
)
val_dataset = CarDataset(
    [data[i] for i in val_indices], img_dir, transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

# sanity check
images, labels = next(iter(train_loader))
print("Batch shapes:", images.shape, labels.shape)
print("Label range:", labels.min().item(), "–", labels.max().item())

# -------------------------
# DEVICE
# -------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

# -------------------------
# MODEL
# Fine-tune strategy: train the full network but with a lower LR on the
# backbone and higher LR on the new head.
# -------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 196)
model = model.to(device)

# -------------------------
# LOSS + OPTIMIZER
# Two param groups: lower LR for pretrained backbone, higher for new head
# -------------------------
backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
head_params     = list(model.fc.parameters())

optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params,     "lr": 1e-3},
])

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Cosine annealing — smoothly decays LR over all epochs
num_epochs = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# -------------------------
# METRICS
# -------------------------
def batch_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item() / labels.size(0)

# -------------------------
# TRAINING LOOP
# -------------------------
best_val_acc = 0.0

for epoch in range(num_epochs):
    # --- Train ---
    model.train()
    train_loss = train_acc = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc  += batch_accuracy(outputs, labels)

    scheduler.step()

    # --- Validate ---
    model.eval()
    val_loss = val_acc = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            val_acc  += batch_accuracy(outputs, labels)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc  = train_acc  / len(train_loader)
    avg_val_loss   = val_loss   / len(val_loader)
    avg_val_acc    = val_acc    / len(val_loader)
    current_lr     = scheduler.get_last_lr()[0]

    print(f"\nEpoch {epoch+1:02d}/{num_epochs}  lr={current_lr:.2e}")
    print(f"  Train — loss: {avg_train_loss:.4f}  acc: {avg_train_acc:.4f}")
    print(f"  Val   — loss: {avg_val_loss:.4f}  acc: {avg_val_acc:.4f}")

    # --- Checkpoint (save every epoch + keep best) ---
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": avg_val_acc,
    }
    torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch+1:02d}.pth")

    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(checkpoint, "car_classifier_best.pth")
        print(f"  ✓ New best val acc: {best_val_acc:.4f} — saved car_classifier_best.pth")

print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")