import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow

# ── Config ─────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 10
LR_HEAD     = 0.001   # higher lr for new classifier head
LR_BACKBONE = 0.0001  # lower lr for pretrained backbone
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS_DIR  = Path("data/splits")
CKPT_DIR    = Path("models/checkpoints")
FIGS_DIR    = Path("reports/figures")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── Load splits ────────────────────────────────────────────────────────
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df   = pd.read_csv(SPLITS_DIR / "val.csv")

with open(SPLITS_DIR / "class_names.json") as f:
    class_names = json.load(f)
NUM_CLASSES = len(class_names)
print(f"Classes: {NUM_CLASSES}")

# ── Transforms ─────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.1),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ── Dataset ────────────────────────────────────────────────────────────
class PlantVillageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["path"]).convert("RGB"))
        label = int(row["label"])
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

train_loader = DataLoader(
    PlantVillageDataset(train_df, train_transform),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    PlantVillageDataset(val_df, val_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)

# ── Model — ResNet18 WITH pretrained ImageNet weights ──────────────────
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze entire backbone first
for param in model.parameters():
    param.requires_grad = False

# Replace head — this is the only part that trains in phase 1
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

# Two param groups — different learning rates
backbone_params   = [p for n, p in model.named_parameters()
                     if "fc" not in n and p.requires_grad]
head_params       = list(model.fc.parameters())

print(f"Trainable params (head only): {sum(p.numel() for p in head_params):,}")

optimizer = torch.optim.Adam([
    {"params": head_params, "lr": LR_HEAD},
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Train / eval ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total

# ── Phase 1: Train head only (5 epochs) ───────────────────────────────
print("\n── Phase 1: Head only (backbone frozen) ──")
mlflow.set_experiment("crop_disease_detector")

history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
best_val_acc = 0.0

with mlflow.start_run(run_name="resnet18_pretrained_transfer"):
    mlflow.log_params({
        "model": "resnet18_pretrained",
        "strategy": "freeze_then_unfreeze",
        "epochs": EPOCHS,
        "lr_head": LR_HEAD,
        "lr_backbone": LR_BACKBONE,
        "label_smoothing": 0.1,
        "scheduler": "cosine",
    })

    # Phase 1 — head only
    for epoch in range(1, 6):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        mlflow.log_metrics({
            "train_loss": tr_loss, "val_loss": vl_loss,
            "train_acc": tr_acc,   "val_acc": vl_acc,
        }, step=epoch)

        print(f"[Phase1] Epoch {epoch:02d}/05 | "
              f"Train Acc: {tr_acc*100:.2f}% | Val Acc: {vl_acc*100:.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CKPT_DIR / "resnet18_pretrained_best.pt")
            print(f"  ✓ Best saved: {best_val_acc*100:.2f}%")

    # Phase 2 — unfreeze last 2 blocks + retrain with lower lr
    print("\n── Phase 2: Unfreeze last 2 blocks ──")
    for name, param in model.named_parameters():
        if "layer4" in name or "layer3" in name or "fc" in name:
            param.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": [p for n,p in model.named_parameters()
                    if ("layer3" in n or "layer4" in n) and p.requires_grad],
         "lr": LR_BACKBONE},
        {"params": list(model.fc.parameters()), "lr": LR_HEAD},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params now: {trainable:,}")

    for epoch in range(6, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        mlflow.log_metrics({
            "train_loss": tr_loss, "val_loss": vl_loss,
            "train_acc": tr_acc,   "val_acc": vl_acc,
        }, step=epoch)

        print(f"[Phase2] Epoch {epoch:02d}/10 | "
              f"Train Acc: {tr_acc*100:.2f}% | Val Acc: {vl_acc*100:.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CKPT_DIR / "resnet18_pretrained_best.pt")
            print(f"  ✓ Best saved: {best_val_acc*100:.2f}%")

    mlflow.log_metric("best_val_acc", best_val_acc)
    print(f"\n✓ Training complete. Best val acc: {best_val_acc*100:.2f}%")

# ── Plot ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"],   label="Val")
ax1.axvline(x=4.5, color="gray", linestyle="--", label="Unfreeze point")
ax1.set_title("Loss — ResNet18 pretrained")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot([a*100 for a in history["train_acc"]], label="Train")
ax2.plot([a*100 for a in history["val_acc"]],   label="Val")
ax2.axvline(x=4.5, color="gray", linestyle="--", label="Unfreeze point")
ax2.set_title("Accuracy — ResNet18 pretrained")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.legend()

plt.tight_layout()
plt.savefig(FIGS_DIR / "day5_transfer_curves.png", dpi=150)
plt.show()

# ---- Comparison summary ----
print("\n --- Comparison ---")
print(f"Day 4 scratch baseline:     95.73%")
print(f"Day 5 pretrained transfer:  {best_val_acc*100:.2f}%")
print(f"Delta: {(best_val_acc - 0.9573)*100:+.2f}%")
