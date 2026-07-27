import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image   
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow

# ---- CONFIG ----
BATCH_SIZE=64
EPOCHS=10
LR_HEAD=0.001
LR_BACKBONE=0.0001
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS_DIR=Path("data/splits")
CKPT_DIR=Path("models/checkpoints")
FIGS_DIR=Path("reports/figures/day7")
CKPT_DIR.mkdir(parents=True,exist_ok=True)
FIGS_DIR.mkdir(parents=True,exist_ok=True)

print(f"DEVICE: {DEVICE}")

# ---- Load Data ----
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df   = pd.read_csv(SPLITS_DIR / "val.csv")
with open(SPLITS_DIR / "class_names.json") as f:
    class_names = json.load(f)
NUM_CLASSES = len(class_names)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    A.Resize(260, 260),
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

# ---- Model - EfficientNet-B2 via timm ----
model=timm.create_model(
    "efficientnet_b2",
    pretrained=True,
    num_classes=NUM_CLASSES,
    drop_rate=0.3
)
model=model.to(DEVICE)

# Model size
param_count=sum(p.numel() for p in model.parameters())
print(f"EfficientNet-B2 parameters: {param_count:,}")

# Freeze all except classifier head first
for name,param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad,model.parameters()),
    lr=LR_HEAD,weight_decay=1e-4
)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)

# ---- Train / Eval ----
def train_one_epoch(model,loader,optimizer,criterion,device):
    model.train()
    total_loss,correct,total=0,0,0
    pbar=tqdm(loader,desc="Training",leave=False)
    for images,labels in pbar:
        images,labels=images.to(DEVICE),labels.to(DEVICE)
        optimizer.zero_grad()
        output=model(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*images.size(0)
        correct += (output.argmax(1) == labels).sum().item() 
        total += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss/total , correct/total

def evaluate(model,loader,criterion,device):
    model.eval()
    total_loss,correct,total=0,0,0
    with torch.no_grad():
        for images,labels in tqdm(loader,desc="Evaluating",leave=False):
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            outputs=model(images)
            loss=criterion(outputs,labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        return total_loss/total , correct/total
    
# ---- Training ----\
mlflow.set_experiment("crop_disease_detector")
history={"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
best_val_acc = 0.0

with mlflow.start_run(run_name="efficientnet_b2_pretrained"):
    mlflow.log_params({
        "model": "efficientb2",
        "pretrained": True,
        "epochs": EPOCHS,
        "lr_head": LR_HEAD,
        "lr_backbone":LR_BACKBONE,
    })

    # Phase 1 - Head only
    print("\n ---- Head Only ----")
    for epoch in range(1,6):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        mlflow.log_metrics({
            "train_loss": tr_loss, "val_loss": vl_loss,
            "train_acc":  tr_acc,  "val_acc":  vl_acc,
        }, step=epoch)

        print(f"[Phase1] Epoch {epoch:02d}/05 | "
              f"Train: {tr_acc*100:.2f}% | Val: {vl_acc*100:.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CKPT_DIR / "efficientnet_b2_best.pt")
            print(f"  ✓ Best: {best_val_acc*100:.2f}%")

    # Phase 2 - unfreeze all
    print("\n ---- Phase 2 : Full fine-tune ---- ")
    for param in model.parameters():
        param.requires_grad=True

    optimizer = torch.optim.Adam([
        {"params":[p for n,p in model.named_parameters()
                   if "classifier" not in n],"lr":LR_BACKBONE},
        {"params":model.classifier.parameters(),"lr":LR_HEAD},
    ],weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

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
            "train_acc":  tr_acc,  "val_acc":  vl_acc,
        }, step=epoch)

        print(f"[Phase2] Epoch {epoch:02d}/10 | "
              f"Train: {tr_acc*100:.2f}% | Val: {vl_acc*100:.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CKPT_DIR / "efficientnet_b2_best.pt")
            print(f"  ✓ Best: {best_val_acc*100:.2f}%")

    mlflow.log_metric("best_val_acc", best_val_acc)

# ---- Model size comaprision ----
import os
resnet_size=os.path.getsize(CKPT_DIR/"resnet18_pretrained_best.pt")/1e6
effnet_size=os.path.getsize(CKPT_DIR/"efficientnet_b2_best.pt")/1e6

print(f"\n ---- Model Comparison ----")
print(f"ResNet18 pretrained:  99.93% | {resnet_size:.1f} MB")
print(f"EfficientNet-B2:      {best_val_acc*100:.2f}% | {effnet_size:.1f} MB")

# ---- Plot ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
ax1.plot(history["train_loss"],label="Train")
ax1.plot(history["val_loss"],label="Val")
ax1.axvline(x=4.5,color="gray",linestyle='--',label="Unfreeze")
ax1.set_title("Loss-EfficientNet-B2")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot([a*100 for a in history["train_acc"]],label="Train")
ax2.plot([a*100 for a in history["val_acc"]],label="Val")
ax2.axvline(x=4.5,color="gray",linestyle="--",label="Unfreeze")
ax2.set_title("Accuracy-EfficientNet-B2")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

plt.tight_layout()
plt.savefig(FIGS_DIR/"day7_efficient_curves.png",dpi=150)
plt.show()

print(f"\n✓ Day 7 complete. Best val acc: {best_val_acc*100:.2f}%")