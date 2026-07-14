import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image 
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow
import mlflow.pytorch

# ---- Config ----
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "gpu")
SPLITS_DIR=Path("data/splits")
REPORTS_DIR=Path("reports/figures")
CKPT_DIR=Path("models/heckpoints")
CKPT_DIR.mkdir(parents=True,exist_ok=True)
REPORTS_DIR.mkdir(parents=True,exist_ok=True)

print(f"Device: {DEVICE}")

# ---- Load splits + class names ----
train_df=pd.read_csv(SPLITS_DIR / "train.csv")
val_df=pd.read_csv(SPLITS_DIR / "val.csv")

with open(SPLITS_DIR / "class_names.json") as f:
    class_names = json.load(f)
NUM_CLASSES = len(class_names)
print(f"Classes: {NUM_CLASSES}")

# ---- Dataset ----
MEAN = [0.485,0.456,0.406]
STD = [0.229,0.224,0.225]

train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

class PlantVillageDataset(Dataset):
    def __init__(self,dataframe,transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform=transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        image=np.array(Image.open(row["path"]).convert("RGB"))
        label=int(row["label"])
        if self.transform:
            image=self.transform(image=image)["image"]
        return image,label
    
train_loader = DataLoader(
    PlantVillageDataset(train_df,train_transform),
    batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True
)
val_loader=DataLoader(
    PlantVillageDataset(val_df,val_transform),
    batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True
)

# ---- Model - ResNet18 from Srcatch ----
model=models.resnet18(weights=None)
model.fc=nn.Linear(model.fc.in_features,NUM_CLASSES)
model=model.to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- Loss + optimizer ----
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LR)

# ---- Train + eval functions ----
def train_one_epoch(model,loader,optimizer,criterion,device):
    model.train()
    total_loss , correct , total =0,0,0
    pbar=tqdm(loader,desc='training',leave=False)
    for images,labels in loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss/total , correct/total

def evaluate(model,loader,criterion,value):
    model.eval()
    total_loss,correct,total=0,0,0
    with torch.no_grad():
        for images,labels in loader:
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            outputs=model(images)
            loss=criterion(outputs,labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total

# ---- Training loop with mlflow ----
mlflow.set_experiment("crop_disease_detector")

with mlflow.start_run(run_name="resnet18_scratch_baseline"):
    mlflow.log_params({
        "model" : "resnet18_scratch",
        "epochs" : EPOCHS,
        "lr" : LR,
        "batch_size" : BATCH_SIZE,
        "optimizer" : "adam",
        "pretrained" : False,
    })

    history = {"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
    best_val_acc = 0.0

    for epoch in range(1,EPOCHS+1):
        tr_loss , tr_acc = train_one_epoch(model,train_loader,optimizer,criterion,DEVICE)
        vl_loss , vl_acc = evaluate(model,val_loader,criterion,DEVICE)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        mlflow.log_metrics({
            "train_loss":tr_loss,
            "val_loss":vl_loss,
            "train_acc":tr_acc,
            "val_acc":vl_acc,
        },step=epoch)

        print(f"EPOCH {epoch:02d}/{EPOCHS} |"
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} |"
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}"
              )
        
        # ---- Save Best Checkpoint ----
        if vl_acc > best_val_acc:
            best_val_acc=vl_acc
            torch.save(model.state_dict(),CKPT_DIR/"resnet18_search_best.pt")
            print(f"New Best saved: {best_val_acc:.4f}")

    mlflow.log_metric("best_val_acc",best_val_acc)
    print(f"\nBest Val accuracy: {best_val_acc:.4f}")

# ---- pllot training curves ----
fig , (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))

ax1.plot(history["train_loss"],label="Train Loss")
ax1.plot(history["val_loss"],label="Val Loss")
ax1.set_title("Loss curves - ResNet18 scratch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.legend()

ax2.plot(history["train_acc"],label="Train Loss")
ax2.plot(history["val_acc"],label="Val Loss")
ax2.set_title("Accuracy curves - ResNet18 scratch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()


plt.tight_layout()
plt.savefig(REPORTS_DIR / "day4_baseline_curves.png", dpi=150)
plt.show()
print("Saved: reports/figures/day4_baseline_curves.png")

print("\n✓ Day 4 complete.")
print(f"  Best val accuracy: {best_val_acc*100:.2f}%")
print(f"  Checkpoint: models/checkpoints/resnet18_scratch_best.pt")
print(f"  View MLflow: run 'mlflow ui' then open http://localhost:5000")