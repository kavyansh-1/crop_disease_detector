import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- CONFIG ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS_DIR = Path("data/splits")
CKPT_DIR   = Path("models/checkpoints")
FIGS_DIR   = Path("reports/figures/day10")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
test_df= pd.read_csv(SPLITS_DIR / "test.csv")
with open(SPLITS_DIR/'class_names.json') as f:
    class_name=json.load(f)
NUM_CLASSES=len(class_name)
print(f"Test images: {len(test_df)}")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


class PlantVillageDataset(Dataset):
    def __init__(self,dataframe,transform=None):
        self.df=dataframe.reset_index(drop=True)
        self.transform=transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        image=np.array(Image.open(row['path']).convert('RGB'))
        label=int(row['label'])
        if self.transform:
            image=self.transform(image=image)['image']
        return image,label

test_loader = DataLoader(
    PlantVillageDataset(test_df,test_transform),
    batch_size=64,shuffle=False,num_workers=0
)

# ---- Load best model ----
model = models.resnet18(weights=None)
model.fc=nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(512,NUM_CLASSES)
)
model.load_state_dict(torch.load(
    CKPT_DIR/'resnet18_pretrained_best.pt',
    map_location=DEVICE
))
model=model.to(DEVICE)
model.eval()
print("Model Loaded.")

# ---- Test set accuracy (never seen before) ----
correct,total=0,0
all_preds,all_labels = [],[]

with torch.no_grad():
    for images,labels in tqdm(test_loader,desc="Training"):
        images=images.to(DEVICE)
        outputs=model(images)
        preds=outputs.argmax(1)
        correct += (preds.cpu() == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

test_acc = correct/total
print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}%")

# ---- Inference speed becnchmark ----
print("\nBenchmarking inference speed...")
dummy = torch.randn(1,3,224,224).to(DEVICE)

# Warm up
for _ in range(10):
    with torch.no_grad():
        _=model(dummy)

# Benchmarks
times=[]
for _ in range(100):
    start=time.perf_counter()
    with torch.no_grad():
        _=model(dummy)
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        times.append((time.perf_counter()-start)*1000)

avg_ms=np.mean(times)
print(f"Inference speed (GPU): {avg_ms:.2f}ms per image")
print(f"Throughput: {1000/avg_ms:.0f} images/sec")

#CPU inference 
model_cpu=model.cpu()
dummy_cpu=torch.randn(1,3,224,224)
times_cpu=[]
for _ in range(50):
    start=time.perf_counter()
    with torch.no_grad():
        _=model_cpu(dummy_cpu)
    times_cpu.append((time.perf_counter()-start)*1000)

avg_cpu_ms = np.mean(times_cpu)
print(f"Inference speed (CPU): {avg_cpu_ms:.2f}ms per image")
model = model.to(DEVICE)

# Model size
import os
ckpt_size = os.path.getsize(CKPT_DIR / "resnet18_pretrained_best.pt") / 1e6
print(f"\nCheckpoint size: {ckpt_size:.1f} MB")

# ---- 4 confidence distribution ----
all_confs=[]
with torch.no_grad():
    for images,labels in tqdm(test_loader,desc="Confidence"):
        images=images.to(DEVICE)
        probs=torch.softmax(model(images),1)
        confs=probs.max(dim=1).values
        all_confs.extend(confs.cpu().numpy())

all_confs = np.array(all_confs)
print(f"\nConfidence distribution:")
print(f"  Mean:    {all_confs.mean()*100:.2f}%")
print(f"  Median:  {np.median(all_confs)*100:.2f}%")
print(f"  >90%:    {(all_confs > 0.9).mean()*100:.1f}% of predictions")
print(f"  >80%:    {(all_confs > 0.8).mean()*100:.1f}% of predictions")
print(f"  <60%:    {(all_confs < 0.6).mean()*100:.1f}% of predictions (uncertain)")

fig,ax=plt.subplots(figsize=(10,4))
ax.hist(all_confs*100,bins=50,color="#2ecc71",edgecolor="white",linewidth=0.5)
ax.set_xlabel("Confidence (%)")
ax.set_ylabel("Count")
ax.set_title("Model Confidence distribution on test set")
ax.axvline(x=60,color="red",linestyle="--",label="60 % threshold (show warning below this)")
ax.axvline(x=90,color="blue",linestyle="--",label="90 % threshold (high confidence)")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / "confidence_distribution.png", dpi=150)
plt.show()
print("Saved: confidence_distribution.png")

# Phase 1 summary 

print(f"""
╔══════════════════════════════════════════════════╗
║           PHASE 1 COMPLETE — SUMMARY             ║
╠══════════════════════════════════════════════════╣
║  Model:        ResNet18 pretrained (ImageNet)    ║
║  Val accuracy: 99.93%                            ║
║  Test accuracy:{test_acc*100:>7.2f}%             ║
║  Macro F1:     99.15%                            ║
║  GPU inference:{avg_ms:>7.2f}ms                  ║
║  CPU inference:{avg_cpu_ms:>7.2f}m               ║
║  Model size:  {ckpt_size:>7.1f} MB               ║
║  Misclassified:  55 / 8146 val images            ║
╠══════════════════════════════════════════════════╣
║  VERDICT: Ready for Phase 2 (Deployment)         ║
╚══════════════════════════════════════════════════╝
""")