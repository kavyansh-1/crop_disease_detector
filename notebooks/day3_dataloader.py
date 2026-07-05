import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

#---- Find Datatset ---------
DATA_ROOT=Path("data/raw")
possible=[
    DATA_ROOT/"plantvillage dataset"/"color",
    DATA_ROOT/"color",
    DATA_ROOT/"Plantvillage"
]
DATASET_DIR=next((p for p in possible if p.exists()),None)
if DATASET_DIR is None:
    for p in DATA_ROOT.rglob("*"):
        if p.is_dir() and len(list(p.iterdir())) > 30:
             DATASET_DIR=p
             break
print(f"Dataset: {DATASET_DIR}")

# ---- Build image paths _ labels dataframe ------
records=[]
class_names=sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
class_to_idx={name:i for i,name in enumerate(class_names)}

for class_name in class_names:
    class_dir=DATASET_DIR/class_name
    for img_path in class_dir.rglob("*"):
        if img_path.suffix.lower() in [".jpg",".JPG",".png"]:
            records.append({
                "path":str(img_path),
                "label":class_to_idx[class_name],
                "class_name":class_name,
            })

df=pd.DataFrame(records)
print(f"Total Images : {len(df)}")
print(f"Total classes : {len(class_names)}")

#--- Train / val / test split
train_df , temp_df = train_test_split(
    df,test_size=0.30,stratify=df["label"],random_state=42
)
val_df,test_df=train_test_split(
    temp_df , test_size=0.50 , stratify=temp_df["label"],random_state=42
)

print(f"\nSplit sizes:")
print(f" Train: {len(train_df)} images")
print(f" Test: {len(test_df)} images")
 
#Save splits
os.makedirs("data/splits",exist_ok=True)
train_df.to_csv("data/splits/train.csv",index=False)
val_df.to_csv("data/splits/val.csv",index=False)
test_df.to_csv("data/splits/test.csv",index=False)

#Save class names
with open("data/splits/class_names.json","w") as f:
    json.dump(class_names,f,indent=2)

#--- ImageNet ---
IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]

# --- transforms ---
train_transform=A.Compose([
    A.Resize(256,256),
    A.RandomCrop(224,224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.1),
    A.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform=A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
    ToTensorV2(),
])

# --- Dataset Class ---
class PlantVillageDataset(Dataset):
    def __init__(self,dataframe,transform=None):
        self.df=dataframe.reset_index(drop=True)
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
    
# --- DataLoaders ---
train_dataset=PlantVillageDataset(train_df,transform=train_transform)
val_dataset=PlantVillageDataset(val_df,transform=val_transform)
test_dataset=PlantVillageDataset(test_df,transform=val_transform)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=0,pin_memory=True)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=True,num_workers=0,pin_memory=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=0,pin_memory=True)


print(f"\nDataLoaders ready:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")

# -- Sanity check - visualize one batch ---
print("\nRunning sanity check on one batch")
images,labels=next(iter(train_loader))
print(f"Batch shape: {images.shape}")
print(f"Label shape: {labels.shape}")
print(f"Image dtype: {images.dtype}")
print(f"Min/Max pixel: {images.min():.2f}/{images.max():.2f}")

# --- Denormalize for visualization ----
def denormalize(tensor):
    mean=torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std=torch.tensor(IMAGENET_STD).view(3,1,1)
    return torch.clamp(tensor * std + mean, 0, 1)


os.makedirs("reports/figures",exist_ok=True)
fig,axes=plt.subplots(4,8,figsize=(20,10))
axes=axes.flatten()
for i in range(32):
    img=denormalize(images[i]).permute(1,2,0).numpy()
    axes[i].imshow(img)
    axes[i].set_title(class_names[labels[i]][:20],fontsize=6)
    axes[i].axis("off")

plt.suptitle("One training batch (32 images) — augmented", fontsize=13)
plt.tight_layout()
plt.savefig("reports/figures/sample_batch.png", dpi=150)
plt.show()
print("Saved: reports/figures/sample_batch.png")

print("\n✓ Day 3 complete. DataLoader pipeline working.")

