import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    confusion_matrix,classification_report,
    ConfusionMatrixDisplay,f1_score
)

# ---- CONFIG ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS_DIR=Path("data/splits")
CKTP_DIR=Path("models/checkpoints")
FIGS_DIR=Path("reports/figure/day6")
FIGS_DIR.mkdir(parents=True,exist_ok=True)
print(f"DEVICE: {DEVICE}")

# ---- Load Data ----
val_df=pd.read_csv(SPLITS_DIR/"val.csv")
with open(SPLITS_DIR/"class_names.json") as f:
    class_name=json.load(f)
NUM_CLASSES=len(class_name)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

val_transform = A.Compose([
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
        image=np.array(Image.open(row["path"]).convert("RGB"))
        label=int(row['label'])
        if self.transform:
            image=self.transform(image=image)["image"]
        return image,label,row['path']


# ---- Custom collate to handle path strings ----
def collate_fn(batch):
    images=torch.stack([b[0] for b in batch])   
    labels=torch.tensor([b[1] for b in batch])
    paths=[b[2] for b in batch]
    return images,labels,paths

val_loader=DataLoader(
    PlantVillageDataset(val_df,val_transform),
    batch_size=64,shuffle=False,num_workers=0,collate_fn=collate_fn
)

# ---- Load Best Model ----
model = models.resnet18(weights=None)
model.fc=nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(512,NUM_CLASSES)
)
model.load_state_dict(torch.load(
    CKTP_DIR/"resnet18_pretrained_best.pt",
    map_location=DEVICE
))
model=model.to(DEVICE)
model.eval()
print("Model Loaded")

# ---- Run Inference on full val set ----
all_preds=[]
all_labels=[]
all_probs=[]
all_paths=[]

with torch.no_grad():
    for images,labels,paths in tqdm(val_loader,desc="Evaluating"):
        images=images.to(DEVICE)
        outputs=model(images)
        probs=torch.softmax(outputs,dim=1)
        preds=outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_paths.extend(paths)

all_preds=np.array(all_preds)
all_labels=np.array(all_labels)
all_probs=np.array(all_probs)

# ---- Overall Accuracy ----
accuracy = (all_preds ==  all_labels).mean()
print(f"\n Val Accuracy: {accuracy*100:.2f}%")

# ---- Per-class f1 score ----
f1_scores=f1_score(all_labels,all_preds,average=None)
f1_df=pd.DataFrame({
    "class": class_name,
    "f1":  f1_scores,
}).sort_values("f1",ascending=False)

print(f"\nBottom 5 classes by F1:")
print(f1_df.head(5).to_string(index=False))
print(f"\nTop 5 classes by f1")
print(f1_df.tail(5).to_string(index=False))

# ---- Plot per-class F1 ----
fig,ax = plt.subplots(figsize=(14,10))
colors=["#e74c3c" if f< 0.95 else "#2ecc71" for f in f1_df["f1"]]
ax.barh(f1_df["class"],f1_df["f1"],color=colors)
ax.set_title("Per-class F1 score (red=below 0.95)")
ax.set_xlabel("F1 Score")
ax.axvline(x=0.95,color="black",linestyle="--",alpha=0.5)
plt.tight_layout()
plt.savefig(FIGS_DIR/"per_class_f1.png",dpi=150)
plt.show()
print("Saved: per_class_f1.png")

# ---- Confusion Marix ----
cm=confusion_matrix(all_labels,all_preds)


short_names=[c.replace("___","\n").replace("_"," ")[:25] for c in class_name]

fig,ax=plt.subplots(figsize=(20,18))
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=short_names)
disp.plot(ax=ax,xticks_rotation=90,colorbar=False,cmap="Blues")
ax.set_title("Confusion Matrix - ResNet18 pretrained (val set)")
plt.tight_layout()
plt.savefig(FIGS_DIR/"confusion_matrix.png",dpi=120)
print("Saved: Confusion_matrix.png")

# ---- Find most confused pairs -----
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag,0)

confused_pairs=[]
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i!=j and cm_no_diag[i,j] > 0:
            confused_pairs.append({
                "true":  class_name[i],
                "predicted": class_name[j],
                "count": cm_no_diag[i,j],
            })

confused_df=pd.DataFrame(confused_pairs).sort_values("count",ascending=False)
print(f"\nTop 10 most confused pairs")
print(confused_df.head(10).to_string(index=False))
confused_df.to_csv(FIGS_DIR / "confused_pairs.csv", index=False)

# ---- Visualize Misclassified images ----
wrong_mask = all_preds != all_labels
wrong_idx = np.where(wrong_mask)[0]
wrong_confs=all_probs[wrong_idx,all_preds[wrong_idx]]

# Sort by highest confidence wrong predictions (model was most sure but wrong)
sorted_wrong=wrong_idx[np.argsort(wrong_confs)[::-1]]
top_wrong=sorted_wrong[:min(16,len(sorted_wrong))]

if len(top_wrong) > 0:
    cols=4
    rows=(len(top_wrong)+cols-1) // cols
    fig,axes=plt.subplots(rows,cols,figsize=(16,rows*4))
    axes=axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i,idx in enumerate(top_wrong):
        img=Image.open(all_paths[idx]).convert("RGB")
        true_name = class_name[all_labels[idx]].split("___")[-1].replace("_"," ")
        pred_name = class_name[all_preds[idx]].split("___")[-1].replace("_"," ")
        conf = all_probs[idx,all_preds[idx]] * 100

        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.1f}%",
            fontsize=8,color="red"
        )
        axes[i].axis("off")

    for j in range(i+1,len(axes)):
        axes[j].axis("off")

    plt.suptitle("Most confidently wrong prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS_DIR/"misclassified_images.png",dpi=150)
    plt.show()
    print(f"Saved: misclassified_images.png")
    print(f"Total misclassified: {len(wrong_idx)} / {len(all_labels)}")
else:
    print("No misclassified images found — perfect val accuracy!")

# ---- Save Full Classification Report ----
report=classification_report(
    all_labels,all_preds,
    target_names=class_name,output_dict=True
)
report_df=pd.DataFrame(report).transpose()
report_df.to_csv(FIGS_DIR/"Classification_report.csv")
print(f"\nSaved full classification report to reports/figures/day6/")

# ---- Summary ----
print("\n ---- DAY 6 Summary ----")
print(f"Val accurcay: {accuracy*100:.2f}")
print(f"Macro F1:     {f1_score(all_labels,all_preds,average='macro')*100:.2f}")
print(f"Misclassified: {len(wrong_idx) / len(all_labels)} images")
print(f"Worst class F1: {f1_df.iloc[0]['f1']*100:.2f}% ({f1_df.iloc[0]['class']})")
print(f"Files saved to: reports/figures/day6/")
print("\n Day 6 complete")