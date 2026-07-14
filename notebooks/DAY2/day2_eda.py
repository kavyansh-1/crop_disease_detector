import os
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Dataset path
DATA_ROOT=Path("data/raw")
possible=[
    DATA_ROOT/ "plantvillage dataset" / "color",
    DATA_ROOT/ "color",
    DATA_ROOT/ "PlantVillage"
]
DATASET_DIR =  next((p for p in possible if p.exists()),None)

if DATASET_DIR is None:
    # Auto-find: look for a folder containing 38 subfolders
    for p in DATA_ROOT.rglob("*"):
        if p.is_dir() and len(list(p.iterdir())) > 30:
            DATASET_DIR = p
            break
print(f"Dataset found at: {DATASET_DIR}")

#Count images per class
class_counts={}
for class_dir in sorted(DATASET_DIR.iterdir()):
    if class_dir.is_dir():
        images=list(class_dir.glob("*.jpg"))+list(class_dir.glob("*.JPG"))+list(class_dir.glob("*.png"))
        class_counts[class_dir.name]=len(images)

print(f"\nTotal classes: {len(class_counts)}")
print(f"Total images: {sum(class_counts.values())}")

#Build a clean data frame
df=pd.DataFrame([
    {"class":k,"count":v,
     "crop":k.split("___")[0].replace("_"," "),
     "disease":k.split("___")[1].replace("_"," ") if "___" in k else "unknown"
    }
    for k,v in class_counts.items()
])
df=df.sort_values("count",ascending=False)
print("\nTop 10 classes by image count:")
print(df[["class","count"]].head(10).to_string(index=False))

#Bar Chart images per class
fig,ax=plt.subplots(figsize=(16,7))
colors=["#e74c3c" if "healthy" not in c.lower() else "#2ecc71"
          for c in df["class"]]
bars=ax.barh(df["class"],df["count"],color=colors)
ax.set_xlabel("Number of images",fontsize=12)
ax.set_title("PlantVillage - images per class\n(red=diseased,green=healthy)",fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("reports/figures/class_distribution.png",dpi=150)
plt.show()
print("Saved:reports/figures/class_distribution.png")

#Crop level  summary
crop_summary=df.groupby("crop")["count"].sum().sort_values(ascending=False)
print("\nImages per crop" )
print(crop_summary.to_string())

fx,ax=plt.subplots(figsize=(10,5))
crop_summary.plot(kind="bar",ax=ax,color="#3498db")
ax.set_title("Total Images per crop",fontsize=13)
ax.set_ylabel(" Image count")
ax.set_xlabel("")
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.savefig("reports/figures/crop_distribution.png", dpi=150)
plt.show()
print("Saved: reports/figures/crop_distribution.png")

#Haryana Relevant crops check
haryana_crops=["Wheat","Tomata","Potato","Corn","Rice"]
print("/nHaryana-relevant crops in dataset")
for crop in haryana_crops:
    matches=df[df["crop"].str.lower()==crop.lower()]
    if not matches.empty:
        total=matches["count"].sum()
        classes=len(matches)
        print(f"{crop}:{total} images across {classes} classes")
    else:
        print(f" {crop}:NOT in dataset - need to collect manually")

#   Visualize sample images
print("\n Saving sample images")
sample_classes=df["class"].head(9).tolist()
fig,axes=plt.subplots(3,3,figsize=(13,13))
axes=axes.flatten()

for i,class_name in enumerate(sample_classes):
    class_path=DATASET_DIR/class_name
    imgs=list(class_path.glob("*.jpg"))[:1]+list(class_path.glob("*.JPG"))[:1]
    if imgs:
        img=mpimg.imread(str(imgs[0]))
        axes[i].imshow(img)
        short=class_name.replace("__","\n").replace("_"," ")
        axes[i].set_title(short,fontsize=8)
        axes[i].axis("off")

plt.suptitle("Sample images from PlantVillage",fontsize=14,y=1.01)
plt.tight_layout()
plt.savefig("reports/figures/sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/figures/sample_images.png")

# Save class info to json
output = {
    "total_classes": len(class_counts),
    "total_images": int(sum(class_counts.values())),
    "class_counts": class_counts,
    "crops": crop_summary.to_dict(),
}
with open("reports/class_info.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✓ Day 2 EDA complete.")
print(f"  Classes: {len(class_counts)}")
print(f"  Images:  {sum(class_counts.values())}")
print("  Check reports/figures/ for all charts")