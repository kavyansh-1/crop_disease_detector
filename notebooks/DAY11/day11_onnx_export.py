import json
import time
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = Path("models/checkpoints")
EXPORT_DIR = Path("models/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

with open("data/splits/class_names.json") as f:
    class_names = json.load(f)
NUM_CLASSES = len(class_names)

# ---- Load model ----
model = models.resnet18(weights=None)
model.fc=nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(512,NUM_CLASSES)
)

model.load_state_dict(torch.load(
    CKPT_DIR / "resnet18_pretrained_best.pt",
    map_location=DEVICE,
    weights_only=True
))

model=model.to(DEVICE)
model.eval()
print("Model Loaded")

# ---- Export to ONNX ----
dummy_input = torch.randn(1,3,224,224).to(DEVICE)
onnx_path = EXPORT_DIR / "resnet18_pretrained_best.onnx"

torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)
print(f"ONNX model exported: {onnx_path}")

# ---- Validate ONNX output matches Pytorch ----
print("\nValidating ONNX vs PyTorch outputs...")
test_input = torch.randn(4, 3, 224, 224).to(DEVICE)

with torch.no_grad():
    pytorch_out=model(test_input).cpu().numpy()

session=ort.InferenceSession(str(onnx_path))
onnx_out=session.run(
    ["output"],
    {'input':test_input.cpu().numpy()}
)[0]

max_diff = np.abs(pytorch_out - onnx_out).max()
print(f"Max output difference: {max_diff:.8f}")
print(f"Outputs match: {max_diff < 0.001}")


# ---- ONNX Inference speed ----
print("\nBenchmarking ONNX inference speed...")
single = np.random.randn(1, 3, 224, 224).astype(np.float32)

times=[]
for _ in range(10):
    start = time.perf_counter()
    session.run(["output"],{"input":single})
    times.append((time.perf_counter() - start)*1000)


avg_onnx_ms = np.mean(times)
print(f"ONNX inference speed: {avg_onnx_ms:.2f}ms")

# ---- Compare Sizes ----
import os
pt_size   = os.path.getsize(CKPT_DIR   / "resnet18_pretrained_best.pt") / 1e6
onnx_size = os.path.getsize(onnx_path) / 1e6
print(f"\nSize comparison:")
print(f"  PyTorch .pt:  {pt_size:.1f} MB")
print(f"  ONNX:         {onnx_size:.1f} MB")

# ----  Test on real image ----
print("\nTesting on real image...")
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

import pandas as pd
test_df =pd.read_csv("data/splits/test.csv")
sample_row=test_df.iloc[0]
img = np.array(Image.open(sample_row["path"]).convert("RGB"))
img_tensor = transform(image=img)["image"].unsqueeze(0).numpy()

output = session.run(["output"],{"input":img_tensor})[0]
probs = np.exp(output) / np.exp(output).sum()
top3_idx=np.argsort(probs[0])[::-1][:3]

print(f"True label:  {class_names[sample_row['label']]}")
print(f"Predictions:")
for i in top3_idx:
    print(f"  {class_names[i]}: {probs[0][i]*100:.2f}%")

# ---- Save model and class names ----
with open(EXPORT_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)
print(f"\nSaved class_names.json to exports/")

print(f"""
---- Day 11 Summary ----
  ONNX exported:     {onnx_path}
  ONNX size:         {onnx_size:.1f} MB
  Outputs match PT:  {max_diff < 0.001}
  ONNX speed:        {avg_onnx_ms:.2f}ms
""")
print("✓ Day 11 complete.")