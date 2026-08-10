import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic,QuantType
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS_DIR = Path("data/splits")
EXPORT_DIR = Path("models/exports")
CKPT_DIR = Path("models/checkpoints")

with open(SPLITS_DIR/"class_names.json") as f:
    class_names = json.load(f)
NUM_CLASSES = len(class_names)

ONNX_PATH = EXPORT_DIR / "resnet18_pretrained_best.onnx"
QUANTIZED_PATH = EXPORT_DIR / "crop_disease_model_int8.onnx"

# ---- Quantize ----
print("Quantizing model to INT8...")
quantize_dynamic(
    model_input=str(ONNX_PATH),
    model_output=str(QUANTIZED_PATH),
    weight_type=QuantType.QInt8,
)
print("Quantization complete.")

# ---- Size Comparison ----
import os
orig_size = os.path.getsize(ONNX_PATH) / 1e6
quant_size = os.path.getsize(QUANTIZED_PATH) / 1e6
print(f"\nSize comparison:")
print(f" Original ONNX:  {orig_size:.1f} MB")
print(f" Quantized INT8: {quant_size:.1f} MB")
print(f" Reduction: {(1 - quant_size/orig_size)*100:.1f}%")

# ---- Speed Comparison ----
print("\nBenchmarking speed...")
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

orig_session = ort.InferenceSession(str(ONNX_PATH))
quant_session = ort.InferenceSession(str(QUANTIZED_PATH))

def benchmark(session,n=10):
    for _ in range(10):
        session.run(["output"],{"input":dummy})
    times=[]
    for _ in range(10):
        start=time.perf_counter()
        session.run(["output"],{"input":dummy})
        times.append((time.perf_counter()-start)*1000)
    return np.mean(times)

orig_ms  = benchmark(orig_session)
quant_ms = benchmark(quant_session)
print(f"  Original ONNX: {orig_ms:.2f}ms")
print(f"  Quantized INT8: {quant_ms:.2f}ms")
print(f"  Speedup:  {orig_ms/quant_ms:.2f}x")


# ---- Accuracy check on val set ----
print("\nChecking accuracy on val set...")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

val_df = pd.read_csv(SPLITS_DIR / "val.csv")

sample_df=val_df.sample(500,random_state=42).reset_index(drop=True)

orig_correct = 0
quant_correct = 0

for idx in range(len(sample_df)):
    row=sample_df.iloc[idx]
    img = np.array(Image.open(row['path']).convert('RGB'))
    tensor=transform(image=img)["image"].unsqueeze(0).numpy()
    label=int(row["label"])

    orig_pred=orig_session.run(['output'],{"input":tensor})[0].argmax()
    quant_pred=quant_session.run(['output'],{"input":tensor})[0].argmax()

    if orig_pred  == label: orig_correct  += 1
    if quant_pred == label: quant_correct += 1

    if idx % 100 == 0:
        print(f"  Progress: {idx}/500")

orig_acc  = orig_correct  / len(sample_df)
quant_acc = quant_correct / len(sample_df)

print(f"\nAccuracy on 500 val samples:")
print(f"  Original ONNX: {orig_acc*100:.2f}%")
print(f"  Quantized INT8: {quant_acc*100:.2f}%")
print(f"  Accuracy drop: {(orig_acc - quant_acc)*100:.2f}%")

# ---- Test on real image ----
print("\nReal image test...")

test_df=pd.read_csv(SPLITS_DIR/'test.csv')
sample_row=test_df.iloc[42]
image=np.array(Image.open(sample_row['path']).convert('RGB'))
tensor=transform(image=img)["image"].unsqueeze(0).numpy()

output = quant_session.run(["output"], {"input": tensor})[0]
probs  = np.softmax(output[0]) if hasattr(np, 'softmax') else \
         np.exp(output[0]) / np.exp(output[0]).sum()
top3   = np.argsort(probs)[::-1][:3]

print(f"True:  {class_names[int(sample_row['label'])]}")
print("Top 3 predictions:")
for i in top3:
    print(f"  {class_names[i]}: {probs[i]*100:.2f}%")


# ---- Summary ----
print(f"""
   Day 12 Summary 
  Original ONNX:    {orig_size:.1f} MB  |  {orig_ms:.2f}ms
  Quantized INT8:   {quant_size:.1f} MB  |  {quant_ms:.2f}ms
  Size reduction:   {(1 - quant_size/orig_size)*100:.1f}%
  Speedup:          {orig_ms/quant_ms:.2f}x
  Accuracy drop:    {(orig_acc - quant_acc)*100:.2f}%
  Production model: models/exports/crop_disease_model_int8.onnx
""")
print("✓ Day 12 complete.")