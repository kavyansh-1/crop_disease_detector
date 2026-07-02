import torch
import torchvision
import timm
import albumentations as A

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: None — using CPU")

# tensor test
x = torch.randn(3, 224, 224)
print(f"Tensor shape: {x.shape}")

# timm test
model = timm.create_model('efficientnet_b2', pretrained=False)
out = model(x.unsqueeze(0))
print(f"EfficientNet output shape: {out.shape}")

print("\n✓ Day 1 complete. Environment working.")