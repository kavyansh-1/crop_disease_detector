### Day 3 — DataLoader Pipeline
- Built stratified 70/15/15 train/val/test splits
- Saved splits as CSV files to `data/splits/`
- Built custom `PlantVillageDataset` PyTorch class
- Added albumentations augmentation pipeline (flip, crop, brightness, blur)
- Verified batch shape, dtype, pixel range
- Visualized one full training batch — saved to `reports/figures/sample_batch.png`