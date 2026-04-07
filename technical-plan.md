# Technical Plan: Model Selection & Dataset Strategy

> Actionable reference for the team. This is what we're actually going to do.

---

## Phase 0: Zero-Shot Baseline (no training needed)

**Goal**: Get a working classifier on Day 1 using CLIP.

```python
# Pseudocode — classify a violation photo with zero training
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

violation_labels = [
    "overgrown grass and vegetation",
    "abandoned vehicle on property",
    "trash and debris accumulation",
    "broken or missing fence",
    "peeling paint on building exterior",
    "damaged roof or gutters",
    "vehicle parked on unimproved surface",
    "illegal sign or banner",
    "standing water or drainage issue",
    # ... add more from Riverdale Park code categories
]

image = load_inspection_photo("photo.jpg")
inputs = processor(text=violation_labels, images=image, return_tensors="pt")
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
# → ranked list of violation types with confidence scores
```

**Why this matters**: Even if accuracy is moderate (~40-60%), it validates the approach and gives us a comparison point.

---

## Phase 1: Proxy Dataset Assembly

Since client data isn't available yet, assemble a training set from public sources:

### Priority datasets to download

| Dataset | Download From | Target Violations It Covers |
|---------|--------------|----------------------------|
| BD3 (3,965 imgs) | [GitHub](https://github.com/Praveenkottari/BD3-Dataset) | Cracks, peeling, spalling, stains |
| TACO (1,500 imgs) | [tacodataset.org](http://tacodataset.org/) | Trash, litter, debris |
| Aerial Dumping (1,555 imgs) | [Roboflow](https://universe.roboflow.com/object-detection-of-illegal-dumping-sites/aerial-dumping-sites) | Illegal dumping |
| Grass-Weeds (2,486 imgs) | [Roboflow](https://universe.roboflow.com/roboflow-100/grass-weeds) | Overgrown vegetation |

**Total: ~9,500 images across 4 violation domains**

### Label mapping strategy

Map public dataset classes → our violation categories:

```
BD3 "major crack" + "minor crack"  →  "structural damage"
BD3 "peeling" + "spalling"         →  "building exterior deterioration"
TACO (all classes)                  →  "trash/debris accumulation"
Aerial Dumping                      →  "illegal dumping"
Grass-Weeds                         →  "overgrown vegetation"
```

When real data arrives, we fine-tune further on the actual Riverdale Park categories.

---

## Phase 2: Model Training Pipeline

### Recommended approach: DINOv2 + LP-FT

```
Step 1: Feature extraction (DINOv2 frozen)
        └── Train linear head only, 20-50 epochs, lr=0.001

Step 2: Full fine-tuning (initialized from Step 1)
        └── All layers unfrozen, 20-50 epochs, lr=1e-5 to 5e-5
        └── Cosine annealing scheduler + 5% warmup
        └── AdamW optimizer, weight_decay=0.01
        └── Early stopping on validation F1
```

### Model comparison experiment

Run all three to compare:

| Model | Library | Model ID |
|-------|---------|----------|
| DINOv2 ViT-B/14 | timm / HF | `facebook/dinov2-base` |
| ConvNeXt V2 Base | timm | `convnextv2_base.fcmae_ft_in22k_in1k` |
| EfficientNetV2-S | timm | `tf_efficientnetv2_s.in21k_ft_in1k` |

### Augmentation config (Albumentations)

```python
import albumentations as A

train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.RandomRain(p=0.1),       # inspectors shoot in rain
    A.RandomFog(p=0.1),        # and fog
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Plus MixUp (alpha=0.2) and CutMix (alpha=1.0) via `timm` during training.

### Multi-label setup

```python
import torch.nn as nn

# Sigmoid per class, not softmax
criterion = nn.BCEWithLogitsLoss()

# Or Asymmetric Loss for class imbalance
# pip install asym-loss  →  AsymmetricLoss(gamma_neg=4, gamma_pos=0)
```

### Evaluation metrics

```python
from sklearn.metrics import classification_report, f1_score

# Per-class precision, recall, F1
print(classification_report(y_true, y_pred, target_names=class_names))

# Top-3 accuracy: is the correct label in the top 3 predictions?
top3_acc = (y_true_onehot * top3_pred).any(axis=1).mean()
```

---

## Phase 3: Swap in Real Data

When client data arrives:

1. Map Riverdale Park violation codes → our category taxonomy
2. Combine with proxy dataset (if categories overlap) or train from scratch on real data only
3. Re-run LP-FT pipeline
4. Run fairness analysis by neighborhood (Niping's evaluation framework)

---

## Project Structure (proposed)

```
FinalProject/
├── research-for-teammates.md    # This doc (learning material)
├── technical-plan.md            # This doc (actionable plan)
├── data/
│   ├── raw/                     # Downloaded datasets
│   ├── processed/               # Unified format, train/val/test splits
│   └── label_mapping.json       # Public dataset class → our categories
├── src/
│   ├── dataset.py               # DataLoader, augmentations
│   ├── model.py                 # Model definitions (DINOv2, ConvNeXt, etc.)
│   ├── train.py                 # Training loop
│   ├── evaluate.py              # Metrics, fairness analysis
│   └── clip_baseline.py         # Zero-shot CLIP baseline
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_clip_zero_shot.ipynb  # CLIP baseline experiment
│   └── 03_model_comparison.ipynb # DINOv2 vs ConvNeXt vs EfficientNet
└── results/
    ├── metrics/                 # Per-experiment metrics JSON
    └── figures/                 # Training curves, confusion matrices
```

---

## Dependencies

```toml
# Add to pyproject.toml
[project.dependencies]
torch = ">=2.2"
torchvision = ">=0.17"
timm = ">=1.0"
transformers = ">=4.40"
albumentations = ">=1.4"
scikit-learn = ">=1.4"
pandas = ">=2.2"
matplotlib = ">=3.8"
seaborn = ">=0.13"
```

---

## Timeline

| Week | What | Who |
|------|------|-----|
| Now | Share research docs with team; download proxy datasets | Chihao |
| Week 1 | CLIP zero-shot baseline; EDA on proxy datasets | Chihao + Niping |
| Week 2 | DINOv2 / ConvNeXt / EfficientNet comparison | Chihao |
| Week 3 | Best model LP-FT on proxy data; evaluation framework | Chihao + Niping |
| When data arrives | Swap in real data; re-train; fairness analysis | All |
