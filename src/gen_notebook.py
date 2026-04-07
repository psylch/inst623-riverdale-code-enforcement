"""Generate the experiment notebook with checkpoint/resume support."""

import json
from pathlib import Path

cells = []


def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip()})


def code(source: str):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": source.strip(),
        "outputs": [],
        "execution_count": None,
    })


# ── Title ────────────────────────────────────────────────────────

md("""## AI Adoption Clinic — Final Project: Model Comparison Experiment
-----

**Task**: Image classification for municipal code enforcement — given a property inspection photo, predict the violation type.

**Dataset**: ~8,900 images assembled from 4 public proxy datasets (BD3, TACO, Grass-Weeds, Aerial Dumping), mapped into 5 violation categories.

**Models compared**: CLIP zero-shot (no training) vs DINOv2 ViT-B/14 (LP-FT) vs EfficientNetV2-S (LP-FT).""")

# ── Setup ────────────────────────────────────────────────────────

code("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from src.data import build_catalogue, split_dataset, ViolationDataset, CLASSES, CLASS2IDX
from src.clip_baseline import load_clip, predict_zero_shot
from src.models import create_model, get_transforms, unfreeze_all
from src.train import train_model, TrainConfig, TrainResult
from src.evaluate import predict, print_metrics, plot_confusion_matrix, plot_training_curves, comparison_table
from src.cache import (
    save_predictions, load_predictions, save_model, load_model,
    load_result, save_splits, load_splits, has_checkpoint, save_result,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

np.random.seed(42)
torch.manual_seed(42)
plt.rcParams["figure.dpi"] = 100
sns.set_style("whitegrid")""")

# ── Step 1: Data Preparation ────────────────────────────────────

md("""## Step 1: Data Preparation

I'm combining four public datasets into a unified 5-class classification task. BD3 covers building defects, TACO covers trash/litter, Grass-Weeds covers overgrown vegetation, and Aerial Dumping covers illegal dumping sites. The original datasets have different formats — BD3 is already class-folder based, while the others are COCO object detection format — but since we're doing whole-image classification, I just need the images and a single label per image.""")

code("""catalogue = build_catalogue()

print("=" * 55)
print("DATASET CATALOGUE")
print("=" * 55)
print(f"  Total images: {len(catalogue)}")
print()

for label in CLASSES:
    sub = catalogue[catalogue["label"] == label]
    sources = sub["source"].value_counts()
    src_str = ", ".join(f"{k}={v}" for k, v in sources.items())
    print(f"  {label:<28} {len(sub):>5}  ({src_str})")""")

code("""cached = load_splits()
if cached is not None:
    train_df, val_df, test_df = cached
    print("Loaded cached splits")
else:
    train_df, val_df, test_df = split_dataset(catalogue)
    save_splits(train_df, val_df, test_df)
    print("Created and cached new splits")

print("=" * 55)
print("TRAIN / VAL / TEST SPLIT (80/10/10, stratified)")
print("=" * 55)
print(f"  Train: {len(train_df):>5}")
print(f"  Val:   {len(val_df):>5}")
print(f"  Test:  {len(test_df):>5}")""")

code("""fig, axes = plt.subplots(1, 3, figsize=(14, 4))
short = [c.replace("_", "\\n") for c in CLASSES]

for ax, (name, df) in zip(axes, [("Train", train_df), ("Val", val_df), ("Test", test_df)]):
    counts = df["label"].value_counts().reindex(CLASSES)
    ax.barh(short, counts.values, color=sns.color_palette("muted", len(CLASSES)))
    ax.set_title(f"{name} ({len(df)})")
    ax.set_xlabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(v + 10, i, str(v), va="center", fontsize=9)

plt.suptitle("Class Distribution Across Splits", fontweight="bold")
plt.tight_layout()
plt.show()""")

code("""from PIL import Image

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for col, label in enumerate(CLASSES):
    samples = train_df[train_df["label"] == label].sample(2, random_state=42)
    for row, (_, s) in enumerate(samples.iterrows()):
        img = Image.open(s["path"]).convert("RGB")
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(label.replace("_", "\\n"), fontsize=9)

plt.suptitle("Sample Images per Class", fontweight="bold")
plt.tight_layout()
plt.show()""")

md("""The class distribution is reasonably balanced — the smallest class has around 1,200 images and the largest around 2,500, roughly a 2x ratio. That's workable without aggressive resampling. I'll just use a `WeightedRandomSampler` during training to keep things fair.

One thing worth noting: Grass-Weeds and Aerial Dumping were originally object detection datasets with bounding box annotations. I'm ignoring the boxes and treating each image as a single-label classification sample. For our use case this makes sense — a code enforcement inspector takes a photo of one violation scene, they don't need multi-object detection within the frame.""")

# ── Step 2: CLIP Zero-Shot ───────────────────────────────────────

md("""## Step 2: CLIP Zero-Shot Baseline

CLIP can classify images using natural language descriptions without any training. I'm using ViT-B/32 with three text prompts per violation class (averaged) to get a Day 1 baseline. The point isn't to get great accuracy — it's to see how far we can get with zero training cost and to have a meaningful comparison point for the trained models.""")

code("""cached_clip = load_predictions("clip")
if cached_clip is not None:
    clip_true, clip_pred, clip_probs = cached_clip
    print("Loaded cached CLIP predictions")
else:
    clip_model, clip_preprocess, clip_tokenizer = load_clip(DEVICE)

    clip_test_ds = ViolationDataset(test_df, transform=clip_preprocess)
    clip_test_loader = DataLoader(clip_test_ds, batch_size=64, num_workers=4)

    clip_true, clip_pred, clip_probs = predict_zero_shot(
        clip_model, clip_preprocess, clip_tokenizer, clip_test_loader, DEVICE
    )
    save_predictions("clip", clip_true, clip_pred, clip_probs)
    del clip_model  # free memory

clip_metrics = print_metrics(clip_true, clip_pred, clip_probs, "CLIP Zero-Shot")""")

code("""plot_confusion_matrix(clip_true, clip_pred, "CLIP Zero-Shot — Confusion Matrix")
plt.tight_layout()
plt.show()""")

md("""PLACEHOLDER — will be written after execution based on actual CLIP results.""")

# ── Step 3: DINOv2 ───────────────────────────────────────────────

md("""## Step 3: DINOv2 ViT-B/14 (LP-FT)

DINOv2 is a self-supervised vision model from Meta, trained on 142M images without labels. The idea is that its frozen features should already be strong enough to classify our violation types with just a linear head on top. I'll first do a linear probe (backbone frozen), then unlock everything for full fine-tuning — this LP-FT approach avoids the feature distortion problem that can happen when you fine-tune from a random head.""")

code("""# --- shared data loaders for DINOv2 ---
dino_model_tmp = create_model("dinov2", num_classes=len(CLASSES))
dino_train_tf = get_transforms(dino_model_tmp, is_training=True)
dino_val_tf = get_transforms(dino_model_tmp, is_training=False)
del dino_model_tmp

train_ds = ViolationDataset(train_df, transform=dino_train_tf)
val_ds = ViolationDataset(val_df, transform=dino_val_tf)
test_ds = ViolationDataset(test_df, transform=dino_val_tf)

class_counts = train_df["label_idx"].value_counts().sort_index().values
weights = 1.0 / class_counts
sample_weights = weights[train_df["label_idx"].values]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=64, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=64, num_workers=4)
print("DINOv2 data loaders ready")""")

code("""# Phase 1: Linear Probe (frozen backbone)
cached_lp = load_predictions("dino_lp")
if cached_lp is not None:
    dino_lp_true, dino_lp_pred, dino_lp_probs = cached_lp
    dino_lp_result_dict = load_result("dino_lp")
    # still need the model for FT phase
    dino_lp = create_model("dinov2", num_classes=len(CLASSES), freeze_backbone=True)
    load_model("dino_lp", dino_lp)
    print("Loaded cached DINOv2 LP checkpoint")
else:
    dino_lp = create_model("dinov2", num_classes=len(CLASSES), freeze_backbone=True)
    lp_config = TrainConfig(epochs=30, lr=1e-3, patience=5, warmup_epochs=2)
    dino_lp, dino_lp_result = train_model(
        dino_lp, train_loader, val_loader, lp_config, DEVICE, label="DINOv2 LP"
    )
    dino_lp_true, dino_lp_pred, dino_lp_probs = predict(dino_lp, test_loader, DEVICE)
    save_model("dino_lp", dino_lp, dino_lp_result)
    save_predictions("dino_lp", dino_lp_true, dino_lp_pred, dino_lp_probs)

dino_lp_metrics = print_metrics(dino_lp_true, dino_lp_pred, dino_lp_probs, "DINOv2 Linear Probe")""")

code("""# Phase 2: Full Fine-Tune (initialized from LP checkpoint)
cached_ft = load_predictions("dino_ft")
if cached_ft is not None:
    dino_ft_true, dino_ft_pred, dino_ft_probs = cached_ft
    dino_ft_result_dict = load_result("dino_ft")
    print("Loaded cached DINOv2 FT checkpoint")
else:
    unfreeze_all(dino_lp)
    ft_config = TrainConfig(epochs=20, lr=5e-5, patience=5, warmup_epochs=2, weight_decay=0.01)
    dino_ft, dino_ft_result = train_model(
        dino_lp, train_loader, val_loader, ft_config, DEVICE, label="DINOv2 FT"
    )
    dino_ft_true, dino_ft_pred, dino_ft_probs = predict(dino_ft, test_loader, DEVICE)
    save_model("dino_ft", dino_ft, dino_ft_result)
    save_predictions("dino_ft", dino_ft_true, dino_ft_pred, dino_ft_probs)
    del dino_ft
del dino_lp
torch.mps.empty_cache() if DEVICE == "mps" else None

dino_ft_metrics = print_metrics(dino_ft_true, dino_ft_pred, dino_ft_probs, "DINOv2 LP-FT")""")

md("""PLACEHOLDER — will be written after execution based on actual DINOv2 results.""")

# ── Step 4: EfficientNetV2-S ─────────────────────────────────────

md("""## Step 4: EfficientNetV2-S (LP-FT)

EfficientNetV2-S is a well-established CNN pretrained on ImageNet-21k. It's my familiar baseline — I've used EfficientNet variants in past projects. Running the same LP-FT procedure here lets me directly compare a modern self-supervised ViT (DINOv2) against a strong supervised CNN on the same data, same splits, same training setup.""")

code("""# --- shared data loaders for EfficientNetV2 ---
eff_model_tmp = create_model("efficientnetv2", num_classes=len(CLASSES))
eff_train_tf = get_transforms(eff_model_tmp, is_training=True)
eff_val_tf = get_transforms(eff_model_tmp, is_training=False)
del eff_model_tmp

eff_train_ds = ViolationDataset(train_df, transform=eff_train_tf)
eff_val_ds = ViolationDataset(val_df, transform=eff_val_tf)
eff_test_ds = ViolationDataset(test_df, transform=eff_val_tf)

eff_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

eff_train_loader = DataLoader(eff_train_ds, batch_size=32, sampler=eff_sampler, num_workers=4)
eff_val_loader = DataLoader(eff_val_ds, batch_size=64, num_workers=4)
eff_test_loader = DataLoader(eff_test_ds, batch_size=64, num_workers=4)
print("EfficientNetV2 data loaders ready")""")

code("""# Phase 1: Linear Probe
cached_lp = load_predictions("eff_lp")
if cached_lp is not None:
    eff_lp_true, eff_lp_pred, eff_lp_probs = cached_lp
    eff_lp = create_model("efficientnetv2", num_classes=len(CLASSES), freeze_backbone=True)
    load_model("eff_lp", eff_lp)
    print("Loaded cached EfficientNetV2 LP checkpoint")
else:
    eff_lp = create_model("efficientnetv2", num_classes=len(CLASSES), freeze_backbone=True)
    lp_config = TrainConfig(epochs=30, lr=1e-3, patience=5, warmup_epochs=2)
    eff_lp, eff_lp_result = train_model(
        eff_lp, eff_train_loader, eff_val_loader, lp_config, DEVICE, label="EffNetV2 LP"
    )
    eff_lp_true, eff_lp_pred, eff_lp_probs = predict(eff_lp, eff_test_loader, DEVICE)
    save_model("eff_lp", eff_lp, eff_lp_result)
    save_predictions("eff_lp", eff_lp_true, eff_lp_pred, eff_lp_probs)

eff_lp_metrics = print_metrics(eff_lp_true, eff_lp_pred, eff_lp_probs, "EfficientNetV2 Linear Probe")""")

code("""# Phase 2: Full Fine-Tune
cached_ft = load_predictions("eff_ft")
if cached_ft is not None:
    eff_ft_true, eff_ft_pred, eff_ft_probs = cached_ft
    print("Loaded cached EfficientNetV2 FT checkpoint")
else:
    unfreeze_all(eff_lp)
    ft_config = TrainConfig(epochs=20, lr=5e-5, patience=5, warmup_epochs=2, weight_decay=0.01)
    eff_ft, eff_ft_result = train_model(
        eff_lp, eff_train_loader, eff_val_loader, ft_config, DEVICE, label="EffNetV2 FT"
    )
    eff_ft_true, eff_ft_pred, eff_ft_probs = predict(eff_ft, eff_test_loader, DEVICE)
    save_model("eff_ft", eff_ft, eff_ft_result)
    save_predictions("eff_ft", eff_ft_true, eff_ft_pred, eff_ft_probs)
    del eff_ft
del eff_lp
torch.mps.empty_cache() if DEVICE == "mps" else None

eff_ft_metrics = print_metrics(eff_ft_true, eff_ft_pred, eff_ft_probs, "EfficientNetV2 LP-FT")""")

md("""PLACEHOLDER — will be written after execution based on actual EfficientNetV2 results.""")

# ── Step 5: Comparison ───────────────────────────────────────────

md("""## Step 5: Model Comparison""")

code("""all_metrics = [clip_metrics, dino_lp_metrics, dino_ft_metrics, eff_lp_metrics, eff_ft_metrics]
comparison_table(all_metrics)""")

code("""# load training curves from cache if needed
def _load_or_use(name, local_var_name):
    \"\"\"Return TrainResult from local scope or cache.\"\"\"
    import src.train as _t
    cached = load_result(name)
    if cached is None:
        return None
    r = _t.TrainResult()
    r.train_losses = cached["train_losses"]
    r.val_losses = cached["val_losses"]
    r.val_accs = cached["val_accs"]
    r.best_val_acc = cached["best_val_acc"]
    r.best_epoch = cached["best_epoch"]
    return r

curves = {}
for key, var in [("DINOv2 LP", "dino_lp_result"), ("DINOv2 FT", "dino_ft_result"),
                 ("EffNetV2 LP", "eff_lp_result"), ("EffNetV2 FT", "eff_ft_result")]:
    tag = key.lower().replace(" ", "_").replace("dinov2", "dino").replace("effnetv2", "eff")
    r = _load_or_use(tag, var)
    if r is not None:
        curves[key] = r

if curves:
    plot_training_curves(curves)
else:
    print("No training curves available")""")

code("""fig, axes = plt.subplots(1, 3, figsize=(20, 5))
plot_confusion_matrix(clip_true, clip_pred, "CLIP Zero-Shot", ax=axes[0])
plot_confusion_matrix(dino_ft_true, dino_ft_pred, "DINOv2 LP-FT", ax=axes[1])
plot_confusion_matrix(eff_ft_true, eff_ft_pred, "EfficientNetV2 LP-FT", ax=axes[2])
plt.tight_layout()
plt.show()""")

md("""PLACEHOLDER — final comparison discussion will be written after execution.""")

md("""## Reflection

PLACEHOLDER — will be written after seeing all results.""")

# ── Write notebook ───────────────────────────────────────────────

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "experiment.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Notebook written to {out}")
