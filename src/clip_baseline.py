"""CLIP zero-shot classification baseline."""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import open_clip

from .data import CLASSES

# Violation descriptions for zero-shot prompts
PROMPTS = {
    "structural_damage": [
        "a photo of cracks in a wall or building structure",
        "a photo of structural damage on a building surface",
        "cracked concrete or masonry on a building",
    ],
    "exterior_deterioration": [
        "a photo of peeling paint on a building exterior",
        "a photo of building surface deterioration with mold, stains, or spalling",
        "deteriorated building exterior with algae, rust, or flaking material",
    ],
    "trash_debris": [
        "a photo of trash, litter, and debris on the ground",
        "a photo of garbage accumulation in an outdoor area",
        "waste and rubbish scattered on the street or property",
    ],
    "overgrown_vegetation": [
        "a photo of overgrown grass and weeds",
        "a photo of unmaintained vegetation and tall grass",
        "overgrown plants and weeds on a property",
    ],
    "illegal_dumping": [
        "a photo of an illegal dumping site with waste materials",
        "a photo of illegally dumped garbage and bulky waste",
        "piles of discarded materials dumped illegally outdoors",
    ],
}


def load_clip(device: str = "cpu"):
    """Load CLIP model and return (model, preprocess, tokenizer)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


@torch.no_grad()
def build_text_features(model, tokenizer, device: str = "cpu") -> torch.Tensor:
    """Encode multi-prompt text features for each class. Returns (num_classes, dim)."""
    class_features = []
    for cls in CLASSES:
        texts = PROMPTS[cls]
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # average across prompts
        avg = feats.mean(dim=0)
        avg = avg / avg.norm()
        class_features.append(avg)
    return torch.stack(class_features)  # (num_classes, dim)


@torch.no_grad()
def predict_zero_shot(
    model,
    preprocess,
    tokenizer,
    dataloader: DataLoader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run zero-shot prediction. Returns (y_true, y_pred, y_probs)."""
    text_features = build_text_features(model, tokenizer, device)

    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="CLIP zero-shot"):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity → softmax
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)

        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    y_pred = y_probs.argmax(axis=1)
    return y_true, y_pred, y_probs
