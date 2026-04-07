"""Dataset loading, unification, and splitting."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

# BD3 class → unified label
BD3_MAP = {
    "major_crack": "structural_damage",
    "minor_crack": "structural_damage",
    "peeling": "exterior_deterioration",
    "spalling": "exterior_deterioration",
    "algae": "exterior_deterioration",
    "stain": "exterior_deterioration",
    # "plain" excluded — normal walls, not a violation
}

CLASSES = [
    "structural_damage",
    "exterior_deterioration",
    "trash_debris",
    "overgrown_vegetation",
    "illegal_dumping",
]

CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


# ── catalogue builders ──────────────────────────────────────────


def _catalogue_bd3() -> list[dict]:
    base = ROOT / "BD3" / "BD3_original_dataset" / "train"
    rows = []
    for folder, label in BD3_MAP.items():
        d = base / folder
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rows.append({"path": str(f), "label": label, "source": "BD3"})
    return rows


def _catalogue_taco() -> list[dict]:
    base = ROOT / "TACO" / "data"
    ann = json.loads((base / "annotations.json").read_text())
    rows = []
    for img in ann["images"]:
        p = base / img["file_name"]
        if p.exists():
            rows.append({"path": str(p), "label": "trash_debris", "source": "TACO"})
    return rows


def _catalogue_roboflow(dataset_dir: str, label: str) -> list[dict]:
    """Catalogue a Roboflow COCO-format dataset (uses images only)."""
    base = ROOT / dataset_dir
    rows = []
    for split_dir in ("train", "valid", "test"):
        d = base / split_dir
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rows.append({"path": str(f), "label": label, "source": dataset_dir})
    return rows


def build_catalogue() -> pd.DataFrame:
    """Build a unified DataFrame of all images with labels."""
    rows = []
    rows += _catalogue_bd3()
    rows += _catalogue_taco()
    rows += _catalogue_roboflow("grass-weeds", "overgrown_vegetation")
    rows += _catalogue_roboflow("aerial-dumping", "illegal_dumping")

    df = pd.DataFrame(rows)
    df["label_idx"] = df["label"].map(CLASS2IDX)
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label_idx"], random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label_idx"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ── PyTorch Dataset ─────────────────────────────────────────────


class ViolationDataset(Dataset):
    """Image classification dataset backed by a catalogue DataFrame."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label_idx"])
