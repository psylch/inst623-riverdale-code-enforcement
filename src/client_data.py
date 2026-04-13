"""Client (Riverdale Park) violation data loading and taxonomy.

The client dataset lives under `data/client-data/<§-folder>/*.jpg`. Each
folder name is the official violation code + section. We only keep the
folders that actually contain images (5 of the 14 are empty at the moment).

This module intentionally does NOT reuse `data.CLASSES` — the proxy taxonomy
(5 coarse buckets) and the client taxonomy (14 fine-grained codes) are
different label spaces. Zero-shot evaluation lives in its own namespace.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

CLIENT_ROOT = Path(__file__).resolve().parent.parent / "data" / "client-data"

# Canonical client classes. Order is the label index order.
# Folders that currently have 0 images are omitted from the active taxonomy.
CLIENT_CLASSES: list[str] = [
    "boarded_windows",
    "broken_windows",
    "damaged_roof_shingles",
    "deteriorating_chimney",
    "graffiti",
    "inoperable_vehicle",
    "junk_trash_accumulation",
    "overgrown_vegetation",
    "peeling_paint",
]

CLIENT_CLASS2IDX = {c: i for i, c in enumerate(CLIENT_CLASSES)}

# Folder name (as it appears on disk) → canonical class id.
# Folders are matched via substring on the official code (e.g. "§ 304.13")
# because some have a leading space or hyphenation variance.
FOLDER_KEYWORD_MAP: dict[str, str] = {
    "§ 304.13 - 108.2": "boarded_windows",
    "§ 304.13.1":        "broken_windows",
    "§ 304.7":           "damaged_roof_shingles",
    "§ 304.11":          "deteriorating_chimney",
    "§ 302.9":           "graffiti",
    "§ 302.8":           "inoperable_vehicle",
    "§ 302.1 - 308.1":   "junk_trash_accumulation",
    "§ 302.4":           "overgrown_vegetation",
    "§ 304.2":           "peeling_paint",
}

# Short human-readable display names for plot labels.
CLIENT_DISPLAY: dict[str, str] = {
    "boarded_windows":        "Boarded\nWindows",
    "broken_windows":         "Broken\nWindows",
    "damaged_roof_shingles":  "Damaged\nRoof Shingles",
    "deteriorating_chimney":  "Deteriorating\nChimney",
    "graffiti":               "Graffiti",
    "inoperable_vehicle":     "Inoperable\nVehicle",
    "junk_trash_accumulation":"Junk / Trash\nAccumulation",
    "overgrown_vegetation":   "Overgrown\nVegetation",
    "peeling_paint":          "Peeling\nPaint",
}

# CLIP zero-shot prompts. 3 paraphrases per class, averaged in clip_baseline
# style. Keep them natural and visually grounded.
CLIENT_CLIP_PROMPTS: dict[str, list[str]] = {
    "boarded_windows": [
        "a photo of a house with windows covered by wooden boards",
        "a building exterior with plywood nailed over the windows",
        "boarded-up windows on a vacant or damaged property",
    ],
    "broken_windows": [
        "a photo of a broken or shattered window on a building",
        "a house exterior with cracked or smashed glass windows",
        "a window with holes, cracks, or missing glass panes",
    ],
    "damaged_roof_shingles": [
        "a photo of a roof with missing or damaged shingles",
        "a rooftop with peeling, lifted, or broken asphalt shingles",
        "visible damage and gaps on the shingles of a house roof",
    ],
    "deteriorating_chimney": [
        "a photo of a deteriorating brick chimney on a house",
        "a crumbling chimney with cracked bricks and missing mortar",
        "a damaged residential chimney in poor structural condition",
    ],
    "graffiti": [
        "a photo of graffiti spray-painted on an exterior wall",
        "spray-painted tags and markings on a building surface",
        "a wall or fence vandalized with painted graffiti",
    ],
    "inoperable_vehicle": [
        "a photo of an abandoned inoperable car on a property",
        "a derelict vehicle with flat tires, rust, or no license plate",
        "an unused junk car parked on a residential lot",
    ],
    "junk_trash_accumulation": [
        "a photo of junk, debris, and trash piled on a property",
        "outdoor accumulation of discarded items and household rubbish",
        "a yard filled with garbage bags, broken furniture, and debris",
    ],
    "overgrown_vegetation": [
        "a photo of overgrown grass and weeds on a property",
        "a lawn with tall unmaintained grass and weeds",
        "an unkempt yard with excessive vegetation",
    ],
    "peeling_paint": [
        "a photo of peeling and chipping paint on a building exterior",
        "flaking and deteriorated paint on wooden siding",
        "a house facade with cracked, peeling exterior paint",
    ],
}

# Human-readable descriptions for VLM (Gemma) prompts. Fed into an
# instruction prompt that asks the model to rank categories.
CLIENT_DESCRIPTIONS: dict[str, str] = {
    "boarded_windows":
        "Boarded Windows — windows covered with plywood or boards, typical of vacant or damaged properties.",
    "broken_windows":
        "Broken Windows — visibly cracked, shattered, or missing window glass on a building.",
    "damaged_roof_shingles":
        "Damaged/Missing Roof Shingles — visible damage, gaps, peeling, or missing shingles on a roof.",
    "deteriorating_chimney":
        "Deteriorating Chimney — a chimney with structural damage, cracking, or crumbling bricks.",
    "graffiti":
        "Graffiti — spray-painted tags, markings, or drawings on exterior surfaces.",
    "inoperable_vehicle":
        "Inoperable/Unlicensed Vehicles — abandoned cars, vehicles missing plates, flat tires, or in obvious disrepair.",
    "junk_trash_accumulation":
        "Junk/Debris/Trash Accumulation — piles of garbage, junk, or discarded materials on a property.",
    "overgrown_vegetation":
        "Long Grass/Overgrown Vegetation — grass or weeds that are excessively tall or unmaintained.",
    "peeling_paint":
        "Peeling/Deteriorating Exterior Paint — exterior paint that is chipping, flaking, or peeling off.",
}


def build_client_catalogue() -> pd.DataFrame:
    """Scan `data/client-data/` and return a DataFrame with path/label/label_idx.

    Folders with no images are skipped. Folder names are matched against
    FOLDER_KEYWORD_MAP by substring.
    """
    rows: list[dict] = []
    for folder in sorted(CLIENT_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        label: str | None = None
        for keyword, canonical in FOLDER_KEYWORD_MAP.items():
            if keyword in folder.name:
                label = canonical
                break
        if label is None:
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rows.append({
                    "path": str(f),
                    "label": label,
                    "folder": folder.name.strip(),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["label_idx"] = df["label"].map(CLIENT_CLASS2IDX)
    return df


class ClientImageDataset(Dataset):
    """Thin wrapper for CLIP preprocessing of the client eval set."""

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
