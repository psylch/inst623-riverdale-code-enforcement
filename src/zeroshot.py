"""Generic zero-shot classifiers (CLIP + Gemma 4 VLM).

Unlike `clip_baseline.py`, nothing here is hardcoded to a specific class
list. Callers pass in their own classes and prompts, so the same code
serves both the proxy taxonomy and the client taxonomy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import open_clip


# ── CLIP zero-shot ──────────────────────────────────────────────


def load_clip(device: str = "cpu", model_name: str = "ViT-B-32",
              pretrained: str = "laion2b_s34b_b79k"):
    """Load an open_clip model. Defaults to ViT-B/32 / laion2b."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


@torch.no_grad()
def build_text_features(
    model,
    tokenizer,
    classes: Sequence[str],
    prompts: dict[str, list[str]],
    device: str = "cpu",
) -> torch.Tensor:
    """Encode averaged text embeddings for each class. Returns (K, dim)."""
    class_feats = []
    for cls in classes:
        texts = prompts[cls]
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        avg = feats.mean(dim=0)
        avg = avg / avg.norm()
        class_feats.append(avg)
    return torch.stack(class_feats)


@torch.no_grad()
def clip_zero_shot(
    model,
    tokenizer,
    dataloader: DataLoader,
    classes: Sequence[str],
    prompts: dict[str, list[str]],
    device: str = "cpu",
    desc: str = "CLIP zero-shot",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run CLIP zero-shot over a dataloader. Returns (y_true, y_pred, y_probs)."""
    text_features = build_text_features(model, tokenizer, classes, prompts, device)

    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc=desc):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)

        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    y_pred = y_probs.argmax(axis=1)
    return y_true, y_pred, y_probs


# ── Gemma 4 VLM zero-shot ───────────────────────────────────────


GEMMA_PROMPT_TEMPLATE = """You are a municipal code enforcement assistant.

Look at the image and decide which single violation category it best matches.
Choose from this list only:

{class_list}

Return your answer as a single JSON object on one line, with keys:
  "ranked": an array of exactly 3 category ids from the list above, ordered
            from most likely to least likely.

Do not include any other text. Example:
{{"ranked": ["graffiti", "peeling_paint", "broken_windows"]}}
"""


def _build_gemma_prompt(classes: Sequence[str], descriptions: dict[str, str]) -> str:
    class_list = "\n".join(f"- {c}: {descriptions[c]}" for c in classes)
    return GEMMA_PROMPT_TEMPLATE.format(class_list=class_list)


def _parse_ranked(text: str, classes: Sequence[str]) -> list[str]:
    """Extract a ranked list of class ids from Gemma's output. Fuzzy but strict."""
    # Try JSON first.
    m = re.search(r"\{[^{}]*\"ranked\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            ranked = obj.get("ranked", [])
            if isinstance(ranked, list):
                clean = [str(r).strip() for r in ranked if str(r).strip() in classes]
                if clean:
                    return clean
        except json.JSONDecodeError:
            pass

    # Fallback: scan for class ids in order of appearance.
    found: list[str] = []
    lower = text.lower()
    positions: list[tuple[int, str]] = []
    for c in classes:
        idx = lower.find(c.lower())
        if idx >= 0:
            positions.append((idx, c))
    positions.sort()
    for _, c in positions:
        if c not in found:
            found.append(c)
    return found


def _ranked_to_probs(ranked: list[str], classes: Sequence[str]) -> np.ndarray:
    """Convert a ranked list into a pseudo-probability vector.

    We use reciprocal-rank weights (1/1, 1/2, 1/3, …) softmaxed over the full
    class set. Unranked classes get a small uniform residual so downstream
    metrics (top-k, macro F1) remain well-defined.

    If the model returns the same class id more than once (it happens), we
    keep the FIRST occurrence only. Otherwise a class at rank 0 and rank 2
    would be overwritten with the worse score 1/3, and top-1 would flip to
    whatever other class sits at rank 1 — this was a real bug.
    """
    seen: list[str] = []
    for c in ranked:
        if c not in seen:
            seen.append(c)

    K = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    scores = np.full(K, -1e3, dtype=np.float32)
    for rank, cls in enumerate(seen):
        idx = class_to_idx.get(cls)
        if idx is not None:
            scores[idx] = 1.0 / (rank + 1)
    # softmax with a moderate temperature
    t = 0.3
    shifted = (scores - scores.max()) / t
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    return probs


def load_gemma_vlm(model_id: str = "mlx-community/gemma-4-e4b-it-4bit"):
    """Load a Gemma 4 VLM via mlx-vlm. Returns (model, processor, config)."""
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(model_id)
    config = load_config(model_id)
    return model, processor, config


def gemma_zero_shot(
    model,
    processor,
    config,
    image_paths: Sequence[str],
    labels: Sequence[int],
    classes: Sequence[str],
    descriptions: dict[str, str],
    max_tokens: int = 80,
    desc: str = "Gemma-4 zero-shot",
    stream_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run Gemma 4 zero-shot classification over a list of image paths.

    Returns (y_true, y_pred, y_probs, raw_outputs). Per-image, we ask the
    model to rank the full class list and keep the top-3; reciprocal-rank
    is then softmaxed into a probability vector so that top-k accuracy and
    macro F1 can be computed like any other classifier.

    If `stream_path` is given, each image's result is appended as one JSON
    line so long runs can be monitored with `tail -f` and resumed from
    cache on crash.
    """
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    question = _build_gemma_prompt(classes, descriptions)

    all_probs = []
    all_preds = []
    raw_outputs: list[str] = []

    stream_fh = None
    if stream_path is not None:
        stream_path = Path(stream_path)
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_fh = stream_path.open("w")

    try:
        for i, (p, lab) in enumerate(zip(image_paths, labels)):
            formatted = apply_chat_template(processor, config, question, num_images=1)
            out = generate(
                model=model,
                processor=processor,
                prompt=formatted,
                image=[p],
                max_tokens=max_tokens,
                temperature=0.0,
                verbose=False,
            )
            text = out.text if hasattr(out, "text") else str(out)
            raw_outputs.append(text)
            ranked = _parse_ranked(text, classes)
            probs = _ranked_to_probs(ranked, classes)
            pred = int(np.argmax(probs))
            all_probs.append(probs)
            all_preds.append(pred)

            if stream_fh is not None:
                record = {
                    "i": i,
                    "path": str(p),
                    "true": classes[int(lab)],
                    "pred": classes[pred],
                    "ranked": ranked[:3],
                    "ok": classes[int(lab)] == classes[pred],
                    "raw": text.strip()[:400],
                }
                stream_fh.write(json.dumps(record) + "\n")
                stream_fh.flush()
    finally:
        if stream_fh is not None:
            stream_fh.close()

    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    y_probs = np.stack(all_probs, axis=0)
    return y_true, y_pred, y_probs, raw_outputs
