"""CLIP per-class separability analysis.

For each class X, compute CLIP's cosine similarity between every image
and X's prototype, then split those 98 scores by multi-label ground
truth (positive = this photo HAS label X in some folder; negative =
does not). Compute per-class AUC and plot side-by-side histograms.

Outputs:
  - checkpoints/client_clip_similarity.npz  (98×9 raw cosine sim matrix)
  - reports/figures/clip_separability.png   (9-panel histogram grid)
  - reports/figures/clip_separability_summary.png  (AUC bar chart)
  - checkpoints/clip_separability_summary.json     (per-class numbers)

This script does NOT use softmax anywhere. Downstream analyses that
want the binary-detection-style "raw CLIP score per (image, class)"
should read the .npz here, not `client_clip_preds.npz` (which is the
softmaxed classification cache).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.client_data import (
    CLIENT_CLASSES,
    CLIENT_CLIP_PROMPTS,
    CLIENT_DISPLAY,
    ClientImageDataset,
    FOLDER_KEYWORD_MAP,
    build_client_catalogue,
)
from src.zeroshot import build_text_features, load_clip


def build_multilabel_truth(df) -> np.ndarray:
    """Return (N, K) 0/1 matrix of valid labels per physical image.

    Uses md5 hashing to detect byte-identical files duplicated across
    multiple category folders (the client's implicit multi-label
    ground truth).
    """
    root = REPO / "data" / "client-data"
    hash2labels: dict[str, set[str]] = defaultdict(set)
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        for keyword, canonical in FOLDER_KEYWORD_MAP.items():
            if keyword in f.parent.name:
                hash2labels[h].add(canonical)
                break

    path2hash: dict[str, str] = {}
    for f in root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        path2hash[str(f.resolve())] = hashlib.md5(f.read_bytes()).hexdigest()

    K = len(CLIENT_CLASSES)
    Y = np.zeros((len(df), K), dtype=np.int8)
    for i, p in enumerate(df["path"].tolist()):
        h = path2hash[str(Path(p).resolve())]
        for lab in hash2labels[h]:
            Y[i, CLIENT_CLASSES.index(lab)] = 1
    return Y


@torch.no_grad()
def compute_similarity_matrix(device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run CLIP once and return (similarity_matrix, y_true_singlelabel).

    similarity_matrix: (N, K) raw cosine similarities in roughly [-1, 1]
    y_true_singlelabel: (N,) original folder-assigned class indices
    """
    df = build_client_catalogue()
    model, preprocess, tokenizer = load_clip(device)
    text_features = build_text_features(
        model, tokenizer, CLIENT_CLASSES, CLIENT_CLIP_PROMPTS, device
    )  # (K, 512), L2-normalized

    ds = ClientImageDataset(df, transform=preprocess)
    loader = DataLoader(ds, batch_size=32, num_workers=2)

    all_feats = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu())
        all_labels.append(labels.numpy())

    image_features = torch.cat(all_feats, dim=0)          # (N, 512)
    sim = image_features @ text_features.cpu().T          # (N, K) raw cosine sim
    y_true = np.concatenate(all_labels)
    return sim.numpy(), y_true, df


def per_class_stats(sim: np.ndarray, Y_multi: np.ndarray) -> list[dict]:
    """For each class compute AUC + descriptive stats of pos vs neg score dists."""
    stats = []
    for j, cls in enumerate(CLIENT_CLASSES):
        col = sim[:, j]                          # (N,) this class's CLIP score for every image
        pos_mask = Y_multi[:, j] == 1             # photos that are multi-labeled as this class
        neg_mask = ~pos_mask
        pos = col[pos_mask]
        neg = col[neg_mask]
        if len(pos) == 0 or len(neg) == 0:
            auc = float("nan")
        else:
            y_lbl = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            scores = np.concatenate([pos, neg])
            auc = float(roc_auc_score(y_lbl, scores))
        stats.append({
            "class": cls,
            "n_pos": int(pos_mask.sum()),
            "n_neg": int(neg_mask.sum()),
            "pos_mean": float(pos.mean()) if len(pos) else float("nan"),
            "pos_std": float(pos.std()) if len(pos) else float("nan"),
            "neg_mean": float(neg.mean()) if len(neg) else float("nan"),
            "neg_std": float(neg.std()) if len(neg) else float("nan"),
            "auc": auc,
        })
    return stats


def plot_histograms(sim: np.ndarray, Y_multi: np.ndarray, stats: list[dict], out_path: Path) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()
    all_scores_min = sim.min()
    all_scores_max = sim.max()
    bins = np.linspace(all_scores_min, all_scores_max, 30)

    for ax, j, s in zip(axes, range(len(CLIENT_CLASSES)), stats):
        cls = s["class"]
        col = sim[:, j]
        pos = col[Y_multi[:, j] == 1]
        neg = col[Y_multi[:, j] == 0]
        ax.hist(neg, bins=bins, color="#888888", alpha=0.55, label=f"LOO-neg (n={len(neg)})", density=True)
        ax.hist(pos, bins=bins, color="#d95f02", alpha=0.75, label=f"pos (n={len(pos)})", density=True)
        ax.axvline(pos.mean() if len(pos) else 0, color="#d95f02", linestyle="--", linewidth=1)
        ax.axvline(neg.mean() if len(neg) else 0, color="#555555", linestyle="--", linewidth=1)
        ax.set_title(f"{CLIENT_DISPLAY[cls].replace(chr(10), ' ')}  •  AUC={s['auc']:.3f}", fontsize=10)
        ax.set_xlabel("CLIP cosine similarity")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    plt.suptitle(
        "CLIP per-class score separability — client data (multi-label ground truth)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_auc_bar(stats: list[dict], out_path: Path) -> None:
    sns.set_style("whitegrid")
    names = [CLIENT_DISPLAY[s["class"]].replace("\n", " ") for s in stats]
    aucs = [s["auc"] for s in stats]
    order = np.argsort(aucs)[::-1]
    names = [names[i] for i in order]
    aucs = [aucs[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2ca25f" if a >= 0.85 else "#feb24c" if a >= 0.70 else "#de2d26" for a in aucs]
    bars = ax.barh(names, aucs, color=colors)
    ax.set_xlim(0.0, 1.0)
    ax.axvline(0.5, color="#888", linestyle="--", linewidth=1, label="random")
    ax.axvline(0.85, color="#2ca25f", linestyle=":", linewidth=1, label="good threshold")
    ax.set_xlabel("ROC-AUC (LOO within-taxonomy)")
    ax.set_title("CLIP per-class separability ranking", fontweight="bold")
    for bar, a in zip(bars, aucs):
        ax.text(a + 0.01, bar.get_y() + bar.get_height() / 2, f"{a:.2f}", va="center", fontsize=9)
    ax.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[clip-sep] device = {device}", flush=True)

    print("[clip-sep] running CLIP on 98 images ...", flush=True)
    sim, y_true_single, df = compute_similarity_matrix(device)
    print(f"[clip-sep] similarity matrix shape = {sim.shape}, range = [{sim.min():.3f}, {sim.max():.3f}]", flush=True)

    sim_path = REPO / "checkpoints" / "client_clip_similarity.npz"
    np.savez(sim_path, similarity=sim, y_true_single=y_true_single)
    print(f"[clip-sep] saved {sim_path}", flush=True)

    print("[clip-sep] building multi-label ground truth ...", flush=True)
    Y_multi = build_multilabel_truth(df)
    n_multi = (Y_multi.sum(axis=1) > 1).sum()
    print(f"[clip-sep] {n_multi} photos have >1 valid label", flush=True)

    print("[clip-sep] computing per-class stats ...", flush=True)
    stats = per_class_stats(sim, Y_multi)
    for s in stats:
        print(f"  {s['class']:<25}  pos={s['n_pos']:>3} neg={s['n_neg']:>3}  "
              f"pos_mean={s['pos_mean']:.3f} neg_mean={s['neg_mean']:.3f}  AUC={s['auc']:.3f}")

    summary_path = REPO / "checkpoints" / "clip_separability_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    print(f"[clip-sep] saved {summary_path}", flush=True)

    fig_path = REPO / "reports" / "figures" / "clip_separability.png"
    plot_histograms(sim, Y_multi, stats, fig_path)
    print(f"[clip-sep] saved {fig_path}", flush=True)

    bar_path = REPO / "reports" / "figures" / "clip_separability_auc.png"
    plot_auc_bar(stats, bar_path)
    print(f"[clip-sep] saved {bar_path}", flush=True)

    print("[clip-sep] done", flush=True)


if __name__ == "__main__":
    main()
