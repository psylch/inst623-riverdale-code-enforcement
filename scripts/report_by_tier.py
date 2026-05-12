"""Render the κ-tier performance table from CLIENT_REPORT §5.1.

Joins inter-annotator agreement (Cohen's kappa, computed from the 3-rater
labels) with cascade k=5 per-class metrics (CLIP top-5 → Gemma binary
verification, evaluated against folder-name ground truth on 98 violation
images), then groups categories by Landis & Koch tier and macro-averages
precision / recall / F1 within each tier.

Inputs (must exist):
    - checkpoints/client_gemma4_binary_stream.jsonl  (from run_gemma_binary.py)
    - checkpoints/client_clip_similarity.npz         (from run_clip_separability.py)
    - data/client-data/labeling/human/merged_3labelers.csv  (3-rater labels)

Usage:
    uv run python scripts/report_by_tier.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.client_data import CLIENT_CLASSES, CLIENT_DISPLAY
from scripts.analyze_binary_results import (
    build_multilabel_truth,
    load_binary_from_stream,
)
from scripts.compute_iaa import compute_category_kappa, landis_koch_tier
from src.client_data import build_client_catalogue


TIER_ORDER = ["almost perfect", "substantial", "moderate", "fair", "slight"]
TIER_RECOMMENDATION = {
    "almost perfect": "Full automation",
    "substantial":    "Full automation",
    "moderate":       "AI-assisted, inspector verifies",
    "fair":           "AI-assisted",
    "slight":         "Hold until SOP provided",
}
TIER_RANGE = {
    "almost perfect": "≥ 0.81",
    "substantial":    "0.61–0.80",
    "moderate":       "0.41–0.60",
    "fair":           "0.21–0.40",
    "slight":         "< 0.21",
}


def cascade_per_class_at_k(
    clip_sim: np.ndarray,
    binary_scores: np.ndarray,
    Y_multi: np.ndarray,
    k: int,
) -> list[dict]:
    """For each class, compute precision/recall/F1 of cascade(CLIP top-k → Gemma)."""
    N, K = clip_sim.shape
    top_k_idx = np.argsort(-clip_sim, axis=1)[:, :k]
    y_pred = np.zeros((N, K), dtype=bool)
    for i in range(N):
        for j in top_k_idx[i]:
            if binary_scores[i, j] >= 0.5:
                y_pred[i, j] = True

    y_true = Y_multi.astype(bool)
    out = []
    for j, cls in enumerate(CLIENT_CLASSES):
        tp = int((y_pred[:, j] & y_true[:, j]).sum())
        fp = int((y_pred[:, j] & ~y_true[:, j]).sum())
        fn = int((~y_pred[:, j] & y_true[:, j]).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * p * r / (p + r) if (p + r) > 0 and np.isfinite(p) and np.isfinite(r) else float("nan")
        out.append({"class": cls, "precision": p, "recall": r, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn})
    return out


def main() -> None:
    K = len(CLIENT_CLASSES)

    # 1) κ per category
    labels_csv = REPO / "data" / "client-data" / "labeling" / "human" / "merged_3labelers.csv"
    if not labels_csv.exists():
        raise SystemExit(f"missing {labels_csv}; cannot compute κ tiers")
    labels_df = pd.read_csv(labels_csv)
    kappa_by_class = {}
    for cls in CLIENT_CLASSES:
        row = compute_category_kappa(labels_df, cls)
        kappa_by_class[cls] = row["mean_kappa"]

    # 2) cascade k=5 per-class P/R/F1
    df = build_client_catalogue()
    N = len(df)
    stream = REPO / "checkpoints" / "client_gemma4_binary_stream.jsonl"
    scores, _, _, done = load_binary_from_stream(stream, N, K)
    if int(done.sum()) != N * K:
        print(f"[warn] Gemma stream incomplete ({int(done.sum())}/{N*K}); numbers may be partial")
    Y_multi = build_multilabel_truth(df)
    sim = np.load(REPO / "checkpoints" / "client_clip_similarity.npz")["similarity"]
    per_class = cascade_per_class_at_k(sim, scores, Y_multi, k=5)
    metrics_by_class = {r["class"]: r for r in per_class}

    # 3) per-class table sorted by κ
    rows = []
    for cls in CLIENT_CLASSES:
        k_val = kappa_by_class[cls]
        tier, _ = landis_koch_tier(k_val)
        m = metrics_by_class[cls]
        rows.append({
            "class": cls,
            "kappa": k_val,
            "tier": tier,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
    rows.sort(key=lambda r: -r["kappa"])

    print()
    print("=== Per-class cascade k=5 (sorted by human κ) ===")
    print(f"{'class':<25} {'κ':>6} {'tier':<16} {'P':>5} {'R':>5} {'F1':>5}")
    print("-" * 66)
    for r in rows:
        f = lambda x: f"{x:.2f}" if np.isfinite(x) else "  nan"
        print(f"{r['class']:<25} {r['kappa']:>6.3f} {r['tier']:<16} "
              f"{f(r['precision']):>5} {f(r['recall']):>5} {f(r['f1']):>5}")

    # 4) tier-grouped macro-average
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)

    print()
    print("=== CLIENT_REPORT §5.1 — Performance by κ tier (cascade k=5) ===")
    header = f"{'tier':<16} {'κ range':<11} {'categories':<48} {'macro-P':>8} {'macro-R':>8} {'macro-F1':>9}  recommendation"
    print(header)
    print("-" * len(header))
    for tier in TIER_ORDER:
        members = by_tier.get(tier, [])
        if not members:
            continue
        cats = " · ".join(CLIENT_DISPLAY[r["class"]].replace("\n", " ") for r in members)
        if len(cats) > 46:
            cats = cats[:45] + "…"
        macro_p = float(np.nanmean([r["precision"] for r in members]))
        macro_r = float(np.nanmean([r["recall"]    for r in members]))
        macro_f = float(np.nanmean([r["f1"]        for r in members]))
        print(f"{tier:<16} {TIER_RANGE[tier]:<11} {cats:<48} "
              f"{macro_p:>8.2f} {macro_r:>8.2f} {macro_f:>9.2f}  {TIER_RECOMMENDATION[tier]}")
    print()


if __name__ == "__main__":
    main()
