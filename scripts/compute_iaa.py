"""Compute inter-annotator agreement (Cohen's kappa) for the 3-rater labels.

Reads merged_3labelers.csv (Fechi / Niping / Jake × 9 categories, values
-1 / 0 / 1), binarizes unsure -> no, then for each category reports:

- pairwise Cohen's kappa (F-N, F-J, N-J) and their mean
- Landis & Koch tier (almost perfect / substantial / moderate / fair / slight)
- the SOP deployment recommendation that follows from the tier

Usage:
    uv run python scripts/compute_iaa.py
    uv run python scripts/compute_iaa.py --csv path/to/merged_3labelers.csv
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

RATERS = ["Fechi", "Niping", "Jake"]
DEFAULT_CSV = Path("data/client-data/labeling/human/merged_3labelers.csv")


def landis_koch_tier(kappa: float) -> tuple[str, str]:
    """Return (tier_label, deployment_recommendation)."""
    if kappa >= 0.81:
        return "almost perfect", "Full automation candidate"
    if kappa >= 0.61:
        return "substantial", "Full automation candidate"
    if kappa >= 0.41:
        return "moderate", "AI-assisted tier"
    if kappa >= 0.21:
        return "fair", "AI-assisted tier"
    return "slight", "Hold — fix SOP before deploying"


def compute_category_kappa(df: pd.DataFrame, category: str) -> dict:
    cols = {r: f"{r}__{category}" for r in RATERS}
    sub = df[list(cols.values())].copy()
    # -1 (unsure) -> 0 (no), per project methodology
    sub = sub.replace(-1, 0).astype(int)

    pairwise = {}
    for a, b in combinations(RATERS, 2):
        k = cohen_kappa_score(sub[cols[a]], sub[cols[b]])
        pairwise[f"{a[0]}-{b[0]}"] = k
    mean_k = sum(pairwise.values()) / len(pairwise)
    tier, recommendation = landis_koch_tier(mean_k)
    return {
        "category": category,
        **pairwise,
        "mean_kappa": mean_k,
        "tier": tier,
        "recommendation": recommendation,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="Path to merged_3labelers.csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    categories = sorted({c.split("__", 1)[1] for c in df.columns if "__" in c})

    rows = [compute_category_kappa(df, c) for c in categories]
    rows.sort(key=lambda r: r["mean_kappa"], reverse=True)

    print(f"\nInter-annotator agreement — {args.csv}")
    print(f"Raters: {', '.join(RATERS)}   n_images: {len(df)}\n")
    header = f"{'category':<25} {'F-N':>6} {'F-J':>6} {'N-J':>6} {'mean κ':>8}  {'tier':<16} recommendation"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['category']:<25} "
              f"{r['F-N']:>6.2f} {r['F-J']:>6.2f} {r['N-J']:>6.2f} "
              f"{r['mean_kappa']:>8.3f}  {r['tier']:<16} {r['recommendation']}")
    print()


if __name__ == "__main__":
    main()
