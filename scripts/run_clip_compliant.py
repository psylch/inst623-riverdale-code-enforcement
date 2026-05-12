"""CLIP zero-shot similarity over the 20 synthetic compliant images.

Saves per-image (N, K) similarity matrix so we can compute:
  - CLIP-only top-k FPR (top-k flagged → false alarms by definition)
  - CLIP→Gemma cascade FPR (top-k filtered through Gemma binary)
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from PIL import Image

from src.client_data import CLIENT_CLASSES, CLIENT_CLIP_PROMPTS
from src.zeroshot import load_clip, build_text_features


COMPLIANT_DIR = REPO / "data" / "synthetic" / "code-enforcement-compliant"


def main() -> None:
    paths = sorted(p for p in COMPLIANT_DIR.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
    print(f"[run] {len(paths)} compliant images, {len(CLIENT_CLASSES)} classes")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[run] device={device}")
    model, preprocess, tokenizer = load_clip(device=device)
    text_feats = build_text_features(model, tokenizer, CLIENT_CLASSES, CLIENT_CLIP_PROMPTS, device=device)

    sims = []
    with torch.no_grad():
        for p in paths:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = (100.0 * feat @ text_feats.T).softmax(dim=-1).cpu().numpy()[0]
            sims.append(sim)
    sims = np.stack(sims)  # (20, 9)

    out = REPO / "checkpoints" / "compliant_clip_sims.npz"
    np.savez(out, sims=sims, image_paths=np.array([str(p) for p in paths]))
    print(f"[run] saved {out}")

    # Top-k CLIP "flag" rate (would-be FPR if CLIP alone were used as classifier)
    print("\nCLIP top-k 'flag' rate per class (treating top-k membership as positive):")
    for k in (3, 5):
        topk = np.argsort(-sims, axis=1)[:, :k]
        flagged = np.zeros_like(sims, dtype=int)
        for i, ks in enumerate(topk):
            for j in ks:
                flagged[i, j] = 1
        print(f"\n  k={k}  (per image, {k}/9 classes flagged → {k/9*100:.0f}% baseline)")
        for j, c in enumerate(CLIENT_CLASSES):
            r = flagged[:, j].mean()
            print(f"    {c:<28} flag rate = {r*100:>5.1f}%  ({flagged[:, j].sum()}/20)")
        print(f"  overall: {flagged.mean()*100:.1f}%  (= k/9 by construction)")

    # === Cascade: CLIP top-k AND Gemma said yes ===
    gem_npz = REPO / "checkpoints" / "compliant_gemma4_binary.npz"
    if gem_npz.exists():
        gem = np.load(gem_npz)
        gem_yes = (gem["answers"] == 1).astype(int)  # (20, 9)
        print(f"\n=== CASCADE FPR (CLIP top-k → Gemma binary verify) ===")
        for k in (3, 5):
            topk = np.argsort(-sims, axis=1)[:, :k]
            in_topk = np.zeros_like(sims, dtype=int)
            for i, ks in enumerate(topk):
                for j in ks:
                    in_topk[i, j] = 1
            cascade_yes = in_topk * gem_yes  # both conditions
            print(f"\n  k={k}")
            for j, c in enumerate(CLIENT_CLASSES):
                fp = cascade_yes[:, j].sum()
                print(f"    {c:<28}  cascade FPR = {fp/20*100:>4.1f}%  ({fp}/20)")
            print(f"  overall: {cascade_yes.mean()*100:.1f}%  ({cascade_yes.sum()}/180)")


if __name__ == "__main__":
    main()
