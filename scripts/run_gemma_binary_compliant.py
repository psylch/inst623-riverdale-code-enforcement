"""Run per-class binary verification for Gemma 4 on the 20 synthetic compliant images.

These images contain NO violations — used to measure False Positive Rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.client_data import CLIENT_CLASSES, CLIENT_DESCRIPTIONS
from src.zeroshot import load_gemma_vlm
from src.binary_prompt import gemma_binary_verify


COMPLIANT_DIR = REPO / "data" / "synthetic" / "code-enforcement-compliant"


def build_compliant_catalogue() -> pd.DataFrame:
    rows = []
    for f in sorted(COMPLIANT_DIR.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            rows.append({"path": str(f), "label": "compliant", "label_idx": 0})
    return pd.DataFrame(rows)


def main() -> None:
    df = build_compliant_catalogue()
    print(f"[run] {len(df)} compliant images × {len(CLIENT_CLASSES)} classes = {len(df) * len(CLIENT_CLASSES)} calls", flush=True)

    print("[run] loading gemma-4-e4b-it-4bit ...", flush=True)
    model, processor, config = load_gemma_vlm("mlx-community/gemma-4-e4b-it-4bit")
    print("[run] model ready", flush=True)

    stream_path = REPO / "checkpoints" / "compliant_gemma4_binary_stream.jsonl"
    print(f"[run] streaming to {stream_path}", flush=True)

    out = gemma_binary_verify(
        model, processor, config,
        image_paths=df["path"].tolist(),
        labels=df["label_idx"].tolist(),  # placeholder — all images are negative
        classes=CLIENT_CLASSES,
        descriptions=CLIENT_DESCRIPTIONS,
        max_tokens=60,
        stream_path=stream_path,
    )

    ckpt = REPO / "checkpoints" / "compliant_gemma4_binary.npz"
    np.savez(
        ckpt,
        scores=out["scores"],
        answers=out["answers"],
        parse_ok=out["parse_ok"],
        image_paths=np.array(df["path"].tolist()),
    )
    print(f"[run] saved {ckpt}", flush=True)

    # === FPR analysis (true label is 'no' for all 20×9 cells) ===
    answers = out["answers"]  # shape (20, 9), 1=yes, 0=no
    parse_ok = out["parse_ok"]

    print("\n" + "=" * 70)
    print("FALSE POSITIVE RATE (Stage 1 — clean houses, all true labels = no)")
    print("=" * 70)
    print(f"{'class':<28}{'FPR':>10}{'yes':>6}{'parsed':>9}")
    for k, cls in enumerate(CLIENT_CLASSES):
        valid = parse_ok[:, k]
        ans = answers[valid, k]
        n = len(ans)
        yes = int((ans == 1).sum())
        fpr = yes / n if n else float("nan")
        print(f"{cls:<28}{fpr:>9.1%}{yes:>6}{n:>9}")

    overall_valid = parse_ok.flatten()
    overall_ans = answers.flatten()[overall_valid]
    overall_fpr = (overall_ans == 1).mean()
    print(f"\nOverall FPR: {overall_fpr:.1%} ({int((overall_ans == 1).sum())}/{len(overall_ans)} cells)")
    print(f"Parse success: {parse_ok.mean():.1%}")


if __name__ == "__main__":
    main()
