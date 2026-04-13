"""Run per-class binary verification for Gemma 4 on the client data.

For each of 98 images, ask Gemma 9 binary yes/no questions — one per
violation class. Saves the (N, K) score matrix to npz and also dumps
the full per-call audit log to a JSONL file that can be tailed live.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from src.client_data import build_client_catalogue, CLIENT_CLASSES, CLIENT_DESCRIPTIONS
from src.zeroshot import load_gemma_vlm
from src.binary_prompt import gemma_binary_verify


def main() -> None:
    df = build_client_catalogue()
    print(f"[run] {len(df)} images × {len(CLIENT_CLASSES)} classes = {len(df) * len(CLIENT_CLASSES)} calls", flush=True)

    print("[run] loading gemma-4-e4b-it-4bit ...", flush=True)
    model, processor, config = load_gemma_vlm("mlx-community/gemma-4-e4b-it-4bit")
    print("[run] model ready", flush=True)

    stream_path = REPO / "checkpoints" / "client_gemma4_binary_stream.jsonl"
    print(f"[run] streaming to {stream_path}", flush=True)

    out = gemma_binary_verify(
        model, processor, config,
        image_paths=df["path"].tolist(),
        labels=df["label_idx"].tolist(),
        classes=CLIENT_CLASSES,
        descriptions=CLIENT_DESCRIPTIONS,
        max_tokens=60,
        stream_path=stream_path,
    )

    ckpt = REPO / "checkpoints" / "client_gemma4_binary.npz"
    np.savez(
        ckpt,
        scores=out["scores"],
        answers=out["answers"],
        parse_ok=out["parse_ok"],
    )
    print(f"[run] saved {ckpt}", flush=True)

    # quick sanity stats
    yes_rate = (out["answers"] == 1).mean()
    parse_rate = out["parse_ok"].mean()
    print(f"[run] yes-rate overall = {yes_rate:.3f}", flush=True)
    print(f"[run] parse success    = {parse_rate:.3f}", flush=True)


if __name__ == "__main__":
    main()
