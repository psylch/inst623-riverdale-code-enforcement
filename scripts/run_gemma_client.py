"""Standalone Gemma 4 zero-shot run over the client data.

Runs independently of the notebook so progress can be monitored by tailing
the JSONL stream. Saves final predictions in the same npz format the
notebook cache loader expects, so after this finishes the notebook will
load from cache on its next execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow `python FinalProject/scripts/run_gemma_client.py` from repo root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.client_data import build_client_catalogue, CLIENT_CLASSES, CLIENT_DESCRIPTIONS
from src.zeroshot import load_gemma_vlm, gemma_zero_shot
from src.cache import save_predictions


def main() -> None:
    df = build_client_catalogue()
    print(f"[run] catalogue: {len(df)} images, {df['label'].nunique()} classes", flush=True)

    print("[run] loading gemma-4-e4b-it-4bit ...", flush=True)
    model, processor, config = load_gemma_vlm("mlx-community/gemma-4-e4b-it-4bit")
    print("[run] model ready", flush=True)

    stream_path = REPO / "checkpoints" / "client_gemma4_stream.jsonl"
    print(f"[run] streaming per-image results to {stream_path}", flush=True)

    y_true, y_pred, y_probs, raw = gemma_zero_shot(
        model, processor, config,
        image_paths=df["path"].tolist(),
        labels=df["label_idx"].tolist(),
        classes=CLIENT_CLASSES,
        descriptions=CLIENT_DESCRIPTIONS,
        max_tokens=80,
        stream_path=stream_path,
    )

    save_predictions("client_gemma4", y_true, y_pred, y_probs)
    raw_path = REPO / "checkpoints" / "client_gemma4_raw.txt"
    raw_path.write_text("\n---\n".join(raw))

    acc = (y_true == y_pred).mean()
    print(f"[run] done. top-1 accuracy = {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
