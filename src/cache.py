"""Checkpoint and caching utilities.

Saves intermediate results to FinalProject/checkpoints/ so crashed runs
can resume without re-computing everything.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


# ── predictions cache ───────────────────────────────────────────


def save_predictions(name: str, y_true, y_pred, y_probs):
    np.savez(CKPT_DIR / f"{name}_preds.npz", y_true=y_true, y_pred=y_pred, y_probs=y_probs)


def load_predictions(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    p = CKPT_DIR / f"{name}_preds.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["y_true"], d["y_pred"], d["y_probs"]


# ── model checkpoint ────────────────────────────────────────────


def save_model(name: str, model: nn.Module, result=None):
    torch.save(model.state_dict(), CKPT_DIR / f"{name}_model.pt")
    if result is not None:
        info = {
            "best_val_acc": result.best_val_acc,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "val_accs": result.val_accs,
        }
        (CKPT_DIR / f"{name}_result.json").write_text(json.dumps(info))


def load_model(name: str, model: nn.Module) -> bool:
    """Load state dict into model. Returns True if checkpoint existed."""
    p = CKPT_DIR / f"{name}_model.pt"
    if not p.exists():
        return False
    model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    return True


def load_result(name: str):
    """Load TrainResult from JSON. Returns dict or None."""
    p = CKPT_DIR / f"{name}_result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_result(name: str, result):
    """Save TrainResult independently (also called by save_model)."""
    info = {
        "best_val_acc": result.best_val_acc,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "train_losses": result.train_losses,
        "val_losses": result.val_losses,
        "val_accs": result.val_accs,
    }
    (CKPT_DIR / f"{name}_result.json").write_text(json.dumps(info))


def has_checkpoint(name: str) -> bool:
    return (CKPT_DIR / f"{name}_model.pt").exists() and (CKPT_DIR / f"{name}_preds.npz").exists()


# ── dataframe cache ─────────────────────────────────────────────


def save_splits(train_df, val_df, test_df):
    train_df.to_csv(CKPT_DIR / "train.csv", index=False)
    val_df.to_csv(CKPT_DIR / "val.csv", index=False)
    test_df.to_csv(CKPT_DIR / "test.csv", index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    paths = [CKPT_DIR / f"{s}.csv" for s in ("train", "val", "test")]
    if not all(p.exists() for p in paths):
        return None
    return tuple(pd.read_csv(p) for p in paths)
