"""Evaluation metrics and visualization."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from .data import CLASSES


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str = "mps",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get predictions from a trained model. Returns (y_true, y_pred, y_probs)."""
    model.eval().to(device)
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    y_pred = y_probs.argmax(axis=1)
    return y_true, y_pred, y_probs


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    model_name: str = "Model",
    classes: list[str] | None = None,
) -> dict:
    """Print formatted metrics and return them as a dict.

    When `classes` is None, falls back to the proxy taxonomy (`data.CLASSES`).
    Pass a custom list for client/zero-shot evaluations on a different label space.
    """
    cls = list(classes) if classes is not None else CLASSES
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    top3_acc = top_k_accuracy_score(y_true, y_probs, k=3, labels=range(len(cls)))

    print("=" * 55)
    print(f"{model_name.upper()} — TEST SET RESULTS")
    print("=" * 55)
    print(f"  Top-1 Accuracy:  {acc:.4f}")
    print(f"  Top-3 Accuracy:  {top3_acc:.4f}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print()
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    print(classification_report(
        y_true, y_pred,
        labels=present,
        target_names=[cls[i] for i in present],
        digits=3,
        zero_division=0,
    ))

    return {
        "model": model_name,
        "top1_acc": acc,
        "top3_acc": top3_acc,
        "macro_f1": macro_f1,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    ax: plt.Axes | None = None,
    classes: list[str] | None = None,
) -> plt.Axes:
    """Plot a confusion matrix heatmap."""
    cls = list(classes) if classes is not None else CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=range(len(cls)))
    short_names = [c.replace("_", "\n") for c in cls]

    if ax is None:
        figsize = (max(6, len(cls) * 0.7), max(5, len(cls) * 0.6))
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_training_curves(results: dict[str, "TrainResult"], figsize=(12, 4)):
    """Plot loss and accuracy curves for multiple models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for name, r in results.items():
        epochs = range(1, len(r.val_losses) + 1)
        ax1.plot(epochs, r.train_losses, "--", alpha=0.5, label=f"{name} train")
        ax1.plot(epochs, r.val_losses, "-", label=f"{name} val")
        ax2.plot(epochs, r.val_accs, "-", label=name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def comparison_table(metrics_list: list[dict]):
    """Print a formatted comparison table."""
    print("=" * 65)
    print(f"{'Model':<25} {'Top-1 Acc':>10} {'Top-3 Acc':>10} {'Macro F1':>10}")
    print("-" * 65)
    for m in metrics_list:
        print(f"  {m['model']:<23} {m['top1_acc']:>10.4f} {m['top3_acc']:>10.4f} {m['macro_f1']:>10.4f}")
    print("=" * 65)
