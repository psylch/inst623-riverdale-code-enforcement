"""Training loop and LP-FT pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.01
    patience: int = 5
    warmup_epochs: int = 2
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class TrainResult:
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    best_epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one epoch. If optimizer is None, runs in eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: str = "mps",
    label: str = "",
) -> tuple[nn.Module, TrainResult]:
    """Train with cosine annealing + warmup + early stopping. Returns best model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # separate LR for head vs backbone
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.lr * 0.1})
    param_groups.append({"params": head_params, "lr": config.lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=config.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[config.warmup_epochs])

    result = TrainResult()
    best_state = None
    wait = 0

    pbar = tqdm(range(1, config.epochs + 1), desc=label or "Training")
    for epoch in pbar:
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.val_accs.append(val_acc)

        pbar.set_postfix(tl=f"{train_loss:.3f}", vl=f"{val_loss:.3f}", va=f"{val_acc:.3f}")

        if val_acc > result.best_val_acc:
            result.best_val_acc = val_acc
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, result
