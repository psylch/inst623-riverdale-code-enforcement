"""Model definitions for DINOv2 and EfficientNetV2."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


def create_model(
    name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create a classification model via timm.

    Supported names:
        - "dinov2"        → vit_base_patch14_dinov2.lvd142m
        - "efficientnetv2" → tf_efficientnetv2_s.in21k_ft_in1k
    """
    model_id = {
        "dinov2": "vit_base_patch14_dinov2.lvd142m",
        "efficientnetv2": "tf_efficientnetv2_s.in21k_ft_in1k",
    }[name]

    model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze classification head
        head = model.get_classifier()
        for param in head.parameters():
            param.requires_grad = True

    return model


def get_transforms(model: nn.Module, is_training: bool = False):
    """Get appropriate transforms for a timm model."""
    data_config = timm.data.resolve_model_data_config(model)
    return timm.data.create_transform(**data_config, is_training=is_training)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
