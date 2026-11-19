"""Normalization Layers and Factory for Aethel Detector.

Provides normalization layer wrappers and factory function for BatchNorm2d,
GroupNorm, LayerNorm, and Identity normalization.
"""

from typing import Optional, Union

import torch
import torch.nn as nn


def get_normalization(
    norm_type: Union[str, bool, nn.Module, None] = True,
    num_features: int = 0,
) -> nn.Module:
    """Normalization layer factory function.

    Args:
        norm_type: Normalization specification.
            - True / 'batch' / 'bn': Returns nn.BatchNorm2d(num_features)
            - 'group': Returns nn.GroupNorm(num_groups=min(32, num_features), num_channels=num_features)
            - 'layer': Returns nn.LayerNorm(num_features)
            - False / None / 'identity': Returns nn.Identity()
            - nn.Module: Returns norm_type instance as-is.
        num_features: Number of channel features for normalization layer instantiation.

    Returns:
        Instantiated nn.Module normalization layer.

    Raises:
        ValueError: If an unknown normalization type string is provided.
    """
    if isinstance(norm_type, nn.Module):
        return norm_type

    if norm_type is True or norm_type in ("batch", "bn"):
        return nn.BatchNorm2d(num_features)
    elif norm_type == "group":
        groups = min(32, num_features) if num_features > 0 else 1
        return nn.GroupNorm(num_groups=groups, num_channels=num_features)
    elif norm_type == "layer":
        return nn.LayerNorm(num_features)
    elif norm_type is False or norm_type is None or norm_type == "identity":
        return nn.Identity()

    raise ValueError(f"Unsupported normalization layer type: {norm_type}")
