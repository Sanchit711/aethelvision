"""Activation Function Layers and Factory for Aethel Detector.

Provides activation layer wrappers and a factory function for SiLU, ReLU, GELU,
Hardswish, LeakyReLU, and Identity activations.
"""

from typing import Optional, Union

import torch
import torch.nn as nn


def get_activation(
    act_type: Union[str, bool, nn.Module, None] = True
) -> nn.Module:
    """Activation layer factory function.

    Args:
        act_type: Activation specification.
            - True / 'silu' / 'swish': Returns nn.SiLU(inplace=True)
            - 'relu': Returns nn.ReLU(inplace=True)
            - 'gelu': Returns nn.GELU()
            - 'hardswish': Returns nn.Hardswish(inplace=True)
            - 'leaky': Returns nn.LeakyReLU(0.1, inplace=True)
            - False / None / 'identity': Returns nn.Identity()
            - nn.Module: Returns act_type instance as-is.

    Returns:
        Instantiated nn.Module activation layer.

    Raises:
        ValueError: If an unknown activation type string is provided.
    """
    if isinstance(act_type, nn.Module):
        return act_type

    if act_type is True or act_type in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    elif act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "hardswish":
        return nn.Hardswish(inplace=True)
    elif act_type == "leaky":
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type is False or act_type is None or act_type == "identity":
        return nn.Identity()

    raise ValueError(f"Unsupported activation layer type: {act_type}")
