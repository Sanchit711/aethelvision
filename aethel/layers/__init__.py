"""Aethel Neural Network Layers Submodule."""

from aethel.layers.attention import CBAM, AttentionHook, ChannelAttention, SpatialAttention
from aethel.layers.blocks import SPPF, Bottleneck, CSPBlock
from aethel.layers.conv import Conv, DWConv, get_activation
from aethel.layers.repconv import RepConv
from aethel.layers.init import initialize_weights
from aethel.registry.registry import LAYER_REGISTRY

# Register layer primitives
LAYER_REGISTRY.register("Conv")(Conv)
LAYER_REGISTRY.register("DWConv")(DWConv)
LAYER_REGISTRY.register("RepConv")(RepConv)
LAYER_REGISTRY.register("Bottleneck")(Bottleneck)
LAYER_REGISTRY.register("CSPBlock")(CSPBlock)
LAYER_REGISTRY.register("SPPF")(SPPF)
LAYER_REGISTRY.register("CBAM")(CBAM)

__all__ = [
    "Conv",
    "DWConv",
    "RepConv",
    "get_activation",
    "AttentionHook",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Bottleneck",
    "CSPBlock",
    "SPPF",
    "initialize_weights",
]
