"""Neck Subpackage for Aethel Detector."""

from aethel_detector.neck.fpn import FeaturePyramidNetwork
from aethel_detector.neck.fusion import FeatureFusionBlock, UpsampleBlock
from aethel_detector.neck.pafpn import PAFPN

__all__ = [
    "FeaturePyramidNetwork",
    "PAFPN",
    "UpsampleBlock",
    "FeatureFusionBlock",
]
