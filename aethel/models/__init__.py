"""Aethel Models Submodule."""

from aethel.models.backbone.csp import CSPBackbone
from aethel.models.backbone.stem import DetectorStem
from aethel.models.base import BaseModel
from aethel.models.detector import SCALE_FACTORS, AethelDetector
from aethel.models.head.decoupled import DecoupledDetectHead
from aethel.models.aethelnet import (
    AETHELNET_PRESETS,
    AethelNet,
    build_aethelnet_large,
    build_aethelnet_medium,
    build_aethelnet_small,
    build_aethelnet_tiny,
    build_aethelnet_xlarge,
    build_model,
)
from aethel.models.neck.pafpn import PAFPN
from aethel.models.obb import (
    AethelOBBModel,
    build_aethel_obb_medium,
    build_aethel_obb_small,
)
from aethel.models.pose import (
    AethelPoseModel,
    build_aethel_pose_medium,
    build_aethel_pose_small,
)
from aethel.models.segment import (
    AethelSegmentModel,
    build_aethel_segment_medium,
    build_aethel_segment_small,
)
from aethel.models.transformer import (
    AethelTransformer,
    build_transformer_base,
    build_transformer_small,
)

__all__ = [
    "BaseModel",
    "AethelDetector",
    "AethelNet",
    "AethelTransformer",
    "AethelPoseModel",
    "AethelSegmentModel",
    "AethelOBBModel",
    "build_transformer_base",
    "build_transformer_small",
    "build_model",
    "build_aethelnet_tiny",
    "build_aethelnet_small",
    "build_aethelnet_medium",
    "build_aethelnet_large",
    "build_aethelnet_xlarge",
    "build_aethel_pose_small",
    "build_aethel_pose_medium",
    "build_aethel_segment_small",
    "build_aethel_segment_medium",
    "build_aethel_obb_small",
    "build_aethel_obb_medium",
    "AETHELNET_PRESETS",
    "CSPBackbone",
    "DetectorStem",
    "PAFPN",
    "DecoupledDetectHead",
    "SCALE_FACTORS",
]
