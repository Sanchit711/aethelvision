"""Aethel Head Submodule."""

from aethel.models.head.decoupled import DecoupledDetectHead
from aethel.models.head.pose import (
    AethelPoseHead,
    COCO_KEYPOINTS,
    COCO_SKELETON,
    decode_keypoints,
)
from aethel.models.head.segment import AethelSegmentHead, ProtoNet, assemble_masks
from aethel.models.head.obb import AethelOBBHead, decode_angle, obb_to_corners
from aethel.registry.registry import HEAD_REGISTRY

HEAD_REGISTRY.register("DecoupledDetectHead")(DecoupledDetectHead)
HEAD_REGISTRY.register("AethelPoseHead")(AethelPoseHead)
HEAD_REGISTRY.register("AethelSegmentHead")(AethelSegmentHead)
HEAD_REGISTRY.register("AethelOBBHead")(AethelOBBHead)

__all__ = [
    "DecoupledDetectHead",
    "AethelPoseHead",
    "AethelSegmentHead",
    "AethelOBBHead",
    "decode_angle",
    "decode_keypoints",
    "obb_to_corners",
    "ProtoNet",
    "assemble_masks",
    "COCO_KEYPOINTS",
    "COCO_SKELETON",
]
