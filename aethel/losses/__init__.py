"""Aethel Losses Submodule."""

from aethel.losses.base import BaseLoss
from aethel.losses.composite import (
    BoundingBoxLoss,
    ClassificationLoss,
    CompositeDetectionLoss,
    DetectionLoss,
    LossComposer,
)
from aethel.losses.dfl import DistributionFocalLoss
from aethel.losses.focal import BCEWithLogitsLoss, FocalLoss
from aethel.losses.iou import BboxIoULoss
from aethel.losses.obb import AethelOBBLoss, ContinuousAngleLoss, ProbIoULoss
from aethel.losses.pose import AethelPoseLoss, KeypointL1Loss, OKSLoss
from aethel.losses.segment import AethelSegmentLoss, DiceLoss, MaskBCELoss

__all__ = [
    "BaseLoss",
    "BboxIoULoss",
    "FocalLoss",
    "BCEWithLogitsLoss",
    "DistributionFocalLoss",
    "CompositeDetectionLoss",
    "DetectionLoss",
    "ClassificationLoss",
    "BoundingBoxLoss",
    "LossComposer",
    "OKSLoss",
    "KeypointL1Loss",
    "AethelPoseLoss",
    "DiceLoss",
    "MaskBCELoss",
    "AethelSegmentLoss",
    "ContinuousAngleLoss",
    "ProbIoULoss",
    "AethelOBBLoss",
]
