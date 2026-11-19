"""Losses Subpackage for Aethel Detector."""

try:
    from aethel_detector.losses.assignment import TargetAssigner
    from aethel_detector.losses.composite import CompositeDetectionLoss
    from aethel_detector.losses.dfl import DistributionFocalLoss
    from aethel_detector.losses.focal import BCEClassificationLoss, FocalLoss
    from aethel_detector.losses.iou import BboxIoULoss, CIoULoss, GIoULoss, IoULoss, bbox_iou
    from aethel_detector.losses.objectness import ObjectnessLoss
except Exception:
    TargetAssigner = None  # type: ignore
    CompositeDetectionLoss = None  # type: ignore
    DistributionFocalLoss = None  # type: ignore
    BCEClassificationLoss = FocalLoss = None  # type: ignore
    BboxIoULoss = CIoULoss = GIoULoss = IoULoss = bbox_iou = None  # type: ignore
    ObjectnessLoss = None  # type: ignore

__all__ = [
    "bbox_iou",
    "IoULoss",
    "GIoULoss",
    "CIoULoss",
    "BboxIoULoss",
    "DistributionFocalLoss",
    "BCEClassificationLoss",
    "FocalLoss",
    "ObjectnessLoss",
    "TargetAssigner",
    "CompositeDetectionLoss",
]
