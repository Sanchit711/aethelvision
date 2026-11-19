"""AethelPose: Complete End-to-End Pose Estimation Architecture Assembly.

Assembles CSPBackbone, PAFPN Neck, and AethelPoseHead into an integrated
neural network estimating 17 human keypoints and bounding boxes in real time.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

from aethel.models.backbone.csp import CSPBackbone
from aethel.models.base import BaseModel
from aethel.models.head.pose import AethelPoseHead, COCO_KEYPOINTS, COCO_SKELETON
from aethel.models.neck.pafpn import PAFPN
from aethel.registry.registry import MODEL_REGISTRY
from aethel.utils.logger import get_logger

logger = get_logger("aethel.models.pose")


@MODEL_REGISTRY.register("AethelPose")
class AethelPoseModel(BaseModel):
    """Integrated Real-Time Pose Estimation Neural Network Model."""

    def __init__(
        self,
        num_classes: int = 1,
        num_keypoints: int = 17,
        scale: str = "medium",
        depth_multiple: float = 0.67,
        width_multiple: float = 0.75,
        reg_max: int = 16,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.scale = scale
        self.reg_max = reg_max

        # 1. Backbone
        self.backbone = CSPBackbone(
            in_channels=3,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
        )

        # 2. Neck
        self.neck = PAFPN(
            in_channels_list=self.backbone.out_channels,
            depth_multiple=depth_multiple,
        )

        # 3. Pose Estimation Head
        self.head = AethelPoseHead(
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            in_channels_list=self.neck.out_channels,
            reg_max=reg_max,
        )

        logger.info(
            f"Initialized AethelPose-{scale.upper()} (keypoints={num_keypoints})"
        )

    def forward(self, x: Any) -> Any:
        if torch is None:
            return {"cls": None, "box": None, "kpt": None}

        # Backbone -> Neck -> Pose Head
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck((p3, p4, p5))
        return self.head((n3, n4, n5))


# Register helper variant builders
@MODEL_REGISTRY.register("aethel_pose_small")
def build_aethel_pose_small(num_classes: int = 1, **kwargs: Any) -> AethelPoseModel:
    return AethelPoseModel(num_classes=num_classes, scale="small", depth_multiple=0.33, width_multiple=0.50, **kwargs)

@MODEL_REGISTRY.register("aethel_pose_medium")
def build_aethel_pose_medium(num_classes: int = 1, **kwargs: Any) -> AethelPoseModel:
    return AethelPoseModel(num_classes=num_classes, scale="medium", depth_multiple=0.67, width_multiple=0.75, **kwargs)


__all__ = ["AethelPoseModel", "build_aethel_pose_small", "build_aethel_pose_medium"]
