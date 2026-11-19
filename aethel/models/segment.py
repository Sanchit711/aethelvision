"""AethelSegment: Complete End-to-End Instance Segmentation Architecture Assembly.

Assembles CSPBackbone, PAFPN Neck, ProtoNet, and AethelSegmentHead into an
integrated neural network predicting pixel-level masks and bounding boxes in real time.
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
from aethel.models.head.segment import AethelSegmentHead
from aethel.models.neck.pafpn import PAFPN
from aethel.registry.registry import MODEL_REGISTRY
from aethel.utils.logger import get_logger

logger = get_logger("aethel.models.segment")


@MODEL_REGISTRY.register("AethelSegment")
class AethelSegmentModel(BaseModel):
    """Integrated Real-Time Instance Segmentation Neural Network Model."""

    def __init__(
        self,
        num_classes: int = 80,
        num_protos: int = 32,
        scale: str = "medium",
        depth_multiple: float = 0.67,
        width_multiple: float = 0.75,
        reg_max: int = 16,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)
        self.num_classes = num_classes
        self.num_protos = num_protos
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

        # 3. Instance Segmentation Head (with ProtoNet)
        self.head = AethelSegmentHead(
            num_classes=num_classes,
            num_protos=num_protos,
            in_channels_list=self.neck.out_channels,
            reg_max=reg_max,
        )

        logger.info(
            f"Initialized AethelSegment-{scale.upper()} (classes={num_classes}, prototypes={num_protos})"
        )

    def forward(self, x: Any) -> Any:
        if torch is None:
            return {"protos": None, "cls": None, "box": None, "mask_coeffs": None}

        # Backbone -> Neck -> Segment Head
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck((p3, p4, p5))
        return self.head((n3, n4, n5))


# Register helper variant builders
@MODEL_REGISTRY.register("aethel_segment_small")
def build_aethel_segment_small(num_classes: int = 80, **kwargs: Any) -> AethelSegmentModel:
    return AethelSegmentModel(num_classes=num_classes, scale="small", depth_multiple=0.33, width_multiple=0.50, **kwargs)

@MODEL_REGISTRY.register("aethel_segment_medium")
def build_aethel_segment_medium(num_classes: int = 80, **kwargs: Any) -> AethelSegmentModel:
    return AethelSegmentModel(num_classes=num_classes, scale="medium", depth_multiple=0.67, width_multiple=0.75, **kwargs)


__all__ = ["AethelSegmentModel", "build_aethel_segment_small", "build_aethel_segment_medium"]
