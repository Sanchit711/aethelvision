"""AethelOBB: Complete End-to-End Oriented Bounding Box Architecture Assembly.

Assembles CSPBackbone, PAFPN Neck, and AethelOBBHead into an integrated
neural network predicting rotated bounding boxes (cx, cy, w, h, angle) in real time.
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
from aethel.models.head.obb import AethelOBBHead
from aethel.models.neck.pafpn import PAFPN
from aethel.registry.registry import MODEL_REGISTRY
from aethel.utils.logger import get_logger

logger = get_logger("aethel.models.obb")


@MODEL_REGISTRY.register("AethelOBB")
class AethelOBBModel(BaseModel):
    """Integrated Real-Time Oriented Bounding Box Neural Network Model."""

    def __init__(
        self,
        num_classes: int = 15,
        scale: str = "medium",
        depth_multiple: float = 0.67,
        width_multiple: float = 0.75,
        reg_max: int = 16,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)
        self.num_classes = num_classes
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

        # 3. Oriented Bounding Box Head
        self.head = AethelOBBHead(
            num_classes=num_classes,
            in_channels_list=self.neck.out_channels,
            reg_max=reg_max,
        )

        logger.info(
            f"Initialized AethelOBB-{scale.upper()} (classes={num_classes})"
        )

    def forward(self, x: Any) -> Any:
        if torch is None:
            return {"cls": None, "box": None, "angle": None}

        # Backbone -> Neck -> OBB Head
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck((p3, p4, p5))
        return self.head((n3, n4, n5))


# Register helper variant builders
@MODEL_REGISTRY.register("aethel_obb_small")
def build_aethel_obb_small(num_classes: int = 15, **kwargs: Any) -> AethelOBBModel:
    return AethelOBBModel(num_classes=num_classes, scale="small", depth_multiple=0.33, width_multiple=0.50, **kwargs)

@MODEL_REGISTRY.register("aethel_obb_medium")
def build_aethel_obb_medium(num_classes: int = 15, **kwargs: Any) -> AethelOBBModel:
    return AethelOBBModel(num_classes=num_classes, scale="medium", depth_multiple=0.67, width_multiple=0.75, **kwargs)


__all__ = ["AethelOBBModel", "build_aethel_obb_small", "build_aethel_obb_medium"]
