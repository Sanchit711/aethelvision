"""Aethel Detector Object Detection Architecture Assembly.

Assembles Backbone (CSPBackbone), Neck (PAFPN), and Detection Head (DecoupledDetectHead)
into a unified, executable neural network architecture.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

from aethel.layers.init import initialize_weights
from aethel.models.backbone.csp import CSPBackbone
from aethel.models.base import BaseModel
from aethel.models.head.decoupled import DecoupledDetectHead
from aethel.models.neck.pafpn import PAFPN
from aethel.registry.registry import MODEL_REGISTRY
from aethel.utils.exceptions import ModelException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.models.detector")

SCALE_FACTORS = {
    "tiny": (0.33, 0.25),
    "small": (0.33, 0.50),
    "medium": (0.67, 0.75),
    "large": (1.00, 1.00),
    "xlarge": (1.33, 1.25),
}


@MODEL_REGISTRY.register("AethelDetector")
class AethelDetector(BaseModel):
    """Unified Aethel Detector Neural Network Model Assembly."""

    def __init__(
        self,
        num_classes: int = 80,
        scale: str = "large",
        depth_multiple: Optional[float] = None,
        width_multiple: Optional[float] = None,
        reg_max: int = 16,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize AethelDetector."""
        super().__init__(config=config)
        self.num_classes = num_classes
        self.scale = scale
        self.config = config or {}

        if depth_multiple is None or width_multiple is None:
            scale_key = scale.lower()
            if scale_key in SCALE_FACTORS:
                d_mult, w_mult = SCALE_FACTORS[scale_key]
            else:
                d_mult, w_mult = 1.0, 1.0
            depth_multiple = depth_multiple if depth_multiple is not None else d_mult
            width_multiple = width_multiple if width_multiple is not None else w_mult

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.reg_max = reg_max

        # 1. Backbone
        self.backbone = CSPBackbone(
            in_channels=3,
            depth_multiple=self.depth_multiple,
            width_multiple=self.width_multiple,
        )

        # 2. Neck
        self.neck = PAFPN(
            in_channels_list=self.backbone.out_channels,
            depth_multiple=self.depth_multiple,
        )

        # 3. Head
        self.head = DecoupledDetectHead(
            num_classes=self.num_classes,
            in_channels_list=self.neck.out_channels,
            reg_max=self.reg_max,
        )

        initialize_weights(self, mode="kaiming")
        logger.info(f"Initialized AethelDetector (scale='{scale}', depth={self.depth_multiple}, width={self.width_multiple}, num_classes={num_classes})")

    def forward(self, x: Any, **kwargs: Any) -> List[Any]:
        """Forward pass."""
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck((p3, p4, p5))
        raw_outputs = self.head((n3, n4, n5))
        return raw_outputs

    def load_weights(self, weights_path: Union[str, Path], strict: bool = True) -> None:
        """Load model state dict weights from file checkpoint."""
        path = Path(weights_path)
        if not path.exists():
            raise ModelException(message=f"Weights checkpoint file not found: {path}")

        if torch is None:
            logger.info(f"Loaded mock weights from {path} (Fallback mode)")
            return

        try:
            checkpoint = safe_torch_load(path, map_location="cpu", allow_pickle=True)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, strict=strict)
            logger.info(f"Loaded weights from {path} (strict={strict})")
        except Exception as e:
            raise ModelException(message=f"Failed to load checkpoint at {path}", details=str(e))

    def save_weights(self, output_path: Union[str, Path]) -> None:
        """Save model state dict weights to disk checkpoint."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if torch is None:
            mock_payload = {
                "num_classes": self.num_classes,
                "depth_multiple": self.depth_multiple,
                "width_multiple": self.width_multiple,
                "reg_max": self.reg_max,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(mock_payload, f)
            logger.info(f"Saved mock weights checkpoint to {path}")
            return

        try:
            payload = {
                "model_state_dict": self.state_dict(),
                "num_classes": self.num_classes,
                "depth_multiple": self.depth_multiple,
                "width_multiple": self.width_multiple,
                "reg_max": self.reg_max,
            }
            torch.save(payload, path)
            logger.info(f"Saved model weights to {path}")
        except Exception as e:
            raise ModelException(message=f"Failed to save checkpoint to {path}", details=str(e))


__all__ = ["AethelDetector", "SCALE_FACTORS"]
