"""AethelNet First Official Production Model Family for Aethel Vision Framework.

Provides AethelNet (Tiny, Small, Medium, Large, XLarge) architecture variants
optimized for high FPS, low memory footprint, and high object detection accuracy.
"""

from typing import Any, Dict, Optional, Union

from aethel.models.detector import AethelDetector
from aethel.registry.registry import MODEL_REGISTRY
from aethel.utils.exceptions import ModelException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.models.aethelnet")

AETHELNET_PRESETS: Dict[str, Dict[str, float]] = {
    "aethelnet_tiny": {"depth_multiple": 0.33, "width_multiple": 0.25},
    "aethelnet_small": {"depth_multiple": 0.33, "width_multiple": 0.50},
    "aethelnet_medium": {"depth_multiple": 0.67, "width_multiple": 0.75},
    "aethelnet_large": {"depth_multiple": 1.00, "width_multiple": 1.00},
    "aethelnet_xlarge": {"depth_multiple": 1.33, "width_multiple": 1.25},
}


@MODEL_REGISTRY.register("AethelNet")
class AethelNet(AethelDetector):
    """AethelNet Model Architecture Family."""

    def __init__(
        self,
        num_classes: int = 80,
        variant: str = "medium",
        depth_multiple: Optional[float] = None,
        width_multiple: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = f"aethelnet_{variant.lower()}" if not variant.startswith("aethelnet_") else variant.lower()
        if key in AETHELNET_PRESETS:
            d_mult = AETHELNET_PRESETS[key]["depth_multiple"]
            w_mult = AETHELNET_PRESETS[key]["width_multiple"]
        else:
            d_mult, w_mult = 0.67, 0.75

        depth_multiple = depth_multiple if depth_multiple is not None else d_mult
        width_multiple = width_multiple if width_multiple is not None else w_mult

        super().__init__(
            num_classes=num_classes,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            config=config,
        )
        self.variant = variant
        logger.info(f"Initialized AethelNet-{variant.upper()} (num_classes={num_classes})")


# Register variant aliases
@MODEL_REGISTRY.register("aethelnet_tiny")
def build_aethelnet_tiny(num_classes: int = 80, **kwargs: Any) -> AethelNet:
    return AethelNet(num_classes=num_classes, variant="tiny", **kwargs)


@MODEL_REGISTRY.register("aethelnet_small")
def build_aethelnet_small(num_classes: int = 80, **kwargs: Any) -> AethelNet:
    return AethelNet(num_classes=num_classes, variant="small", **kwargs)


@MODEL_REGISTRY.register("aethelnet_medium")
def build_aethelnet_medium(num_classes: int = 80, **kwargs: Any) -> AethelNet:
    return AethelNet(num_classes=num_classes, variant="medium", **kwargs)


@MODEL_REGISTRY.register("aethelnet_large")
def build_aethelnet_large(num_classes: int = 80, **kwargs: Any) -> AethelNet:
    return AethelNet(num_classes=num_classes, variant="large", **kwargs)


@MODEL_REGISTRY.register("aethelnet_xlarge")
def build_aethelnet_xlarge(num_classes: int = 80, **kwargs: Any) -> AethelNet:
    return AethelNet(num_classes=num_classes, variant="xlarge", **kwargs)


def build_model(model_name: str, num_classes: int = 80, **kwargs: Any) -> AethelNet:
    """Factory helper to construct any AethelNet model from registry string.

    Args:
        model_name: Model identifier string (e.g. 'aethelnet_tiny', 'aethelnet_small').
        num_classes: Target classification class count.

    Returns:
        Instantiated AethelNet neural network model.
    """
    key = model_name.lower()
    if MODEL_REGISTRY.contains(key):
        builder = MODEL_REGISTRY.get(key)
        return builder(num_classes=num_classes, **kwargs)
    elif MODEL_REGISTRY.contains("AethelNet"):
        return AethelNet(num_classes=num_classes, variant=key, **kwargs)
    else:
        raise ModelException(message=f"Unknown model architecture '{model_name}' in MODEL_REGISTRY")


__all__ = [
    "AethelNet",
    "build_model",
    "build_aethelnet_tiny",
    "build_aethelnet_small",
    "build_aethelnet_medium",
    "build_aethelnet_large",
    "build_aethelnet_xlarge",
    "AETHELNET_PRESETS",
]
