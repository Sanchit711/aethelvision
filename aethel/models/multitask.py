"""Multi-Task Vision Model Architecture for Aethel Vision Framework.

Provides MultiTaskModel containing a shared backbone (and optional neck) feeding
multiple task-specific heads (Detection, Classification, Segmentation, Extensible for Pose, OCR, Tracking, Face).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

from aethel.models.base import BaseModel
from aethel.models.backbone.cspdarknet import CSPDarknet
from aethel.models.neck.panet import PANet
from aethel.models.head.detection_head import DetectionHead
from aethel.registry.registry import MODEL_REGISTRY
from aethel.utils.logger import get_logger

logger = get_logger("aethel.models.multitask")


class BaseTaskHead(ABC):
    """Abstract interface for all task-specific heads attached to MultiTaskModel."""

    @abstractmethod
    def forward(self, features: Any) -> Any:
        """Forward pass through head module given neck/backbone feature maps."""
        pass


@MODEL_REGISTRY.register("MultiTaskModel")
class MultiTaskModel(BaseModel if BaseModel is not None else object):
    """Shared backbone multi-task vision foundation architecture."""

    def __init__(
        self,
        backbone: Optional[Any] = None,
        neck: Optional[Any] = None,
        heads: Optional[Dict[str, Any]] = None,
        scale: str = "tiny",
        num_classes_dict: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize MultiTaskModel.

        Args:
            backbone: Shared backbone feature extractor.
            neck: Optional shared feature fusion neck (e.g., PANet/FPN).
            heads: Dictionary mapping task names (e.g. 'detection', 'classification') to head modules.
            scale: Model scale variant string.
            num_classes_dict: Dictionary mapping task name to number of active classes.
        """
        if hasattr(super(), "__init__"):
            super().__init__()

        self.scale = scale
        self.num_classes_dict = num_classes_dict or {"detection": 80}

        # 1. Instantiate Shared Backbone & Neck
        self.backbone = backbone if backbone is not None else CSPDarknet(scale=scale)
        self.neck = neck if neck is not None else PANet(scale=scale)

        # 2. Instantiate Task Heads Dictionary
        self.heads = nn.ModuleDict() if nn is not None else {}
        if heads:
            for task_name, head_module in heads.items():
                self.add_task_head(task_name, head_module)
        else:
            # Default single-task detection head fallback
            det_head = DetectionHead(
                num_classes=self.num_classes_dict.get("detection", 80),
                scale=scale,
            )
            self.add_task_head("detection", det_head)

    def add_task_head(self, task_name: str, head_module: Any) -> None:
        """Dynamically attach a new task head module to the shared backbone/neck."""
        key = task_name.lower()
        if nn is not None and isinstance(self.heads, nn.ModuleDict):
            self.heads[key] = head_module
        logger.info(f"Attached task head '{key}' to MultiTaskModel")

    def forward(
        self,
        x: Any,
        tasks: Optional[List[str]] = None,
    ) -> Union[Any, Dict[str, Any]]:
        """Forward pass through shared backbone/neck and active task heads.

        Args:
            x: Input image tensor (B, 3, H, W).
            tasks: Optional list of task names to evaluate. If None, evaluate all registered heads.

        Returns:
            Output prediction tensor (if single task) or dict mapping task name to head predictions.
        """
        if torch is None:
            return {}

        # Shared feature extraction
        features = self.backbone(x)
        if self.neck is not None:
            features = self.neck(features)

        # Execute requested task heads
        outputs = {}
        target_tasks = tasks or list(self.heads.keys())

        for tname in target_tasks:
            tname_lower = tname.lower()
            if tname_lower in self.heads:
                outputs[tname_lower] = self.heads[tname_lower](features)

        if len(outputs) == 1 and tasks is None:
            return list(outputs.values())[0]

        return outputs


__all__ = ["BaseTaskHead", "MultiTaskModel"]
