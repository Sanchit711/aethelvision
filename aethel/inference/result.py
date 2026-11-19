"""Unified Prediction Result Container for Aethel Vision Framework.

Provides a structured PredictionResult dataclass encapsulating detection bounding boxes,
confidence scores, class IDs, class labels, inference timing, and image metadata.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from aethel.utils.logger import get_logger

logger = get_logger("aethel.inference.result")


@dataclass
class PredictionResult:
    """Unified container for object detection inference outputs."""

    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    classes: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    labels: List[str] = field(default_factory=list)
    keypoints: Optional[np.ndarray] = None  # (N, K, 3) [x, y, conf]
    masks: Optional[np.ndarray] = None  # (N, H, W) binary or float mask
    obb_angles: Optional[np.ndarray] = None  # (N,) radians
    obb_corners: Optional[np.ndarray] = None  # (N, 4, 2) 4 vertices
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_shape: tuple = (0, 0, 3)
    orig_shape: tuple = (0, 0, 3)

    def __len__(self) -> int:
        """Return number of detected objects."""
        return len(self.boxes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert PredictionResult instance to a JSON-serializable dictionary."""
        data: Dict[str, Any] = {
            "num_detections": len(self),
            "boxes": self.boxes.tolist() if isinstance(self.boxes, np.ndarray) else [],
            "scores": [round(float(s), 4) for s in self.scores] if isinstance(self.scores, np.ndarray) else [],
            "classes": [int(c) for c in self.classes] if isinstance(self.classes, np.ndarray) else [],
            "labels": self.labels,
            "timing_ms": {k: round(v, 2) for k, v in self.timing.items()},
            "metadata": self.metadata,
            "image_shape": list(self.image_shape),
            "orig_shape": list(self.orig_shape),
        }
        if self.keypoints is not None:
            data["keypoints"] = np.round(self.keypoints, 2).tolist() if isinstance(self.keypoints, np.ndarray) else []
        if self.obb_angles is not None:
            data["obb_angles"] = [round(float(a), 4) for a in self.obb_angles] if isinstance(self.obb_angles, np.ndarray) else []
        if self.obb_corners is not None:
            data["obb_corners"] = np.round(self.obb_corners, 2).tolist() if isinstance(self.obb_corners, np.ndarray) else []
        if self.masks is not None:
            data["num_masks"] = len(self.masks)
        return data

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save prediction result to a JSON report file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved prediction result JSON report to {path}")


__all__ = ["PredictionResult"]
