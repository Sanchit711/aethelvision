"""Visualization Subsystem for Aethel Vision Framework.

Renders bounding box overlays, class labels, and confidence scores onto images.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from aethel.inference.result import PredictionResult
from aethel.registry.registry import VISUALIZER_REGISTRY
from aethel.utils.logger import get_logger
from aethel.visualization.base import BaseVisualizer

logger = get_logger("aethel.visualization")


@VISUALIZER_REGISTRY.register("Visualizer")
class Visualizer(BaseVisualizer):
    """Primary Image and Bounding Box Overlay Visualizer."""

    def __init__(self, color_palette: Optional[List[Tuple[int, int, int]]] = None) -> None:
        super().__init__()
        self.palette = color_palette or self._generate_palette(80)

    def draw_detections(
        self,
        image: Any,
        boxes: Any,
        scores: Any,
        class_ids: Any,
        class_names: Optional[List[str]] = None,
    ) -> Any:
        """Render spatial bounding boxes onto image canvas."""
        res = PredictionResult(
            boxes=np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes,
            scores=np.array(scores) if not isinstance(scores, np.ndarray) else scores,
            classes=np.array(class_ids) if not isinstance(class_ids, np.ndarray) else class_ids,
            labels=class_names or [],
        )
        return np.array(self.draw(image, res))

    def draw(
        self,
        image: Union[np.ndarray, Image.Image],
        result: PredictionResult,
        line_width: int = 2,
    ) -> Image.Image:
        """Render bounding box overlays onto input image canvas.

        Args:
            image: NumPy array image (H, W, 3) or PIL Image.
            result: PredictionResult instance.
            line_width: Bounding box line thickness integer.

        Returns:
            PIL Image instance containing annotated overlays.
        """
        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image.copy()
        draw = ImageDraw.Draw(pil_img)

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            cls_id = int(result.classes[i]) if i < len(result.classes) else 0
            score = float(result.scores[i]) if i < len(result.scores) else 0.0
            label = result.labels[i] if i < len(result.labels) else f"class_{cls_id}"

            color = self.palette[cls_id % len(self.palette)]

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Draw text label banner
            text = f"{label} {score * 100:.1f}%"
            text_bbox = draw.textbbox((x1, y1), text)
            draw.rectangle([text_bbox[0], text_bbox[1] - 14, text_bbox[2] + 4, text_bbox[1]], fill=color)
            draw.text((x1 + 2, y1 - 14), text, fill=(255, 255, 255))

        return pil_img

    def save(
        self,
        image: Union[np.ndarray, Image.Image],
        result: PredictionResult,
        output_path: Union[str, Path],
    ) -> Path:
        """Render and save annotated image output to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        annotated_img = self.draw(image, result)
        annotated_img.save(path)
        logger.info(f"Saved annotated image visualization to {path}")
        return path

    @staticmethod
    def _generate_palette(num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct RGB color palette."""
        import colorsys
        colors = []
        for i in range(num_colors):
            hue = i / float(num_colors)
            rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
        return colors


__all__ = ["Visualizer"]
