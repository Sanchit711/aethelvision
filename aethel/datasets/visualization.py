"""Visualization Utilities for Dataset Inspection in Aethel Vision.

Renders bounding box overlays, class tags, dataset previews, grid layouts, and saves
visualizations to disk.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aethel.datasets.metadata import AnnotationInfo
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.visualization")


class VisualizationManager:
    """Manager for rendering bounding box overlays and saving dataset sanity previews."""

    @staticmethod
    def draw_boxes(
        image: np.ndarray,
        annotations: List[AnnotationInfo],
        class_names: Optional[List[str]] = None,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and class tags onto an RGB image array.

        Args:
            image: NumPy uint8 array (H, W, 3).
            annotations: List of AnnotationInfo objects.
            class_names: Optional class name strings list.
            line_thickness: Box line width.

        Returns:
            Annotated NumPy uint8 array (H, W, 3).
        """
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size

        for ann in annotations:
            x1, y1, x2, y2 = ann.to_xyxy(w, h)
            cid = ann.class_id
            label = class_names[cid] if class_names and 0 <= cid < len(class_names) else f"Class {cid}"

            # Box outline
            draw.rectangle([x1, y1, x2, y2], outline="red", width=line_thickness)

            # Label text
            text = f"{label}"
            draw.text((x1 + 3, y1 + 3), text, fill="yellow")

        return np.array(pil_img, dtype=np.uint8)

    @staticmethod
    def save_sample_preview(
        image: np.ndarray,
        annotations: List[AnnotationInfo],
        output_path: Union[str, Path],
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Render and save a single annotated image preview to disk."""
        annotated = VisualizationManager.draw_boxes(image, annotations, class_names=class_names)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(annotated).save(path)
        logger.info(f"Saved dataset preview visualization to {path}")

    @staticmethod
    def save_grid_preview(
        samples: List[Tuple[np.ndarray, List[AnnotationInfo]]],
        output_path: Union[str, Path],
        cols: int = 4,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Render grid of annotated image samples and save to disk.

        Args:
            samples: List of (image_array, annotations) tuples.
            output_path: Output file path.
            cols: Column count in grid layout.
            class_names: Optional class names list.
        """
        if not samples:
            return

        annotated_imgs = [
            Image.fromarray(VisualizationManager.draw_boxes(img, anns, class_names=class_names))
            for img, anns in samples
        ]

        # Uniform tile sizing
        tile_w, tile_h = 320, 320
        tiles = [img.resize((tile_w, tile_h)) for img in annotated_imgs]

        rows = (len(tiles) + cols - 1) // cols
        grid_img = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(30, 30, 30))

        for idx, tile in enumerate(tiles):
            r = idx // cols
            c = idx % cols
            grid_img.paste(tile, (c * tile_w, r * tile_h))

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(path)
        logger.info(f"Saved dataset grid preview ({len(samples)} samples) to {path}")


__all__ = ["VisualizationManager"]
