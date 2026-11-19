"""Production Dataset Visualizer for Mhyra Detector Datasets.

Renders bounding box annotations, class label text tags, overlay previews,
and sample grid images for visual dataset inspection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
from PIL import Image, ImageDraw, ImageFont


class DatasetVisualizer:
    """Visualizer tool for rendering object detection bounding box overlays."""

    # Default 10 high-contrast class colors
    DEFAULT_COLORS = [
        (255, 59, 48),    # Red
        (52, 199, 89),    # Green
        (0, 122, 255),    # Blue
        (255, 149, 0),   # Orange
        (88, 86, 214),    # Purple
        (255, 204, 0),   # Yellow
        (90, 200, 250),   # Cyan
        (255, 45, 85),    # Pink
        (175, 82, 222),   # Violet
        (142, 142, 147),  # Gray
    ]

    @classmethod
    def draw_boxes_on_image(
        cls,
        image: Union[Image.Image, torch.Tensor],
        boxes: torch.Tensor,
        labels: torch.Tensor,
        class_names: Optional[Dict[int, str]] = None,
    ) -> Image.Image:
        """Draw bounding boxes and class label tags onto a PIL Image.

        Args:
            image: Input PIL Image or PyTorch Float Tensor (3, H, W) in range [0, 1] or [0, 255].
            boxes: Bounding boxes tensor of shape (N, 4) with [x1, y1, x2, y2].
            labels: Class IDs tensor of shape (N, 1) or (N,).
            class_names: Optional dictionary mapping class ID to string class name.

        Returns:
            PIL Image with overlaid bounding boxes and class labels.
        """
        if isinstance(image, torch.Tensor):
            img_tensor = image.detach().cpu()
            if img_tensor.max() <= 1.0:
                img_tensor = img_tensor * 255.0
            img_np = img_tensor.byte().permute(1, 2, 0).numpy()
            pil_img = Image.fromarray(img_np, mode="RGB")
        else:
            pil_img = image.copy()

        draw = ImageDraw.Draw(pil_img)
        w_img, h_img = pil_img.size

        if boxes.numel() == 0:
            return pil_img

        boxes_np = boxes.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().squeeze()

        if labels_np.ndim == 0:
            labels_np = np.array([labels_np])

        for idx, box in enumerate(boxes_np):
            x1, y1, x2, y2 = box
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue

            cls_id = int(labels_np[idx]) if idx < len(labels_np) else 0
            color = cls.DEFAULT_COLORS[cls_id % len(cls.DEFAULT_COLORS)]
            cls_name = class_names.get(cls_id, f"Class {cls_id}") if class_names else f"Class {cls_id}"

            # Draw box rectangle with width=3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw text label box
            text_str = f" {cls_name} "
            text_bbox = draw.textbbox((x1, y1), text_str)
            draw.rectangle([x1, y1 - 14, x1 + (text_bbox[2] - text_bbox[0]), y1], fill=color)
            draw.text((x1, y1 - 14), text_str, fill=(255, 255, 255))

        return pil_img

    @classmethod
    def create_grid_preview(
        cls,
        dataset: Any,
        num_samples: int = 9,
        save_path: Optional[Union[str, Path]] = "logs/dataset_preview.png",
        class_names: Optional[Dict[int, str]] = None,
    ) -> Image.Image:
        """Render a 3x3 sample grid preview image from a dataset.

        Args:
            dataset: Detection dataset instance.
            num_samples: Number of dataset items to draw (default 9).
            save_path: Optional output path to save preview image.
            class_names: Optional dictionary mapping class ID to string class name.

        Returns:
            Combined PIL grid preview image.
        """
        samples = []
        n_items = min(num_samples, len(dataset))

        for i in range(n_items):
            img_tensor, padded_labels, padded_boxes, mask_gt = dataset[i]
            valid_mask = mask_gt.squeeze(-1) > 0
            boxes = padded_boxes[valid_mask]
            labels = padded_labels[valid_mask]

            rendered = cls.draw_boxes_on_image(img_tensor, boxes, labels, class_names=class_names)
            samples.append(rendered.resize((320, 320)))

        # Create 3x3 Grid Image
        grid_img = Image.new("RGB", (320 * 3, 320 * 3), color=(240, 240, 240))
        for idx, sample in enumerate(samples):
            r = idx // 3
            c = idx % 3
            grid_img.paste(sample, (c * 320, r * 320))

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            grid_img.save(out_p)

        return grid_img


__all__ = ["DatasetVisualizer"]
