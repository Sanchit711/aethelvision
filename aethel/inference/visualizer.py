"""Aethel Indigenous Universal Multi-Modal Visualizer.

Provides rendering and visual overlays for:
- 2D Bounding Boxes & Confidence Tags
- 17-Keypoint Human Pose Skeletons (COCO format)
- Semi-Transparent Alpha-Blended Instance Segmentation Masks
- Oriented Bounding Boxes (OBB) with 4-Corner Polygons and Heading Vectors

Supports PIL Image and NumPy arrays natively, with zero dependency on OpenCV.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aethel.inference.result import PredictionResult
from aethel.models.head.pose import COCO_KEYPOINTS, COCO_SKELETON
from aethel.utils.logger import get_logger

logger = get_logger("aethel.inference.visualizer")

# Distinct color palette (R, G, B) for instance rendering
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 65), (37, 210, 216), (23, 163, 229), (0, 102, 255),
    (115, 51, 206), (188, 61, 230), (255, 56, 203), (255, 159, 243),
]

# Pose limb colors (left vs right vs center)
POSE_LIMB_COLORS = {
    "left": (50, 220, 50),     # Bright green
    "right": (230, 50, 50),    # Bright red
    "center": (240, 240, 30),  # Yellow
}


def _get_limb_color(j1: int, j2: int) -> Tuple[int, int, int]:
    """Assign color based on left/right/center human anatomy."""
    # Odd numbers in COCO are typically left, even are right
    left_joints = {1, 3, 5, 7, 9, 11, 13, 15}
    right_joints = {2, 4, 6, 8, 10, 12, 14, 16}
    if j1 in left_joints and j2 in left_joints:
        return POSE_LIMB_COLORS["left"]
    elif j1 in right_joints and j2 in right_joints:
        return POSE_LIMB_COLORS["right"]
    return POSE_LIMB_COLORS["center"]


class AethelVisualizer:
    """Indigenous Universal Visualizer for Detection, Pose, Segmentation, and OBB."""

    def __init__(self, font_size: int = 12, mask_alpha: float = 0.45) -> None:
        self.font_size = font_size
        self.mask_alpha = mask_alpha
        try:
            self.font = ImageFont.load_default()
        except Exception:
            self.font = None

    def _ensure_pil(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Convert input image to PIL RGBA/RGB mode."""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255.0).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            return Image.fromarray(image).convert("RGBA")
        return image.convert("RGBA")

    def draw_detections(
        self,
        image: Union[Image.Image, np.ndarray],
        result: PredictionResult,
        line_width: int = 2,
    ) -> Image.Image:
        """Draw standard bounding boxes and category labels."""
        img = self._ensure_pil(image)
        draw = ImageDraw.Draw(img)

        boxes = result.boxes
        scores = result.scores
        classes = result.classes
        labels = result.labels

        for i, box in enumerate(boxes):
            color = PALETTE[int(classes[i]) % len(PALETTE)] if len(classes) > i else PALETTE[0]
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            label = labels[i] if len(labels) > i else f"cls_{int(classes[i])}" if len(classes) > i else "obj"
            score = scores[i] if len(scores) > i else 1.0
            tag = f"{label} {score:.2f}"

            # Tag background
            draw.rectangle([x1, max(0, y1 - 16), x1 + len(tag) * 7 + 4, y1], fill=color)
            draw.text((x1 + 2, max(0, y1 - 15)), tag, fill=(255, 255, 255), font=self.font)

        return img.convert("RGB")

    def draw_pose(
        self,
        image: Union[Image.Image, np.ndarray],
        result: PredictionResult,
        kpt_conf_threshold: float = 0.3,
        kpt_radius: int = 4,
        limb_width: int = 2,
    ) -> Image.Image:
        """Draw 17-keypoint skeleton lines and joint nodes."""
        img = self._ensure_pil(image)
        draw = ImageDraw.Draw(img)

        if result.keypoints is None or len(result.keypoints) == 0:
            return self.draw_detections(img, result)

        # Draw boxes first
        boxes = result.boxes
        for i, box in enumerate(boxes):
            draw.rectangle(box.tolist(), outline=(180, 180, 180), width=1)

        keypoints_batch = result.keypoints  # (N, 17, 3) or (N, 17, 2)

        for person_idx, kpts in enumerate(keypoints_batch):
            # Draw skeleton limb connections
            for j1, j2 in COCO_SKELETON:
                if j1 >= len(kpts) or j2 >= len(kpts):
                    continue
                kp1, kp2 = kpts[j1], kpts[j2]
                conf1 = kp1[2] if len(kp1) > 2 else 1.0
                conf2 = kp2[2] if len(kp2) > 2 else 1.0

                if conf1 >= kpt_conf_threshold and conf2 >= kpt_conf_threshold:
                    x1, y1 = float(kp1[0]), float(kp1[1])
                    x2, y2 = float(kp2[0]), float(kp2[1])
                    limb_color = _get_limb_color(j1, j2)
                    draw.line([(x1, y1), (x2, y2)], fill=limb_color, width=limb_width)

            # Draw keypoint joint circles
            for j_idx, kp in enumerate(kpts):
                conf = kp[2] if len(kp) > 2 else 1.0
                if conf >= kpt_conf_threshold:
                    x, y = float(kp[0]), float(kp[1])
                    node_color = PALETTE[j_idx % len(PALETTE)]
                    draw.ellipse(
                        [x - kpt_radius, y - kpt_radius, x + kpt_radius, y + kpt_radius],
                        fill=node_color,
                        outline=(255, 255, 255),
                    )

        return img.convert("RGB")

    def draw_segmentation(
        self,
        image: Union[Image.Image, np.ndarray],
        result: PredictionResult,
    ) -> Image.Image:
        """Draw semi-transparent colored masks and bounding boxes."""
        base_img = self._ensure_pil(image)
        w, h = base_img.size

        if result.masks is None or len(result.masks) == 0:
            return self.draw_detections(base_img, result)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        masks = result.masks  # (N, H, W)
        classes = result.classes

        for i, mask in enumerate(masks):
            color = PALETTE[int(classes[i]) % len(PALETTE)] if len(classes) > i else PALETTE[i % len(PALETTE)]
            # If mask is lower resolution, resize to full image
            if mask.shape != (h, w):
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
                mask_bin = np.array(mask_pil) > 127
            else:
                mask_bin = mask > 0.5

            mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            alpha_val = int(255 * self.mask_alpha)
            mask_rgba[mask_bin] = [color[0], color[1], color[2], alpha_val]

            mask_img = Image.fromarray(mask_rgba, mode="RGBA")
            overlay = Image.alpha_composite(overlay, mask_img)

        # Composite mask overlay on top of base image
        blended = Image.alpha_composite(base_img, overlay)

        # Draw bounding boxes and tags on top
        return self.draw_detections(blended, result)

    def draw_obb(
        self,
        image: Union[Image.Image, np.ndarray],
        result: PredictionResult,
        line_width: int = 2,
    ) -> Image.Image:
        """Draw 4-corner oriented bounding polygons and direction indicators."""
        img = self._ensure_pil(image)
        draw = ImageDraw.Draw(img)

        if result.obb_corners is None or len(result.obb_corners) == 0:
            return self.draw_detections(img, result)

        corners_batch = result.obb_corners  # (N, 4, 2)
        angles = result.obb_angles  # (N,)
        classes = result.classes
        scores = result.scores
        labels = result.labels

        for i, corners in enumerate(corners_batch):
            color = PALETTE[int(classes[i]) % len(PALETTE)] if len(classes) > i else PALETTE[0]
            pts = [(float(c[0]), float(c[1])) for c in corners]

            # Draw 4 polygon sides
            draw.polygon(pts, outline=color, width=line_width)

            # Center point
            cx = float(np.mean(corners[:, 0]))
            cy = float(np.mean(corners[:, 1]))
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)

            # Orientation indicator vector
            if angles is not None and len(angles) > i:
                theta = float(angles[i])
                arrow_len = 25.0
                ax = cx + arrow_len * math.cos(theta)
                ay = cy + arrow_len * math.sin(theta)
                draw.line([(cx, cy), (ax, ay)], fill=(255, 255, 0), width=2)
                draw.ellipse([ax - 2, ay - 2, ax + 2, ay + 2], fill=(255, 255, 0))

            # Label tag
            label = labels[i] if len(labels) > i else f"cls_{int(classes[i])}" if len(classes) > i else "obb"
            score = scores[i] if len(scores) > i else 1.0
            deg = math.degrees(float(angles[i])) if angles is not None and len(angles) > i else 0.0
            tag = f"{label} {score:.2f} ({deg:.0f}°)"

            tx, ty = pts[0]
            draw.rectangle([tx, max(0, ty - 16), tx + len(tag) * 7 + 4, ty], fill=color)
            draw.text((tx + 2, max(0, ty - 15)), tag, fill=(255, 255, 255), font=self.font)

        return img.convert("RGB")

    def draw(
        self,
        image: Union[Image.Image, np.ndarray],
        result: PredictionResult,
    ) -> Image.Image:
        """Universal draw dispatcher auto-detecting task modalities."""
        if result.keypoints is not None and len(result.keypoints) > 0:
            return self.draw_pose(image, result)
        elif result.masks is not None and len(result.masks) > 0:
            return self.draw_segmentation(image, result)
        elif result.obb_corners is not None and len(result.obb_corners) > 0:
            return self.draw_obb(image, result)
        return self.draw_detections(image, result)


__all__ = ["AethelVisualizer", "PALETTE", "POSE_LIMB_COLORS"]
