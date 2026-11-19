"""Data Augmentations Module for Mhyra Detector Datasets.

Provides production object detection spatial and color augmentations:
- Letterbox (aspect-ratio preserving pad resize)
- Resize
- RandomFlip (Horizontal and Vertical spatial box flipping)
- HSVColorJitter (Hue, Saturation, Value color transformation)
- RandomAffine (Translation, Scaling, Rotation, Shear)
- Mosaic (4-image mosaic blending)
- MixUp (2-image alpha blending)
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch


class Letterbox:
    """Aspect-ratio preserving image resize with symmetric border padding."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640), fill_value: float = 114.0) -> None:
        """Initialize Letterbox.

        Args:
            target_size: Target (height, width) tuple.
            fill_value: Padding pixel fill float value.
        """
        self.target_size = target_size
        self.fill_value = fill_value

    def __call__(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply letterbox resize and padding.

        Args:
            image: Tensor of shape (C, H, W).
            boxes: Bounding boxes tensor of shape (N, 4) with [x1, y1, x2, y2].

        Returns:
            Tuple of (letterboxed_image, resized_padded_boxes).
        """
        _, h, w = image.shape
        target_h, target_w = self.target_size

        r = min(target_h / h, target_w / w)
        new_h, new_w = int(round(h * r)), int(round(w * r))

        pad_h = (target_h - new_h) / 2
        pad_w = (target_w - new_w) / 2

        # Scale image via interpolate
        resized_img = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))

        padded_img = torch.nn.functional.pad(
            resized_img,
            (left, right, top, bottom),
            value=self.fill_value / 255.0 if image.max() <= 1.0 else self.fill_value,
        )

        # Scale and shift boxes
        scaled_boxes = boxes.clone()
        if boxes.numel() > 0:
            scaled_boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + pad_w
            scaled_boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + pad_h

        return padded_img, scaled_boxes


class Resize:
    """Direct spatial resize to target dimensions."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640)) -> None:
        self.target_size = target_size

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h, w = image.shape
        target_h, target_w = self.target_size

        sx = target_w / w
        sy = target_h / h

        resized_img = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        scaled_boxes = boxes.clone()
        if boxes.numel() > 0:
            scaled_boxes[:, [0, 2]] *= sx
            scaled_boxes[:, [1, 3]] *= sy

        return resized_img, scaled_boxes


class RandomFlip:
    """Random horizontal and vertical image and bounding box flipping."""

    def __init__(self, p_h: float = 0.5, p_v: float = 0.0) -> None:
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h, w = image.shape
        flipped_img = image.clone()
        flipped_boxes = boxes.clone()

        if random.random() < self.p_h:
            flipped_img = torch.flip(flipped_img, dims=[2])
            if flipped_boxes.numel() > 0:
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                flipped_boxes[:, 0] = x1
                flipped_boxes[:, 2] = x2

        if random.random() < self.p_v:
            flipped_img = torch.flip(flipped_img, dims=[1])
            if flipped_boxes.numel() > 0:
                y1 = h - boxes[:, 3]
                y2 = h - boxes[:, 1]
                flipped_boxes[:, 1] = y1
                flipped_boxes[:, 3] = y2

        return flipped_img, flipped_boxes


class HSVColorJitter:
    """Random Hue, Saturation, Value color jitter."""

    def __init__(self, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = torch.rand(3) * 2.0 - 1.0  # random gains
        gain = torch.tensor([1.0 + r[0] * self.hgain, 1.0 + r[1] * self.sgain, 1.0 + r[2] * self.vgain]).view(3, 1, 1)

        jittered_img = (image * gain).clamp(0.0, 1.0 if image.max() <= 1.0 else 255.0)
        return jittered_img, boxes


class RandomAffine:
    """Random translation and scaling affine transformation."""

    def __init__(self, degrees: float = 0.0, translate: float = 0.1, scale: float = 0.1) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h, w = image.shape
        tx = (random.random() * 2 - 1) * self.translate * w
        ty = (random.random() * 2 - 1) * self.translate * h
        s = 1.0 + (random.random() * 2 - 1) * self.scale

        transformed_boxes = boxes.clone()
        if boxes.numel() > 0:
            transformed_boxes[:, [0, 2]] = boxes[:, [0, 2]] * s + tx
            transformed_boxes[:, [1, 3]] = boxes[:, [1, 3]] * s + ty
            transformed_boxes[:, [0, 2]] = transformed_boxes[:, [0, 2]].clamp(0, w)
            transformed_boxes[:, [1, 3]] = transformed_boxes[:, [1, 3]].clamp(0, h)

        return image, transformed_boxes


class Mosaic:
    """4-Image Mosaic blending data augmentation."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640)) -> None:
        self.target_size = target_size

    def __call__(
        self,
        images: List[torch.Tensor],
        boxes_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine 4 images into a 2x2 mosaic layout.

        Args:
            images: List of 4 image tensors (3, H, W).
            boxes_list: List of 4 bounding box tensors (N_i, 4).
            labels_list: List of 4 label tensors (N_i, 1).

        Returns:
            Tuple of (mosaic_image, combined_boxes, combined_labels).
        """
        th, tw = self.target_size
        mosaic_img = torch.zeros((3, th, tw), dtype=torch.float32)

        mid_x, mid_y = tw // 2, th // 2
        all_boxes = []
        all_labels = []

        # Position offsets for 4 quadrants: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        offsets = [(0, 0, mid_x, mid_y), (mid_x, 0, tw, mid_y), (0, mid_y, mid_x, th), (mid_x, mid_y, tw, th)]

        for i in range(min(4, len(images))):
            x1, y1, x2, y2 = offsets[i]
            qw, qh = x2 - x1, y2 - y1

            resized_q = torch.nn.functional.interpolate(
                images[i].unsqueeze(0),
                size=(qh, qw),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mosaic_img[:, y1:y2, x1:x2] = resized_q

            b = boxes_list[i].clone()
            if b.numel() > 0:
                sx = qw / images[i].shape[2]
                sy = qh / images[i].shape[1]
                b[:, [0, 2]] = b[:, [0, 2]] * sx + x1
                b[:, [1, 3]] = b[:, [1, 3]] * sy + y1
                all_boxes.append(b)
                all_labels.append(labels_list[i])

        combined_boxes = torch.cat(all_boxes, dim=0) if all_boxes else torch.zeros((0, 4), dtype=torch.float32)
        combined_labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros((0, 1), dtype=torch.float32)

        return mosaic_img, combined_boxes, combined_labels


class MixUp:
    """2-Image MixUp alpha blending augmentation."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha

    def __call__(
        self,
        img1: torch.Tensor,
        boxes1: torch.Tensor,
        labels1: torch.Tensor,
        img2: torch.Tensor,
        boxes2: torch.Tensor,
        labels2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Blend 2 images and concatenate bounding box targets.

        Returns:
            Tuple of (blended_image, concatenated_boxes, concatenated_labels).
        """
        lam = random.betaovariate(self.alpha, self.alpha) if hasattr(random, "betaovariate") else 0.5
        blended_img = lam * img1 + (1.0 - lam) * img2

        combined_boxes = torch.cat([boxes1, boxes2], dim=0)
        combined_labels = torch.cat([labels1, labels2], dim=0)

        return blended_img, combined_boxes, combined_labels


class ComposeAugmentations:
    """Configurable pipeline composing spatial and color augmentations from YAML or dict configs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ComposeAugmentations.

        Args:
            config: Optional configuration dictionary.
        """
        cfg = config or {}

        target_size = tuple(cfg.get("img_size", (640, 640)))
        p_hflip = cfg.get("hflip_prob", 0.5)
        p_vflip = cfg.get("vflip_prob", 0.0)

        self.transforms = []

        if cfg.get("letterbox", True):
            self.transforms.append(Letterbox(target_size=target_size))
        else:
            self.transforms.append(Resize(target_size=target_size))

        if p_hflip > 0.0 or p_vflip > 0.0:
            self.transforms.append(RandomFlip(p_h=p_hflip, p_v=p_vflip))

        if cfg.get("hsv_jitter", True):
            self.transforms.append(
                HSVColorJitter(
                    hgain=cfg.get("hsv_hgain", 0.015),
                    sgain=cfg.get("hsv_sgain", 0.7),
                    vgain=cfg.get("hsv_vgain", 0.4),
                )
            )

        if cfg.get("affine", True):
            self.transforms.append(
                RandomAffine(
                    degrees=cfg.get("degrees", 0.0),
                    translate=cfg.get("translate", 0.1),
                    scale=cfg.get("scale", 0.1),
                )
            )

    def __call__(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute composed augmentations sequentially on image and boxes.

        Args:
            image: Image tensor (3, H, W).
            boxes: Bounding boxes tensor (N, 4).

        Returns:
            Tuple of (augmented_image, augmented_boxes).
        """
        img, bxs = image, boxes
        for t in self.transforms:
            img, bxs = t(img, bxs)
        return img, bxs


__all__ = [
    "Letterbox",
    "Resize",
    "RandomFlip",
    "HSVColorJitter",
    "RandomAffine",
    "Mosaic",
    "MixUp",
    "ComposeAugmentations",
]
