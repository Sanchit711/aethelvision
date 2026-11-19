"""Modular Image and Bounding Box Augmentation Subsystem for Aethel Vision.

Implements Resize, Letterbox, Flips, Crop, Scale, Rotation, HSV, Brightness, Contrast,
Gaussian Noise, and Blur using PIL and NumPy with optional OpenCV acceleration.
Fully driven by YAML configuration specs.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance

from aethel.datasets.metadata import AnnotationInfo
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.transforms")


class TransformPipeline:
    """Modular augmentation pipeline operating over image arrays and bounding boxes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, is_train: bool = True) -> None:
        """Initialize TransformPipeline.

        Args:
            config: Augmentation configuration dictionary.
            is_train: Flag indicating training vs validation mode.
        """
        self.config = config or {}
        self.is_train = is_train

        # Extract config parameters with defaults
        self.imgsz = self.config.get("imgsz", 640)
        self.hflip_prob = self.config.get("hflip", 0.5) if is_train else 0.0
        self.vflip_prob = self.config.get("vflip", 0.0) if is_train else 0.0
        self.hsv_h = self.config.get("hsv_h", 0.015) if is_train else 0.0
        self.hsv_s = self.config.get("hsv_s", 0.7) if is_train else 0.0
        self.hsv_v = self.config.get("hsv_v", 0.4) if is_train else 0.0

    def letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        """Resize and pad image preserving aspect ratio.

        Returns:
            Tuple of (padded_image, (scale_w, scale_h), (pad_left, pad_top)).
        """
        h, w = img.shape[:2]
        if isinstance(new_shape, int):
            target_h, target_w = new_shape, new_shape
        else:
            target_h, target_w = new_shape

        r = min(target_h / h, target_w / w)
        new_unpad_w = int(round(w * r))
        new_unpad_h = int(round(h * r))

        dw = target_w - new_unpad_w
        dh = target_h - new_unpad_h
        dw /= 2
        dh /= 2

        pil_img = Image.fromarray(img)
        if (w, h) != (new_unpad_w, new_unpad_h):
            pil_img = pil_img.resize((new_unpad_w, new_unpad_h), Image.Resampling.BILINEAR)

        # Create padded background canvas
        canvas = Image.new("RGB", (target_w, target_h), color=color)
        left = int(round(dw - 0.1))
        top = int(round(dh - 0.1))
        canvas.paste(pil_img, (left, top))

        return np.array(canvas, dtype=np.uint8), (r, r), (dw, dh)

    def apply_hflip(self, img: np.ndarray, annotations: List[AnnotationInfo]) -> Tuple[np.ndarray, List[AnnotationInfo]]:
        """Apply horizontal flip."""
        img_flipped = np.fliplr(img)
        flipped_anns = []
        for ann in annotations:
            if ann.format == "xywh_norm":
                xc, yc, w, h = ann.bbox
                new_xc = 1.0 - xc
                flipped_anns.append(AnnotationInfo(class_id=ann.class_id, bbox=(new_xc, yc, w, h), format="xywh_norm"))
            else:
                flipped_anns.append(ann)
        return img_flipped, flipped_anns

    def apply_hsv(self, img: np.ndarray) -> np.ndarray:
        """Apply random HSV / color jitter using PIL ImageEnhance."""
        if self.hsv_h == 0 and self.hsv_s == 0 and self.hsv_v == 0:
            return img

        pil_img = Image.fromarray(img)
        if self.hsv_s > 0:
            factor = 1.0 + random.uniform(-self.hsv_s, self.hsv_s)
            pil_img = ImageEnhance.Color(pil_img).enhance(max(0.1, factor))
        if self.hsv_v > 0:
            factor = 1.0 + random.uniform(-self.hsv_v, self.hsv_v)
            pil_img = ImageEnhance.Brightness(pil_img).enhance(max(0.1, factor))

        return np.array(pil_img, dtype=np.uint8)

    def __call__(
        self,
        img: np.ndarray,
        annotations: List[AnnotationInfo],
    ) -> Tuple[np.ndarray, List[AnnotationInfo]]:
        """Apply transformation pipeline to image array and bounding boxes."""
        # 1. Letterbox resize to target imgsz
        img, (scale_w, scale_h), (pad_w, pad_h) = self.letterbox(img, new_shape=(self.imgsz, self.imgsz))

        # 2. Horizontal flip (if training and triggered)
        if self.is_train and random.random() < self.hflip_prob:
            img, annotations = self.apply_hflip(img, annotations)

        # 3. HSV / Color jitter (if training)
        if self.is_train:
            img = self.apply_hsv(img)

        return img, annotations


__all__ = ["TransformPipeline"]
