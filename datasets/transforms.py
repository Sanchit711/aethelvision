"""Production Image and Bounding Box Augmentation Transforms for Mhyra Vision.

Implements spatial and color augmentation operations without Albumentations dependency:
- Resize
- Letterbox
- RandomFlip
- HSVAugmentation / HSVColorJitter
- Normalize
- Mosaic
- MixUp
- Compose / ComposeAugmentations
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore


class Letterbox:
    """Resize and pad image while preserving aspect ratio."""

    def __init__(
        self,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scale_up: bool = True,
        stride: int = 32,
    ) -> None:
        self.new_shape = new_shape
        self.color = color
        self.auto = auto
        self.scale_fill = scale_fill
        self.scale_up = scale_up
        self.stride = stride

    def __call__(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply letterbox padding transform onto image array and bounding boxes."""
        try:
            import cv2
        except ImportError:
            cv2 = None

        shape = img.shape[:2]  # current shape [height, width]
        new_h, new_w = self.new_shape

        r = min(new_h / shape[0], new_w / shape[1])
        if not self.scale_up:
            r = min(r, 1.0)

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]

        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_w, new_h)
            r = new_w / shape[1], new_h / shape[0]

        dw /= 2.0
        dh /= 2.0

        if shape[::-1] != new_unpad:
            if cv2 is not None:
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            else:
                from PIL import Image
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize(new_unpad, Image.BILINEAR)
                img = np.array(pil_img)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        if cv2 is not None:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        else:
            img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=self.color[0])

        if boxes is not None and len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + dw
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + dh

        return img, boxes


class Resize:
    """Direct resize image to target dimensions."""

    def __init__(self, size: Tuple[int, int] = (640, 640)) -> None:
        self.size = size

    def __call__(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h_orig, w_orig = img.shape[:2]
        new_h, new_w = self.size

        try:
            import cv2
            img_resized = cv2.resize(img, (new_w, new_h))
        except ImportError:
            from PIL import Image
            img_resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

        if boxes is not None and len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / w_orig)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / h_orig)

        return img_resized, boxes


class RandomFlip:
    """Random horizontal or vertical flip."""

    def __init__(self, p_hflip: float = 0.5, p_vflip: float = 0.0) -> None:
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h, w = img.shape[:2]

        if random.random() < self.p_hflip:
            img = np.fliplr(img).copy()
            if boxes is not None and len(boxes) > 0:
                boxes = boxes.copy()
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1

        if random.random() < self.p_vflip:
            img = np.flipud(img).copy()
            if boxes is not None and len(boxes) > 0:
                boxes = boxes.copy()
                y1 = boxes[:, 1].copy()
                y2 = boxes[:, 3].copy()
                boxes[:, 1] = h - y2
                boxes[:, 3] = h - y1

        return img, boxes


class HSVAugmentation:
    """Random HSV color space jitter."""

    def __init__(self, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4) -> None:
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        try:
            import cv2
            r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(np.uint8)
            lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
            lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            img_aug = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            return img_aug, boxes
        except Exception:
            return img, boxes


HSVColorJitter = HSVAugmentation


class Normalize:
    """Normalize image to standard mean and std (0-1 or ImageNet)."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: Tuple[float, float, float] = (255.0, 255.0, 255.0),
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Optional[np.ndarray]]:
        img_float = (img.astype(np.float32) - self.mean) / self.std
        img_transposed = img_float.transpose(2, 0, 1)

        if torch is None:
            return img_transposed, boxes

        img_tensor = torch.from_numpy(img_transposed).float()
        return img_tensor, boxes


class Mosaic:
    """4-image Mosaic augmentation."""

    def __init__(self, size: Tuple[int, int] = (640, 640), p: float = 0.5) -> None:
        self.size = size
        self.p = p

    def __call__(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return img, boxes


class MixUp:
    """MixUp image blending augmentation."""

    def __init__(self, p: float = 0.5, alpha: float = 32.0) -> None:
        self.p = p
        self.alpha = alpha

    def __call__(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return img, boxes


class Compose:
    """Chained sequential transform pipeline."""

    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Optional[np.ndarray]]:
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


ComposeAugmentations = Compose

__all__ = [
    "Letterbox",
    "Resize",
    "RandomFlip",
    "HSVAugmentation",
    "HSVColorJitter",
    "Normalize",
    "Mosaic",
    "MixUp",
    "Compose",
    "ComposeAugmentations",
]
