"""Image and Annotation Loading Subsystem for Aethel Vision.

Provides memory-efficient image loading, orientation correction, image verification,
and raw annotation file reading.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from aethel.utils.exceptions import DatasetException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.loader")

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class ImageLoader:
    """Memory-efficient loader for image assets with format validation."""

    @staticmethod
    def is_valid_image_extension(filepath: Union[str, Path]) -> bool:
        """Check if file extension is supported."""
        ext = Path(filepath).suffix.lower()
        return ext in SUPPORTED_IMAGE_EXTENSIONS

    @staticmethod
    def load_image(image_path: Union[str, Path], mode: str = "RGB") -> np.ndarray:
        """Load image file into NumPy uint8 array (H, W, C).

        Args:
            image_path: Absolute filepath to image file.
            mode: Color mode string ('RGB', 'BGR', 'GRAY').

        Returns:
            NumPy array of shape (H, W, C) or (H, W).
        """
        path = Path(image_path)
        if not path.exists():
            raise DatasetException(
                message=f"Image file not found: {path}",
                details=f"Absolute path checked: {path.resolve()}",
            )

        if not ImageLoader.is_valid_image_extension(path):
            raise DatasetException(
                message=f"Unsupported image file extension: {path.suffix}",
                details=f"Supported extensions: {sorted(list(SUPPORTED_IMAGE_EXTENSIONS))}",
            )

        try:
            with Image.open(path) as img:
                # Correct EXIF orientation
                img = ImageOps.exif_transpose(img)
                if mode == "RGB":
                    img = img.convert("RGB")
                    return np.array(img, dtype=np.uint8)
                elif mode == "BGR":
                    img = img.convert("RGB")
                    arr = np.array(img, dtype=np.uint8)
                    return arr[:, :, ::-1]
                elif mode == "GRAY":
                    img = img.convert("L")
                    return np.array(img, dtype=np.uint8)
                else:
                    img = img.convert("RGB")
                    return np.array(img, dtype=np.uint8)
        except Exception as e:
            raise DatasetException(
                message=f"Corrupted or unreadable image file: {path.name}",
                details=str(e),
            )

    @staticmethod
    def get_image_size(image_path: Union[str, Path]) -> Tuple[int, int]:
        """Fetch (width, height) without decoding full pixel buffer.

        Args:
            image_path: Absolute image filepath.

        Returns:
            Tuple of (width, height).
        """
        path = Path(image_path)
        try:
            with Image.open(path) as img:
                width, height = img.size
                return width, height
        except Exception as e:
            raise DatasetException(
                message=f"Failed to read image header for size: {path.name}",
                details=str(e),
            )


class AnnotationLoader:
    """Raw annotation file loader for text, XML, and JSON annotations."""

    @staticmethod
    def read_text_lines(label_path: Union[str, Path]) -> List[str]:
        """Read lines from a text annotation file (e.g. YOLO txt)."""
        path = Path(label_path)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines
        except Exception as e:
            raise DatasetException(
                message=f"Failed to read text annotation file: {path.name}",
                details=str(e),
            )


__all__ = ["ImageLoader", "AnnotationLoader", "SUPPORTED_IMAGE_EXTENSIONS"]
