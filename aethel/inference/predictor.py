"""Primary Predictor Subsystem for Aethel Vision Framework.

Provides AethelPredictor supporting single images, image batches, directory paths,
NumPy arrays, PIL Images, OpenCV images, and PyTorch tensors.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aethel.inference.base import BasePredictor
from aethel.inference.decoder import PredictionDecoder
from aethel.inference.postprocess import NonMaximumSuppression
from aethel.inference.result import PredictionResult
from aethel.registry.registry import PREDICTOR_REGISTRY
from aethel.utils.exceptions import InferenceException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.inference.predictor")


@PREDICTOR_REGISTRY.register("AethelPredictor")
class AethelPredictor(BasePredictor):
    """Primary Object Detection Predictor Engine."""

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: Tuple[int, int] = (640, 640),
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize AethelPredictor.

        Args:
            model: Neural network model instance (or loaded via checkpoint).
            config: Inference configuration spec.
            conf_threshold: Confidence threshold.
            iou_threshold: NMS IoU threshold.
            imgsz: Target letterbox spatial dimensions (H, W).
            device: Hardware execution device string.
            class_names: Optional list of class name strings.
        """
        super().__init__(model_path=model or "", device=device)
        self.model = model
        self.config = config or {}
        self.conf_threshold = self.config.get("conf_threshold", conf_threshold)
        self.iou_threshold = self.config.get("iou_threshold", iou_threshold)
        self.imgsz = self.config.get("imgsz", imgsz)
        self.device_str = self.config.get("device", device)
        self.class_names = class_names

        # Decoder & NMS
        num_classes = getattr(self.model, "num_classes", 80) if self.model else 80
        reg_max = getattr(self.model, "reg_max", 16) if self.model else 16

        self.decoder = PredictionDecoder(num_classes=num_classes, reg_max=reg_max)
        self.nms = NonMaximumSuppression(
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )

        if torch is not None and self.model and hasattr(self.model, "to"):
            self.model.to(self.device_str)
            if hasattr(self.model, "eval"):
                self.model.eval()

    def predict(
        self,
        source: Any,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> Union[PredictionResult, List[PredictionResult]]:
        """Run object detection inference on image source(s).

        Args:
            source: Single image, image path, list of paths, directory, NumPy array, or PIL Image.

        Returns:
            PredictionResult instance or list of PredictionResult instances.
        """
        if self.model is None:
            raise InferenceException(message="Model must be assigned to predictor before calling predict()")

        # 1. Preprocess input images into tensor batch
        input_tensor, metadata_list = self.preprocess(source)

        # 2. Forward pass timing
        t0 = time.perf_counter()
        if torch is not None and hasattr(input_tensor, "to"):
            input_tensor = input_tensor.to(self.device_str)

        if torch is not None:
            with torch.no_grad():
                raw_outputs = self.model(input_tensor)
        else:
            raw_outputs = self.model(input_tensor)
        t_infer = (time.perf_counter() - t0) * 1000.0

        # 3. Decode raw predictions
        t1 = time.perf_counter()
        boxes, scores = self.decoder.decode(raw_outputs)
        t_decode = (time.perf_counter() - t1) * 1000.0

        # 4. Apply Non-Maximum Suppression (NMS)
        t2 = time.perf_counter()
        orig_shape = metadata_list[0]["orig_shape"]
        results = self.nms(
            boxes=boxes,
            scores=scores,
            orig_shape=orig_shape,
            letterbox_shape=self.imgsz,
            class_names=self.class_names,
        )
        t_nms = (time.perf_counter() - t2) * 1000.0

        # Populate timing and metadata
        for res in results:
            res.timing = {
                "inference_ms": t_infer,
                "decode_ms": t_decode,
                "nms_ms": t_nms,
                "total_ms": t_infer + t_decode + t_nms,
            }
            res.metadata = {
                "device": self.device_str,
                "model": self.model.__class__.__name__,
            }

        return results[0] if not isinstance(source, list) and len(results) == 1 else results

    def preprocess(self, source: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        """Preprocess diverse image inputs into normalized PyTorch image tensor."""
        image_list = []
        metadata_list = []

        # Handle torch.Tensor input directly
        if torch is not None and isinstance(source, torch.Tensor):
            if source.ndim == 4:
                meta = [{"orig_shape": (source.shape[2], source.shape[3], 3)} for _ in range(source.shape[0])]
                return source, meta
            elif source.ndim == 3:
                meta = [{"orig_shape": (source.shape[1], source.shape[2], 3)}]
                return source.unsqueeze(0), meta

        # Handle directory path
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                image_paths = sorted([p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])
                for p in image_paths:
                    img = Image.open(p).convert("RGB")
                    image_list.append(np.array(img))
                    metadata_list.append({"orig_shape": (img.height, img.width, 3), "path": str(p)})
            elif path.is_file():
                img = Image.open(path).convert("RGB")
                image_list.append(np.array(img))
                metadata_list.append({"orig_shape": (img.height, img.width, 3), "path": str(path)})
        elif isinstance(source, Image.Image):
            image_list.append(np.array(source.convert("RGB")))
            metadata_list.append({"orig_shape": (source.height, source.width, 3)})
        elif isinstance(source, np.ndarray):
            if source.ndim == 3:
                image_list.append(source)
                metadata_list.append({"orig_shape": source.shape})
            elif source.ndim == 4:
                for img in source:
                    image_list.append(img)
                    metadata_list.append({"orig_shape": img.shape})
        elif isinstance(source, list):
            for item in source:
                tensor, meta = self.preprocess(item)
                image_list.append(tensor)
                metadata_list.extend(meta)
            if torch is not None:
                return torch.cat(image_list, dim=0), metadata_list

        # Letterbox and convert to NCHW PyTorch tensor
        tensors = []
        for img in image_list:
            resized_img = self.letterbox_resize(img, self.imgsz)
            chw = resized_img.transpose(2, 0, 1).astype(np.float32) / 255.0
            if torch is not None:
                tensors.append(torch.from_numpy(chw).unsqueeze(0))

        batch_tensor = torch.cat(tensors, dim=0) if torch is not None else None
        return batch_tensor, metadata_list

    @staticmethod
    def letterbox_resize(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize and pad image into square letterboxed canvas."""
        target_h, target_w = target_shape
        h, w = img.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        pil_img = Image.fromarray(img)
        resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
        resized_arr = np.array(resized)

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_arr
        return canvas


    def predict_batch(self, images: List[Any], **kwargs: Any) -> List[List[Dict[str, Any]]]:
        """Run batch image inference.

        Args:
            images: List of input images.

        Returns:
            List of prediction dictionary lists per image.
        """
        results = self.predict(images, **kwargs)
        if not isinstance(results, list):
            results = [results]
        batch_out = []
        for res in results:
            if isinstance(res, PredictionResult):
                batch_out.append([res.to_dict()])
            elif isinstance(res, list):
                batch_out.append([r.to_dict() if hasattr(r, "to_dict") else r for r in res])
            else:
                batch_out.append([res])
        return batch_out


BatchPredictor = AethelPredictor
InferencePipeline = AethelPredictor
PredictionSession = AethelPredictor

__all__ = [
    "AethelPredictor",
    "BatchPredictor",
    "InferencePipeline",
    "PredictionSession",
]
