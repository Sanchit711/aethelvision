"""Post-Processing Coordinate Decoder & NMS Wrapper for Aethel Detector.

Decodes raw multi-scale detector output logits into spatial bounding box coordinates
and applies Non-Maximum Suppression (NMS) confidence thresholding and class filtering.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from aethel_detector.head.decoder import HeadDecoder
from aethel_detector.postprocess.nms import non_max_suppression


class PostprocessDecoder(nn.Module):
    """Post-processing pipeline decoding raw head outputs and applying NMS filtering."""

    def __init__(
        self,
        reg_max: int = 16,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        """Initialize PostprocessDecoder.

        Args:
            reg_max: DFL regression max bin bound integer (default 16).
            conf_threshold: Default confidence threshold float (default 0.25).
            iou_threshold: Default IoU NMS threshold float (default 0.45).
        """
        super().__init__()
        self.head_decoder = HeadDecoder(reg_max=reg_max)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def forward(
        self,
        predictions: Union[List[torch.Tensor], torch.Tensor],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        """Decode raw prediction tensors and apply Non-Maximum Suppression (NMS).

        Args:
            predictions: List of raw prediction tensors from DecoupledDetectHead or decoded tensor.
            conf_threshold: Optional override confidence threshold float.
            iou_threshold: Optional override IoU threshold float.
            classes: Optional list of target class indices to filter.
            max_det: Maximum detections per batch item.

        Returns:
            List of tensors [ (N_0, 6), (N_1, 6), ... ] where each row is
            [x1, y1, x2, y2, confidence, class_id].
        """
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold

        if isinstance(predictions, list):
            decoded_preds = self.head_decoder(predictions)
        else:
            decoded_preds = predictions

        return non_max_suppression(
            prediction=decoded_preds,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            classes=classes,
            max_det=max_det,
        )


__all__ = ["PostprocessDecoder"]
