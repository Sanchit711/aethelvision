"""Post-Processing and Non-Maximum Suppression (NMS) for Aethel Vision Framework.

Filters candidate bounding boxes using confidence thresholding, class filtering,
class-aware/class-agnostic Non-Maximum Suppression (NMS), and coordinate unboxing.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aethel.inference.result import PredictionResult
from aethel.utils.logger import get_logger

logger = get_logger("aethel.inference.postprocess")


class NonMaximumSuppression:
    """Class-aware and class-agnostic Non-Maximum Suppression engine."""

    def __init__(
        self,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        iom_threshold: float = 0.60,
        max_detections: int = 300,
        class_agnostic: bool = False,
    ) -> None:
        """Initialize NonMaximumSuppression.

        Args:
            conf_threshold: Minimum class score confidence threshold (default 0.45).
            iou_threshold: IoU overlap suppression threshold (default 0.45).
            iom_threshold: Containment IoM suppression threshold (default 0.60).
            max_detections: Maximum detections count limit per image (default 300).
            class_agnostic: Whether to suppress boxes regardless of class ID.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.iom_threshold = iom_threshold
        self.max_detections = max_detections
        self.class_agnostic = class_agnostic

    def __call__(
        self,
        boxes: Any,
        scores: Any,
        orig_shape: Tuple[int, int],
        letterbox_shape: Tuple[int, int],
        class_names: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """Filter and process candidate boxes across batch items.

        Args:
            boxes: Candidate boxes tensor of shape (B, N, 4) [x1, y1, x2, y2].
            scores: Candidate class scores tensor of shape (B, N, num_classes).
            orig_shape: Original input image shape (H, W).
            letterbox_shape: Model input tensor shape (H, W).
            class_names: Optional list of class name strings.

        Returns:
            List of PredictionResult instances (one per batch item).
        """
        if torch is None or boxes is None or scores is None:
            return [PredictionResult()]

        batch_size = boxes.shape[0]
        results = []

        for b in range(batch_size):
            b_boxes = boxes[b]  # (N, 4)
            b_scores = scores[b]  # (N, num_classes)

            # Max confidence score and class index per box candidate
            max_scores, class_ids = torch.max(b_scores, dim=-1)
            keep_mask = max_scores >= self.conf_threshold

            filt_boxes = b_boxes[keep_mask]
            filt_scores = max_scores[keep_mask]
            filt_classes = class_ids[keep_mask]

            if filt_boxes.shape[0] == 0:
                results.append(
                    PredictionResult(
                        orig_shape=orig_shape,
                        image_shape=letterbox_shape,
                    )
                )
                continue

            # Class offset trick for class-aware NMS
            if not self.class_agnostic:
                max_coordinate = filt_boxes.max()
                offsets = filt_classes.to(filt_boxes.dtype) * (max_coordinate + 1.0)
                nms_boxes = filt_boxes + offsets.unsqueeze(-1)
            else:
                nms_boxes = filt_boxes

            # Torch NMS execution with containment suppression
            keep_indices = torchvision_nms(nms_boxes, filt_scores, self.iou_threshold, self.iom_threshold)
            keep_indices = keep_indices[: self.max_detections]

            final_boxes = filt_boxes[keep_indices]
            final_scores = filt_scores[keep_indices]
            final_classes = filt_classes[keep_indices]

            # Rescale bounding boxes back to original image spatial dimensions
            rescaled_boxes = self.rescale_boxes(
                final_boxes,
                orig_shape=orig_shape,
                letterbox_shape=letterbox_shape,
            )

            # Map class indices to string labels
            labels = (
                [class_names[c] if class_names and c < len(class_names) else f"class_{c}" for c in final_classes.cpu().numpy()]
                if hasattr(final_classes, "cpu")
                else []
            )

            res = PredictionResult(
                boxes=rescaled_boxes.cpu().numpy() if hasattr(rescaled_boxes, "cpu") else rescaled_boxes,
                scores=final_scores.cpu().numpy() if hasattr(final_scores, "cpu") else final_scores,
                classes=final_classes.cpu().numpy() if hasattr(final_classes, "cpu") else final_classes,
                labels=labels,
                orig_shape=orig_shape,
                image_shape=letterbox_shape,
            )
            results.append(res)

        return results

    @staticmethod
    def rescale_boxes(
        boxes: Any,
        orig_shape: Tuple[int, int],
        letterbox_shape: Tuple[int, int],
    ) -> Any:
        """Rescale letterboxed prediction coordinates back to original image dimensions."""
        if torch is None:
            return boxes

        orig_h, orig_w = orig_shape[:2]
        net_h, net_w = letterbox_shape[:2]

        gain = min(net_w / orig_w, net_h / orig_h)
        pad_w = (net_w - orig_w * gain) / 2.0
        pad_h = (net_h - orig_h * gain) / 2.0

        rescaled = boxes.clone()
        rescaled[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
        rescaled[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain

        rescaled[:, [0, 2]] = rescaled[:, [0, 2]].clamp(min=0, max=orig_w)
        rescaled[:, [1, 3]] = rescaled[:, [1, 3]].clamp(min=0, max=orig_h)

        return rescaled


def torchvision_nms(boxes: Any, scores: Any, iou_threshold: float, iom_threshold: float = 0.60) -> Any:
    """Native PyTorch NMS implementation with Enclosure-Aware Containment Suppression.
    
    Suppresses candidates if standard IoU >= iou_threshold OR
    Containment IoM (Intersection over Minimum) >= iom_threshold.
    """
    if torch is None or boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0].item()
        keep.append(i)

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter + 1e-7
        ovr = inter / union

        min_area = torch.minimum(areas[i], areas[order[1:]]) + 1e-7
        iom = inter / min_area

        suppress = (ovr >= iou_threshold) | (iom >= iom_threshold)
        keep_mask = ~suppress
        ids = keep_mask.nonzero(as_tuple=False).squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class SoftNMS(NonMaximumSuppression):
    """Soft-NMS penalty function hook."""

    def __init__(self, sigma: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sigma = sigma


__all__ = ["NonMaximumSuppression", "SoftNMS", "torchvision_nms"]
