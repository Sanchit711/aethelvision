"""Non-Maximum Suppression (NMS) for Aethel Detector.

Performs Class-aware Non-Maximum Suppression (NMS) over candidate bounding boxes
and confidence scores using pure PyTorch operations.
"""

from typing import List, Optional

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Intersection over Union (IoU) matrix between two box sets.

    Args:
        boxes1: Tensor of shape (N, 4) containing [x1, y1, x2, y2].
        boxes2: Tensor of shape (M, 4) containing [x1, y1, x2, y2].

    Returns:
        IoU matrix tensor of shape (N, M).
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def nms_torch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
) -> torch.Tensor:
    """Pure PyTorch implementation of Non-Maximum Suppression.

    Args:
        boxes: Tensor of shape (N, 4) with [x1, y1, x2, y2].
        scores: Tensor of shape (N,) with confidence scores.
        iou_threshold: Overlap IoU threshold float.

    Returns:
        1D Tensor of kept box indices.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    classes: Optional[List[int]] = None,
    max_det: int = 300,
) -> List[torch.Tensor]:
    """Perform Class-aware Non-Maximum Suppression over batched predictions.

    Args:
        prediction: Decoded prediction tensor of shape (B, Total_Anchors, 4 + num_classes).
                    Format per anchor is [x1, y1, x2, y2, score_0, score_1, ...].
        conf_threshold: Candidate score threshold float (default 0.25).
        iou_threshold: IoU overlap suppression threshold float (default 0.45).
        classes: Optional list of target class indices to filter.
        max_det: Maximum detection output count per image (default 300).

    Returns:
        List of tensors [ (N_0, 6), (N_1, 6), ... ] where each row is
        [x1, y1, x2, y2, confidence, class_id].
    """
    bs = prediction.shape[0]
    num_classes = prediction.shape[2] - 4
    output = []

    for i in range(bs):
        pred = prediction[i]  # (Total_Anchors, 4 + num_classes)

        boxes = pred[:, :4]
        class_scores = pred[:, 4:]

        # Find max class score and class ID per anchor
        scores, class_ids = torch.max(class_scores, dim=1)

        # Apply confidence threshold mask
        mask = scores >= conf_threshold
        if not mask.any():
            output.append(torch.empty((0, 6), device=prediction.device))
            continue

        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Optional class filtering
        if classes is not None:
            cls_mask = torch.tensor([c in classes for c in class_ids.tolist()], device=prediction.device)
            boxes = boxes[cls_mask]
            scores = scores[cls_mask]
            class_ids = class_ids[cls_mask]

        if boxes.numel() == 0:
            output.append(torch.empty((0, 6), device=prediction.device))
            continue

        # Class-aware NMS offset trick: offset boxes by class_id * max_coord
        max_coord = boxes.max() + 1.0
        offsets = class_ids.float() * max_coord
        offset_boxes = boxes + offsets[:, None]

        keep_indices = nms_torch(offset_boxes, scores, iou_threshold=iou_threshold)
        keep_indices = keep_indices[:max_det]

        # Construct final output tensor [x1, y1, x2, y2, confidence, class_id]
        final_boxes = boxes[keep_indices]
        final_scores = scores[keep_indices].unsqueeze(1)
        final_classes = class_ids[keep_indices].float().unsqueeze(1)

        det = torch.cat([final_boxes, final_scores, final_classes], dim=1)
        output.append(det)

    return output


__all__ = ["non_max_suppression", "box_iou", "nms_torch"]
