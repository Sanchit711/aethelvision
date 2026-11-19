"""Canonical Batch Adapter and Normalizer for Aethel Vision Framework.

Provides ONE unified, production-grade batch interface across all Datasets, DataLoaders,
Trainers, Runners, Losses, Metrics, and Validation pipelines.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aethel.utils.exceptions import (
    BatchFormatException,
    DeviceMismatchException,
    InvalidBatchException,
)


@dataclass
class BatchData:
    """Canonical batch container for Aethel Vision detection pipeline."""

    images: Any
    gt_labels: Any
    gt_bboxes: Any
    mask_gt: Any
    raw_targets: Any = None
    batch_size: int = 1

    def to(self, device: Any, non_blocking: bool = True) -> "BatchData":
        """Move all batch tensors to specified device."""
        if torch is None or device is None:
            return self

        dev = torch.device(device) if isinstance(device, str) else device

        def _to_dev(t: Any) -> Any:
            if hasattr(t, "to"):
                return t.to(dev, non_blocking=non_blocking)
            elif isinstance(t, dict):
                return {k: _to_dev(v) for k, v in t.items()}
            elif isinstance(t, list):
                return [_to_dev(v) for v in t]
            elif isinstance(t, tuple):
                return tuple(_to_dev(v) for v in t)
            return t

        return BatchData(
            images=_to_dev(self.images),
            gt_labels=_to_dev(self.gt_labels),
            gt_bboxes=_to_dev(self.gt_bboxes),
            mask_gt=_to_dev(self.mask_gt),
            raw_targets=_to_dev(self.raw_targets) if self.raw_targets is not None else None,
            batch_size=self.batch_size,
        )

    @property
    def targets_dict(self) -> Dict[str, Any]:
        """Return standardized targets dictionary for loss/evaluator functions."""
        return {
            "cls_targets": self.gt_labels,
            "target_boxes": self.gt_bboxes,
            "mask_gt": self.mask_gt,
            "raw_targets": self.raw_targets,
        }

    def as_2tuple(self) -> Tuple[Any, Any]:
        """Return (images, targets_dict) tuple for 2-element API compatibility."""
        return self.images, self.targets_dict

    def as_4tuple(self) -> Tuple[Any, Any, Any, Any]:
        """Return (images, gt_labels, gt_bboxes, mask_gt) tuple for 4-element API compatibility."""
        return self.images, self.gt_labels, self.gt_bboxes, self.mask_gt

    def __getitem__(self, idx: int) -> Any:
        if idx == 0:
            return self.images
        elif idx == 1:
            return self.gt_labels
        elif idx == 2:
            return self.gt_bboxes
        elif idx == 3:
            return self.mask_gt
        raise IndexError(f"BatchData index {idx} out of range [0..3]")

    def __iter__(self):
        yield self.images
        yield self.gt_labels
        yield self.gt_bboxes
        yield self.mask_gt

    def __len__(self) -> int:
        return 4


def _targets_tensor_to_padded(
    targets_tensor: Any, batch_size: int, device: Any = None
) -> Tuple[Any, Any, Any]:
    """Convert [N, 6] tensor [batch_idx, class_id, x1, y1, x2, y2] into padded batch tensors."""
    if torch is None:
        return targets_tensor, targets_tensor, targets_tensor

    if not isinstance(targets_tensor, torch.Tensor) or targets_tensor.numel() == 0 or targets_tensor.ndim != 2:
        zero_lbl = torch.zeros((batch_size, 0), dtype=torch.int64, device=device)
        zero_box = torch.zeros((batch_size, 0, 4), dtype=torch.float32, device=device)
        zero_msk = torch.zeros((batch_size, 0), dtype=torch.float32, device=device)
        return zero_lbl, zero_box, zero_msk

    dev = targets_tensor.device if device is None else device

    # Group targets per batch_idx
    per_batch_labels: List[torch.Tensor] = []
    per_batch_boxes: List[torch.Tensor] = []
    max_objs = 0

    for b in range(batch_size):
        b_mask = targets_tensor[:, 0] == b
        b_targets = targets_tensor[b_mask]
        num_b = b_targets.shape[0]
        if num_b > max_objs:
            max_objs = num_b
        if num_b > 0:
            per_batch_labels.append(b_targets[:, 1].long())
            per_batch_boxes.append(b_targets[:, 2:6].float())
        else:
            per_batch_labels.append(torch.zeros((0,), dtype=torch.int64, device=dev))
            per_batch_boxes.append(torch.zeros((0, 4), dtype=torch.float32, device=dev))

    max_objs = max(max_objs, 1)
    padded_labels = torch.zeros((batch_size, max_objs), dtype=torch.int64, device=dev)
    padded_boxes = torch.zeros((batch_size, max_objs, 4), dtype=torch.float32, device=dev)
    padded_masks = torch.zeros((batch_size, max_objs), dtype=torch.float32, device=dev)

    for b in range(batch_size):
        lbls = per_batch_labels[b]
        boxes = per_batch_boxes[b]
        n = lbls.shape[0]
        if n > 0:
            padded_labels[b, :n] = lbls
            padded_boxes[b, :n] = boxes
            padded_masks[b, :n] = 1.0

    return padded_labels, padded_boxes, padded_masks


def normalize_batch(batch: Any, device: Optional[Any] = None) -> BatchData:
    """Normalize any dataset or dataloader output batch into canonical BatchData.

    Supports:
    - BatchData instances
    - 4-tuples/lists (images, gt_labels, gt_bboxes, mask_gt)
    - 2-tuples/lists (images, targets) where targets is tensor [N,6], dict, or list
    - Dicts {"images": ..., "gt_labels": ..., "gt_bboxes": ..., "mask_gt": ...}

    Args:
        batch: Raw batch output from DataLoader iteration.
        device: Optional target device string or torch.device.

    Returns:
        Canonical BatchData object.

    Raises:
        BatchFormatException: If batch cannot be parsed into canonical representation.
    """
    if isinstance(batch, BatchData):
        return batch.to(device) if device is not None else batch

    if torch is not None and isinstance(batch, torch.Tensor):
        raise BatchFormatException(
            message="Received raw torch.Tensor instead of tuple/dict batch.",
            details=f"Tensor shape: {batch.shape}",
        )

    # 1. Dict Format
    if isinstance(batch, dict):
        images = batch.get("images") if "images" in batch else batch.get("image", batch.get("img"))
        if images is None:
            raise BatchFormatException(
                message="Batch dictionary missing required 'images' key.",
                details=f"Received keys: {list(batch.keys())}",
            )
        gt_labels = batch.get("gt_labels", batch.get("labels", batch.get("cls_targets")))
        gt_bboxes = batch.get("gt_bboxes", batch.get("boxes", batch.get("target_boxes")))
        mask_gt = batch.get("mask_gt", batch.get("mask", batch.get("masks")))
        bs = len(images) if hasattr(images, "__len__") else 1
        res = BatchData(images=images, gt_labels=gt_labels, gt_bboxes=gt_bboxes, mask_gt=mask_gt, raw_targets=batch, batch_size=bs)
        return res.to(device) if device is not None else res

    # 2. Sequence (Tuple or List) Formats
    if isinstance(batch, (tuple, list)):
        n_items = len(batch)
        if n_items == 4:
            images, gt_labels, gt_bboxes, mask_gt = batch
            bs = len(images) if hasattr(images, "__len__") else 1
            res = BatchData(images=images, gt_labels=gt_labels, gt_bboxes=gt_bboxes, mask_gt=mask_gt, raw_targets=batch, batch_size=bs)
            return res.to(device) if device is not None else res

        elif n_items == 2:
            images, targets = batch
            bs = len(images) if hasattr(images, "__len__") else 1

            if torch is not None and isinstance(targets, torch.Tensor):
                gt_labels, gt_bboxes, mask_gt = _targets_tensor_to_padded(targets, batch_size=bs, device=device)
                res = BatchData(images=images, gt_labels=gt_labels, gt_bboxes=gt_bboxes, mask_gt=mask_gt, raw_targets=targets, batch_size=bs)
                return res.to(device) if device is not None else res

            elif isinstance(targets, dict):
                gt_labels = targets.get("gt_labels", targets.get("labels", targets.get("cls_targets")))
                gt_bboxes = targets.get("gt_bboxes", targets.get("boxes", targets.get("target_boxes")))
                mask_gt = targets.get("mask_gt", targets.get("mask", targets.get("masks")))
                res = BatchData(images=images, gt_labels=gt_labels, gt_bboxes=gt_bboxes, mask_gt=mask_gt, raw_targets=targets, batch_size=bs)
                return res.to(device) if device is not None else res

            elif isinstance(targets, (list, tuple)):
                gt_labels, gt_bboxes, mask_gt = _targets_tensor_to_padded(targets, batch_size=bs, device=device)
                res = BatchData(images=images, gt_labels=gt_labels, gt_bboxes=gt_bboxes, mask_gt=mask_gt, raw_targets=targets, batch_size=bs)
                return res.to(device) if device is not None else res

            else:
                res = BatchData(images=images, gt_labels=targets, gt_bboxes=targets, mask_gt=targets, raw_targets=targets, batch_size=bs)
                return res.to(device) if device is not None else res

        elif n_items == 1:
            return normalize_batch(batch[0], device=device)

        else:
            raise BatchFormatException(
                message=f"Invalid tuple batch length: expected 2 or 4 items, got {n_items}.",
                details=f"Received item types: {[type(x).__name__ for x in batch]}",
            )

    raise BatchFormatException(
        message=f"Unsupported batch type: {type(batch).__name__}.",
        details="Supported formats: 2-tuple (images, targets), 4-tuple (images, labels, boxes, masks), dict, or BatchData.",
    )


def normalize_batch_2tuple(batch: Any, device: Optional[Any] = None) -> Tuple[Any, Any]:
    """Helper returning canonical (images, targets_dict) tuple."""
    b = normalize_batch(batch, device=device)
    return b.as_2tuple()


def normalize_batch_4tuple(batch: Any, device: Optional[Any] = None) -> Tuple[Any, Any, Any, Any]:
    """Helper returning canonical (images, gt_labels, gt_bboxes, mask_gt) tuple."""
    b = normalize_batch(batch, device=device)
    return b.as_4tuple()


def move_to_device(obj: Any, device: Any, non_blocking: bool = True) -> Any:
    """Recursively move arbitrary structures (Tensor, Dict, List, Tuple, NamedTuple, Dataclass, BatchData) to device.

    Args:
        obj: Target object containing tensors.
        device: Target device string or torch.device.
        non_blocking: Whether to perform asynchronous memory copies.

    Returns:
        New instance with all internal tensors moved to target device.
    """
    if torch is None or device is None:
        return obj

    dev = torch.device(device) if isinstance(device, str) else device

    if isinstance(obj, BatchData):
        return obj.to(dev, non_blocking=non_blocking)
    elif hasattr(obj, "to") and callable(getattr(obj, "to")):
        return obj.to(dev, non_blocking=non_blocking)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, dev, non_blocking=non_blocking) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, dev, non_blocking=non_blocking) for v in obj]
    elif isinstance(obj, tuple):
        if hasattr(obj, "_fields"):  # NamedTuple
            return type(obj)(*(move_to_device(v, dev, non_blocking=non_blocking) for v in obj))
        return tuple(move_to_device(v, dev, non_blocking=non_blocking) for v in obj)
    elif hasattr(obj, "__dataclass_fields__"):
        field_values = {f: move_to_device(getattr(obj, f), dev, non_blocking=non_blocking) for f in obj.__dataclass_fields__}
        return type(obj)(**field_values)
    return obj


def verify_device_alignment(model: Any, images: Any, targets: Any = None) -> None:
    """Assert pre-forward device alignment between model parameters and input tensors.

    Raises DeviceMismatchException with detailed diagnostic report if any mismatch is found.
    """
    if torch is None:
        return

    # Determine model expected device
    expected_dev = None
    if hasattr(model, "parameters"):
        try:
            expected_dev = next(model.parameters()).device
        except StopIteration:
            pass

    if expected_dev is None:
        return

    def _check_tensor_device(tensor_obj: Any, name: str) -> None:
        if isinstance(tensor_obj, torch.Tensor):
            if tensor_obj.device != expected_dev:
                raise DeviceMismatchException(
                    message=f"Pre-forward device mismatch detected for '{name}'.",
                    details=(
                        f"Expected device (from model parameters): {expected_dev} | "
                        f"Received device for '{name}': {tensor_obj.device} | "
                        f"Tensor shape: {tensor_obj.shape}, dtype: {tensor_obj.dtype}. "
                        f"Fix: Ensure move_to_device(batch, device) is called prior to model forward pass."
                    ),
                )
        elif isinstance(tensor_obj, dict):
            for k, v in tensor_obj.items():
                _check_tensor_device(v, f"{name}['{k}']")
        elif isinstance(tensor_obj, (list, tuple)):
            for idx, item in enumerate(tensor_obj):
                _check_tensor_device(item, f"{name}[{idx}]")
        elif isinstance(tensor_obj, BatchData):
            _check_tensor_device(tensor_obj.images, f"{name}.images")
            _check_tensor_device(tensor_obj.gt_labels, f"{name}.gt_labels")
            _check_tensor_device(tensor_obj.gt_bboxes, f"{name}.gt_bboxes")
            _check_tensor_device(tensor_obj.mask_gt, f"{name}.mask_gt")

    _check_tensor_device(images, "images")
    if targets is not None:
        _check_tensor_device(targets, "targets")


__all__ = [
    "BatchData",
    "normalize_batch",
    "normalize_batch_2tuple",
    "normalize_batch_4tuple",
    "move_to_device",
    "verify_device_alignment",
]
