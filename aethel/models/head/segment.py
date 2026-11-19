"""Indigenous Real-Time Instance Segmentation Head for Aethel Vision Platform.

Implements high-speed Prototype Mask Architecture (Proto-Net + Mask Coefficients):
- Generates 32 global prototype feature masks on P3.
- Predicts 32 instance-specific mask coefficients alongside bounding boxes.
- Assembles pixel-accurate binary masks via matrix multiplication and bounding-box cropping.
- 100% indigenous PyTorch implementation (zero external framework wrappers).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _Module = nn.Module
except ImportError:
    torch = None  # type: ignore
    class _Dummy:
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any: return args[0] if args else None
    class _NN:
        Module = _Dummy
        ModuleList = list
        Sequential = _Dummy
        Conv2d = _Dummy
        ConvTranspose2d = _Dummy
        ReLU = _Dummy
        Sigmoid = _Dummy
    nn = _NN()  # type: ignore
    _Module = nn.Module

from aethel.layers.conv import Conv
import numpy as np


def assemble_masks(
    protos: Any,
    mask_coeffs: Any,
    boxes: Optional[Any] = None,
) -> Any:
    """Assemble instance masks via matrix multiplication and crop to bounding boxes.

    Args:
        protos: Prototype masks tensor or ndarray (B, num_protos, H, W) or (num_protos, H, W).
        mask_coeffs: Mask coefficients (B, N, num_protos) or (N, num_protos).
        boxes: Optional bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        Assembled sigmoid masks bounded in [0, 1].
    """
    if torch is not None and isinstance(protos, torch.Tensor):
        if protos.dim() == 3:
            protos = protos.unsqueeze(0)
        if mask_coeffs.dim() == 2:
            mask_coeffs = mask_coeffs.unsqueeze(0)

        B, num_protos, H_mask, W_mask = protos.shape
        flat_protos = protos.view(B, num_protos, -1)
        masks = torch.bmm(mask_coeffs, flat_protos)
        masks = torch.sigmoid(masks).view(B, -1, H_mask, W_mask)

        if boxes is not None:
            if boxes.dim() == 2:
                boxes = boxes.unsqueeze(0)
            N = boxes.shape[1]
            for b in range(B):
                for i in range(N):
                    x1, y1, x2, y2 = boxes[b, i]
                    # Check if normalized or absolute
                    if x2 <= 1.0 and y2 <= 1.0:
                        mx1 = int(float(x1) * W_mask)
                        my1 = int(float(y1) * H_mask)
                        mx2 = int(float(x2) * W_mask)
                        my2 = int(float(y2) * H_mask)
                    else:
                        mx1 = max(0, min(W_mask, int(float(x1))))
                        my1 = max(0, min(H_mask, int(float(y1))))
                        mx2 = max(0, min(W_mask, int(float(x2))))
                        my2 = max(0, min(H_mask, int(float(y2))))

                    if my1 > 0:
                        masks[b, i, :my1, :] = 0.0
                    if my2 < H_mask:
                        masks[b, i, my2:, :] = 0.0
                    if mx1 > 0:
                        masks[b, i, :, :mx1] = 0.0
                    if mx2 < W_mask:
                        masks[b, i, :, mx2:] = 0.0

        return masks.squeeze(0) if masks.shape[0] == 1 else masks

    # NumPy fallback
    p_arr = np.asarray(protos, dtype=np.float32)
    c_arr = np.asarray(mask_coeffs, dtype=np.float32)
    if p_arr.ndim == 4:
        p_arr = p_arr[0]
    if c_arr.ndim == 3:
        c_arr = c_arr[0]

    num_protos, H_mask, W_mask = p_arr.shape
    p_flat = p_arr.reshape(num_protos, -1)
    masks_flat = c_arr @ p_flat  # (N, H*W)
    masks = 1.0 / (1.0 + np.exp(-masks_flat))
    masks = masks.reshape(-1, H_mask, W_mask)

    if boxes is not None:
        b_arr = np.asarray(boxes, dtype=np.float32)
        if b_arr.ndim == 3:
            b_arr = b_arr[0]
        for i, box in enumerate(b_arr):
            x1, y1, x2, y2 = box
            if x2 <= 1.0 and y2 <= 1.0:
                mx1, my1 = int(x1 * W_mask), int(y1 * H_mask)
                mx2, my2 = int(x2 * W_mask), int(y2 * H_mask)
            else:
                mx1 = max(0, min(W_mask, int(x1)))
                my1 = max(0, min(H_mask, int(y1)))
                mx2 = max(0, min(W_mask, int(x2)))
                my2 = max(0, min(H_mask, int(y2)))

            if my1 > 0:
                masks[i, :my1, :] = 0.0
            if my2 < H_mask:
                masks[i, my2:, :] = 0.0
            if mx1 > 0:
                masks[i, :, :mx1] = 0.0
            if mx2 < W_mask:
                masks[i, :, mx2:] = 0.0

    return masks


class ProtoNet(_Module):  # type: ignore
    """Prototype Mask Generation Network.
    
    Generates high-resolution prototype feature masks from P3 pyramid features.
    """

    def __init__(self, in_channels: int = 128, num_protos: int = 32) -> None:
        super().__init__()
        self.num_protos = num_protos
        if torch is not None and hasattr(nn, "ConvTranspose2d"):
            self.net = nn.Sequential(
                Conv(in_channels, in_channels, kernel_size=3, stride=1),
                Conv(in_channels, in_channels, kernel_size=3, stride=1),
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                Conv(in_channels, in_channels, kernel_size=3, stride=1),
                nn.Conv2d(in_channels, num_protos, kernel_size=1),
            )
        else:
            self.net = None

    def forward(self, x: Any) -> Any:
        if torch is None or self.net is None:
            return x
        return self.net(x)


class AethelSegmentHead(_Module):  # type: ignore
    """Indigenous Decoupled Multi-Scale Instance Segmentation Head."""

    def __init__(
        self,
        num_classes: int = 80,
        num_protos: int = 32,
        in_channels_list: Tuple[int, int, int] = (128, 256, 512),
        reg_max: int = 16,
        act: Union[str, bool, None] = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_protos = num_protos
        self.reg_max = reg_max
        self.strides = (8, 16, 32)

        # Prototype network attached to highest-resolution feature (P3)
        self.proto_net = ProtoNet(in_channels=in_channels_list[0], num_protos=num_protos)

        self.cls_convs = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.reg_convs = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.mask_convs = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []

        self.cls_preds = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.reg_preds = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.mask_preds = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []

        for in_ch in in_channels_list:
            # 1. Classification Branch
            self.cls_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.cls_preds.append(nn.Conv2d(in_ch, num_classes, kernel_size=1))

            # 2. Bounding Box Regression Branch
            self.reg_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.reg_preds.append(nn.Conv2d(in_ch, 4 * reg_max, kernel_size=1))

            # 3. Mask Coefficients Branch
            self.mask_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.mask_preds.append(nn.Conv2d(in_ch, num_protos, kernel_size=1))

    def forward(self, features: Tuple[Any, Any, Any]) -> Dict[str, Any]:
        """Forward pass predicting prototypes, mask coefficients, classes, and boxes."""
        if torch is None:
            return {"protos": features[0], "cls": list(features), "box": list(features), "mask_coeffs": list(features)}

        # 1. Generate high-resolution prototype masks from P3
        protos = self.proto_net(features[0])

        cls_outputs = []
        reg_outputs = []
        mask_outputs = []

        for i, x in enumerate(features):
            cls_feat = self.cls_convs[i](x)
            cls_outputs.append(self.cls_preds[i](cls_feat))

            reg_feat = self.reg_convs[i](x)
            reg_outputs.append(self.reg_preds[i](reg_feat))

            mask_feat = self.mask_convs[i](x)
            mask_outputs.append(self.mask_preds[i](mask_feat))

        return {
            "protos": protos,
            "cls": cls_outputs,
            "box": reg_outputs,
            "mask_coeffs": mask_outputs,
        }

    assemble_masks = staticmethod(assemble_masks)


__all__ = ["AethelSegmentHead", "ProtoNet", "assemble_masks"]
