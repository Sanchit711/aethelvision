"""Indigenous Oriented Bounding Box (OBB) Detection Head for Aethel Vision Platform.

Implements Rotated Bounding Box Detection (cx, cy, w, h, angle):
- Eliminates angle boundary discontinuity using trigonometric continuous representation (cos 2θ, sin 2θ).
- Converts rotated parameters (cx, cy, w, h, angle) into 4-corner polygon coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)].
- 100% indigenous PyTorch implementation (zero external framework dependencies).
- Ideal for aerial/drone imagery, ship detection, rotated text, and industrial component inspection.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
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
    nn = _NN()  # type: ignore
    _Module = nn.Module

from aethel.layers.conv import Conv


def decode_angle(angle_vector: Any) -> Any:
    """Decode continuous (cos 2θ, sin 2θ) vector into angle in radians [-π/2, π/2].
    Supports NumPy arrays and PyTorch tensors.
    """
    if torch is not None and isinstance(angle_vector, torch.Tensor):
        cos_2theta = angle_vector[..., 0]
        sin_2theta = angle_vector[..., 1]
        return 0.5 * torch.atan2(sin_2theta, cos_2theta)

    angle_arr = np.asarray(angle_vector)
    cos_2theta = angle_arr[..., 0]
    sin_2theta = angle_arr[..., 1]
    return 0.5 * np.arctan2(sin_2theta, cos_2theta)


def obb_to_corners(obb: Any) -> Any:
    """Convert rotated bounding boxes (N, 5) with (cx, cy, w, h, angle) into 4 corners (N, 4, 2).
    Supports NumPy arrays and PyTorch tensors.
    """
    if torch is not None and isinstance(obb, torch.Tensor):
        cx, cy, w, h, angle = obb[..., 0], obb[..., 1], obb[..., 2], obb[..., 3], obb[..., 4]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        hw = w / 2.0
        hh = h / 2.0

        # Offsets
        dx1 = -hw * cos_a + hh * sin_a
        dy1 = -hw * sin_a - hh * cos_a

        dx2 = hw * cos_a + hh * sin_a
        dy2 = hw * sin_a - hh * cos_a

        dx3 = hw * cos_a - hh * sin_a
        dy3 = hw * sin_a + hh * cos_a

        dx4 = -hw * cos_a - hh * sin_a
        dy4 = -hw * sin_a + hh * cos_a

        c1 = torch.stack([cx + dx1, cy + dy1], dim=-1)
        c2 = torch.stack([cx + dx2, cy + dy2], dim=-1)
        c3 = torch.stack([cx + dx3, cy + dy3], dim=-1)
        c4 = torch.stack([cx + dx4, cy + dy4], dim=-1)
        return torch.stack([c1, c2, c3, c4], dim=-2)

    obb_arr = np.asarray(obb, dtype=np.float32)
    cx, cy, w, h, angle = obb_arr[..., 0], obb_arr[..., 1], obb_arr[..., 2], obb_arr[..., 3], obb_arr[..., 4]
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    hw = w / 2.0
    hh = h / 2.0

    dx1 = -hw * cos_a + hh * sin_a
    dy1 = -hw * sin_a - hh * cos_a

    dx2 = hw * cos_a + hh * sin_a
    dy2 = hw * sin_a - hh * cos_a

    dx3 = hw * cos_a - hh * sin_a
    dy3 = hw * sin_a + hh * cos_a

    dx4 = -hw * cos_a - hh * sin_a
    dy4 = -hw * sin_a + hh * cos_a

    c1 = np.stack([cx + dx1, cy + dy1], axis=-1)
    c2 = np.stack([cx + dx2, cy + dy2], axis=-1)
    c3 = np.stack([cx + dx3, cy + dy3], axis=-1)
    c4 = np.stack([cx + dx4, cy + dy4], axis=-1)
    return np.stack([c1, c2, c3, c4], axis=-2)


class AethelOBBHead(_Module):  # type: ignore
    """Indigenous Decoupled Multi-Scale Oriented Bounding Box (OBB) Head."""

    decode_angle = staticmethod(decode_angle)
    obb_to_corners = staticmethod(obb_to_corners)

    def __init__(
        self,
        num_classes: int = 15,  # e.g., DOTA aerial dataset classes
        in_channels_list: Tuple[int, int, int] = (128, 256, 512),
        reg_max: int = 16,
        act: Union[str, bool, None] = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = (8, 16, 32)
        self.angle_dim = 2  # (cos 2θ, sin 2θ) for continuous boundary-free angle regression

        self.cls_convs = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.reg_convs = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.angle_convs = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []

        self.cls_preds = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.reg_preds = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.angle_preds = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []

        for in_ch in in_channels_list:
            # 1. Classification Branch
            self.cls_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.cls_preds.append(nn.Conv2d(in_ch, num_classes, kernel_size=1))

            # 2. Box Spatial Regression Branch (cx, cy, w, h via DFL)
            self.reg_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.reg_preds.append(nn.Conv2d(in_ch, 4 * reg_max, kernel_size=1))

            # 3. Angle Regression Branch (Continuous Trigonometric Vector)
            self.angle_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.angle_preds.append(nn.Conv2d(in_ch, self.angle_dim, kernel_size=1))

    def forward(self, features: Tuple[Any, Any, Any]) -> Dict[str, List[Any]]:
        """Forward pass predicting classification, horizontal boxes, and rotation angles."""
        if torch is None:
            return {"cls": list(features), "box": list(features), "angle": list(features)}

        cls_outputs = []
        reg_outputs = []
        angle_outputs = []

        for i, x in enumerate(features):
            cls_feat = self.cls_convs[i](x)
            cls_outputs.append(self.cls_preds[i](cls_feat))

            reg_feat = self.reg_convs[i](x)
            reg_outputs.append(self.reg_preds[i](reg_feat))

            angle_feat = self.angle_convs[i](x)
            angle_outputs.append(self.angle_preds[i](angle_feat))

        return {
            "cls": cls_outputs,
            "box": reg_outputs,
            "angle": angle_outputs,
        }


__all__ = ["AethelOBBHead", "decode_angle", "obb_to_corners"]
