"""Decoupled Anchor-Free Detection Head for Aethel Vision Framework.

Predicts multi-scale raw classification logits and bounding box regression logits (DFL)
across pyramid feature levels (N3, N4, N5).
Outputs raw prediction tensors only (zero NMS, zero decoding, zero loss calculation).
"""

from typing import Any, List, Tuple, Union

try:
    import torch
    import torch.nn as nn
    _Module = nn.Module
except Exception:
    torch = None  # type: ignore
    class _Dummy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return args[0] if args else None
    class _NN:
        Module = _Dummy
        ModuleList = list
        Sequential = _Dummy
        Conv2d = _Dummy
    nn = _NN()  # type: ignore
    _Module = nn.Module

from aethel.layers.conv import Conv


class DecoupledDetectHead(_Module):  # type: ignore
    """Decoupled anchor-free multi-scale object detection head."""

    def __init__(
        self,
        num_classes: int = 80,
        in_channels_list: Tuple[int, int, int] = (128, 256, 512),
        reg_max: int = 16,
        act: Union[str, bool, None] = True,
        norm: bool = True,
    ) -> None:
        """Initialize DecoupledDetectHead."""
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes + 4 * reg_max
        self.strides = (8, 16, 32)

        self.cls_convs = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.reg_convs = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.cls_preds = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []
        self.reg_preds = nn.ModuleList() if hasattr(nn, "ModuleList") and isinstance(nn.ModuleList, type) else []

        for in_ch in in_channels_list:
            self.cls_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.cls_preds.append(nn.Conv2d(in_ch, num_classes, kernel_size=1))

            self.reg_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.reg_preds.append(nn.Conv2d(in_ch, 4 * reg_max, kernel_size=1))

    def forward(
        self,
        features: Tuple[Any, Any, Any],
    ) -> List[Any]:
        """Forward pass predicting raw classification and regression logits."""
        if torch is None:
            return list(features)

        outputs = []
        for i, x in enumerate(features):
            cls_feat = self.cls_convs[i](x)
            cls_out = self.cls_preds[i](cls_feat)

            reg_feat = self.reg_convs[i](x)
            reg_out = self.reg_preds[i](reg_feat)

            out = torch.cat([cls_out, reg_out], dim=1)
            outputs.append(out)

        return outputs


__all__ = ["DecoupledDetectHead"]
