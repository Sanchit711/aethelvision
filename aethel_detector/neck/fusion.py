"""Feature Fusion & Upsampling Modules for Aethel Detector Neck.

Provides upsampling layers, channel concatenation, and feature fusion blocks.
"""

from typing import Any, Optional, Union

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    class _Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any: return None
    nn = type("nn", (), {"Module": _Module})()  # type: ignore

try:
    from aethel_detector.layers.common import Conv
except Exception:
    Conv = None  # type: ignore


class UpsampleBlock(nn.Module):
    """Upsampling block scaling spatial dimensions by a target scale factor."""

    def __init__(self, scale_factor: int = 2, mode: str = "nearest") -> None:
        super().__init__()
        if torch is not None and hasattr(nn, "Upsample"):
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x: Any) -> Any:
        if hasattr(self, "upsample"):
            return self.upsample(x)
        return x


class FeatureFusionBlock(nn.Module):
    """Feature Fusion block concatenating lateral and upsampled features."""

    def __init__(
        self,
        in_channels: int = 0,
        out_channels: int = 0,
        in_channels_high: Optional[int] = None,
        in_channels_low: Optional[int] = None,
        act: Union[str, bool, Any, None] = True,
        norm: Union[str, bool, Any, None] = True,
    ) -> None:
        super().__init__()
        total_in = in_channels or ((in_channels_high or 0) + (in_channels_low or 0)) or 128
        out_ch = out_channels or 64

        if Conv is not None:
            self.conv = Conv(
                in_channels=total_in,
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                act=act,
                norm=norm,
            )

    def forward(self, high_feat: Any, low_feat: Optional[Any] = None) -> Any:
        if not hasattr(self, "conv"):
            return high_feat
        if low_feat is not None and torch is not None:
            if hasattr(high_feat, "shape") and hasattr(low_feat, "shape") and high_feat.shape[2:] != low_feat.shape[2:]:
                high_feat = torch.nn.functional.interpolate(high_feat, size=low_feat.shape[2:], mode="nearest")
            cat_feat = torch.cat([high_feat, low_feat], dim=1)
            return self.conv(cat_feat)
        return self.conv(high_feat)


__all__ = ["UpsampleBlock", "FeatureFusionBlock"]
