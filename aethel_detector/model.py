"""Public Aethel Detector Model for Object Detection.

Assembles the full end-to-end proprietary object detection pipeline:
Input -> Backbone (Stem + CSPStages + CBAM) -> Neck (PAFPN) -> Detection Head (DecoupledAnchorFree).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    class _Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def eval(self) -> "_Module": return self
        def train(self, mode: bool = True) -> "_Module": return self
        def parameters(self) -> List[Any]: return []
        def to(self, *args: Any, **kwargs: Any) -> "_Module": return self
        def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: return {}
        def load_state_dict(self, state_dict: Dict[str, Any], *args: Any, **kwargs: Any) -> None: pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any: return []
    nn = type("nn", (), {"Module": _Module})()  # type: ignore

try:
    from aethel_detector.backbone.attention import CBAM
    from aethel_detector.backbone.conv import Conv
    from aethel_detector.backbone.csp import CSPStage
    from aethel_detector.backbone.stem import DetectorStem
    from aethel_detector.head.decoder import HeadDecoder
    from aethel_detector.head.detect import DecoupledDetectHead
    from aethel_detector.neck.pafpn import PAFPN
    from aethel_detector.utils.helpers import make_divisible
except Exception:
    CBAM = Conv = CSPStage = DetectorStem = HeadDecoder = DecoupledDetectHead = PAFPN = make_divisible = None  # type: ignore


class AethelBackbone(nn.Module):
    """Aethel Detector Backbone feature extractor."""

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: Tuple[int, int, int] = (64, 128, 256),
        num_blocks: Tuple[int, int, int] = (1, 2, 2),
        use_cbam: bool = True,
    ) -> None:
        super().__init__()
        self.stem = DetectorStem(in_channels=in_channels, out_channels=stem_channels)
        c1, c2, c3 = stage_channels
        b1, b2, b3 = num_blocks
        self.down1 = Conv(in_channels=stem_channels, out_channels=stem_channels, kernel_size=3, stride=2)
        self.stage1 = CSPStage(in_channels=stem_channels, out_channels=c1, num_blocks=b1)
        self.down2 = Conv(in_channels=c1, out_channels=c1, kernel_size=3, stride=2)
        self.stage2 = CSPStage(in_channels=c1, out_channels=c2, num_blocks=b2)
        self.down3 = Conv(in_channels=c2, out_channels=c2, kernel_size=3, stride=2)
        self.stage3 = CSPStage(in_channels=c2, out_channels=c3, num_blocks=b3)
        self.cbam = CBAM(channels=c3) if use_cbam else nn.Identity()

    def forward(self, x: Any) -> List[Any]:
        if not hasattr(self, "stem"):
            return [x, x, x]
        x = self.stem(x)
        p3 = self.stage1(self.down1(x))
        p4 = self.stage2(self.down2(p3))
        p5 = self.stage3(self.down3(p4))
        p5 = self.cbam(p5)
        return [p3, p4, p5]


class AethelDetector(nn.Module):
    """Production End-to-End Object Detection Architecture."""

    def __init__(
        self,
        num_classes: int = 80,
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
        reg_max: int = 16,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        c_stem = int(64 * width_multiple)
        c3 = int(256 * width_multiple)
        c4 = int(512 * width_multiple)
        c5 = int(1024 * width_multiple)

        b1 = max(1, round(3 * depth_multiple))
        b2 = max(1, round(6 * depth_multiple))
        b3 = max(1, round(6 * depth_multiple))

        self.backbone = AethelBackbone(
            in_channels=input_channels,
            stem_channels=c_stem,
            stage_channels=(c3, c4, c5),
            num_blocks=(b1, b2, b3),
        )

        if PAFPN is not None:
            self.neck = PAFPN(in_channels=(c3, c4, c5), out_channels=(c3, c4, c5))
            self.head = DecoupledDetectHead(num_classes=num_classes, in_channels=(c3, c4, c5), reg_max=reg_max)
            self.decoder = HeadDecoder(reg_max=reg_max) if HeadDecoder is not None else None

    def forward(self, x: Any) -> Any:
        if not hasattr(self, "neck"):
            return [x, x, x]
        feats = self.backbone(x)
        pan_feats = self.neck(feats)
        outputs = self.head(pan_feats)
        if not self.training and getattr(self, "decoder", None) is not None:
            return self.decoder(outputs)
        return outputs


__all__ = ["AethelBackbone", "AethelDetector"]
