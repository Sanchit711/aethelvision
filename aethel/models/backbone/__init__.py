"""Aethel Backbone Submodule."""

from aethel.models.backbone.csp import CSPBackbone, make_divisible
from aethel.models.backbone.stem import DetectorStem
from aethel.registry.registry import BACKBONE_REGISTRY

BACKBONE_REGISTRY.register("CSPBackbone")(CSPBackbone)
BACKBONE_REGISTRY.register("DetectorStem")(DetectorStem)

__all__ = ["CSPBackbone", "DetectorStem", "make_divisible"]
