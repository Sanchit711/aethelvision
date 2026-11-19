"""Aethel Neck Submodule."""

from aethel.models.neck.pafpn import PAFPN
from aethel.registry.registry import NECK_REGISTRY

NECK_REGISTRY.register("PAFPN")(PAFPN)

__all__ = ["PAFPN"]
