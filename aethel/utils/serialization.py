"""Safe checkpoint serialization helpers for Aethel Vision.

PyTorch checkpoints that contain optimizer or RNG objects cannot always be loaded
with ``weights_only=True``. This module prefers the safe loader, then falls back
to pickle only when explicitly allowed (local training resume or env override).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

from aethel.utils.exceptions import CheckpointException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.utils.serialization")

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def pickle_load_allowed(explicit: Optional[bool] = None) -> bool:
    """Return whether untrusted pickle load is permitted.

    Precedence:
      1. Explicit caller argument
      2. ``AETHEL_STRICT_CHECKPOINTS=1`` always denies pickle
      3. ``AETHEL_ALLOW_UNTRUSTED_CHECKPOINTS=1`` allows pickle
      4. Default deny (safe)
    """
    if explicit is not None:
        return explicit
    if _env_flag("AETHEL_STRICT_CHECKPOINTS"):
        return False
    return _env_flag("AETHEL_ALLOW_UNTRUSTED_CHECKPOINTS")


def safe_torch_load(
    path: Union[str, Path],
    map_location: Any = "cpu",
    *,
    allow_pickle: Optional[bool] = None,
) -> Any:
    """Load a ``.pt`` file, preferring ``weights_only=True``.

    Args:
        path: Checkpoint path.
        map_location: Torch map_location argument.
        allow_pickle: Override pickle fallback. ``None`` uses environment policy.

    Raises:
        CheckpointException: File missing, unreadable, or pickle denied.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise CheckpointException(
            message="PyTorch is required to load checkpoints",
            details=str(exc),
        ) from exc

    target = Path(path)
    if not target.exists():
        raise CheckpointException(message=f"Checkpoint not found: {target}")

    try:
        return torch.load(target, map_location=map_location, weights_only=True)
    except Exception as safe_err:
        if not pickle_load_allowed(allow_pickle):
            raise CheckpointException(
                message=(
                    f"Refused to unpickle checkpoint '{target}'. "
                    "Re-export weights with tensors only, or set "
                    "AETHEL_ALLOW_UNTRUSTED_CHECKPOINTS=1 for trusted local files."
                ),
                details=str(safe_err),
                recommended_fix="Prefer weights-only checkpoints. Enable pickle only for trusted training resumes.",
            ) from safe_err

        logger.warning(
            "Loading checkpoint with pickle (weights_only=False): %s. "
            "Use AETHEL_STRICT_CHECKPOINTS=1 to forbid this.",
            target,
        )
        try:
            return torch.load(target, map_location=map_location, weights_only=False)
        except Exception as pickle_err:
            raise CheckpointException(
                message=f"Failed to load checkpoint '{target}'",
                details=str(pickle_err),
            ) from pickle_err


def inspect_checkpoint_header(path: Union[str, Path]) -> dict[str, Any]:
    """Best-effort metadata extraction without inventing metrics."""
    meta: dict[str, Any] = {
        "path": str(path),
        "readable": False,
        "epoch": None,
        "architecture": {},
        "metrics": {},
        "hyperparameters": {},
        "framework_version": None,
    }
    try:
        payload = safe_torch_load(path, allow_pickle=True)
    except Exception as exc:
        meta["error"] = str(exc)
        return meta

    if not isinstance(payload, dict):
        meta["readable"] = True
        meta["payload_type"] = type(payload).__name__
        return meta

    meta["readable"] = True
    meta["epoch"] = payload.get("epoch")
    meta["architecture"] = payload.get("architecture") or {}
    meta["metrics"] = payload.get("metrics") or {}
    meta["hyperparameters"] = payload.get("hyperparameters") or {}
    meta["framework_version"] = payload.get("framework_version")
    meta["has_state_dict"] = "state_dict" in payload or "model_state_dict" in payload
    return meta


__all__ = [
    "safe_torch_load",
    "pickle_load_allowed",
    "inspect_checkpoint_header",
]
