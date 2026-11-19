"""Knowledge Distillation Subsystem for Continual Learning in Aethel Vision Framework.

Provides KnowledgeDistillationLoss to distill feature representations and logit outputs
from frozen Teacher models into fine-tuning Student models, mitigating catastrophic forgetting.
"""

from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

from aethel.utils.logger import get_logger

logger = get_logger("aethel.continual.distillation")


class KnowledgeDistillationLoss:
    """Computes Knowledge Distillation loss between Teacher model and Student model outputs."""

    def __init__(
        self,
        temperature: float = 2.0,
        distill_weight: float = 1.0,
        feature_distill_weight: float = 0.5,
    ) -> None:
        """Initialize KnowledgeDistillationLoss.

        Args:
            temperature: Softmax temperature scaling parameter.
            distill_weight: Logit distillation loss weight multiplier.
            feature_distill_weight: Feature representation distillation loss weight multiplier.
        """
        self.temperature = temperature
        self.distill_weight = distill_weight
        self.feature_distill_weight = feature_distill_weight

    def __call__(
        self,
        student_outputs: Any,
        teacher_outputs: Any,
        student_features: Optional[Any] = None,
        teacher_features: Optional[Any] = None,
    ) -> Any:
        """Compute total distillation loss.

        Args:
            student_outputs: Student logits or output predictions tensor.
            teacher_outputs: Teacher logits or output predictions tensor.
            student_features: Optional intermediate feature maps from Student.
            teacher_features: Optional intermediate feature maps from Teacher.

        Returns:
            Scalar distillation loss tensor.
        """
        if torch is None or student_outputs is None or teacher_outputs is None:
            return 0.0

        # Logit distillation loss (KL-Divergence with Softmax temperature)
        s_logits = student_outputs / self.temperature
        t_logits = teacher_outputs / self.temperature

        if hasattr(F, "kl_div"):
            p_s = F.log_softmax(s_logits, dim=-1)
            p_t = F.softmax(t_logits, dim=-1)
            loss_logit = F.kl_div(p_s, p_t, reduction="batchmean") * (self.temperature ** 2)
        else:
            loss_logit = 0.0

        # Feature distillation loss (MSE between intermediate feature maps)
        loss_feat = 0.0
        if student_features is not None and teacher_features is not None and hasattr(F, "mse_loss"):
            if isinstance(student_features, (list, tuple)) and isinstance(teacher_features, (list, tuple)):
                for sf, tf in zip(student_features, teacher_features):
                    loss_feat = loss_feat + F.mse_loss(sf, tf)
            else:
                loss_feat = F.mse_loss(student_features, teacher_features)

        total_distill_loss = (self.distill_weight * loss_logit) + (self.feature_distill_weight * loss_feat)
        return total_distill_loss


__all__ = ["KnowledgeDistillationLoss"]
