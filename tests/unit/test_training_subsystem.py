"""Comprehensive Unit Test Suite for Aurelix Training Subsystem.

Tests loss functions, optimizer/scheduler builders, callbacks, metrics, checkpoint manager,
EMA updating, global determinism seed management, and end-to-end Trainer fitting.
"""

import pytest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

from aurelix.callbacks.manager import CallbackManager
from aurelix.callbacks.standard import EarlyStoppingCallback, ModelCheckpointCallback, ProgressLoggerCallback
from aurelix.configs.manager import ConfigManager
from aurelix.losses.composite import CompositeDetectionLoss
from aurelix.losses.dfl import DistributionFocalLoss
from aurelix.losses.focal import BCEWithLogitsLoss, FocalLoss
from aurelix.losses.iou import BboxIoULoss
from aurelix.metrics.evaluator import DetectionEvaluator, MetricAggregator
from aurelix.models.detector import AurelixDetector
from aurelix.optimizers.builder import OptimizerBuilder
from aurelix.registry.registry import LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY, TRAINER_REGISTRY
from aurelix.schedulers.builder import SchedulerBuilder
from aurelix.trainer.checkpoint import CheckpointManager, ResumeManager
from aurelix.trainer.ema import ModelEMA
from aurelix.trainer.seed import restore_rng_state, set_seed
from aurelix.trainer.trainer import Trainer


def test_registries_contains_training_modules():
    """Verify registry registration of trainer subsystem components."""
    assert LOSS_REGISTRY.contains("BboxIoULoss")
    assert LOSS_REGISTRY.contains("FocalLoss")
    assert LOSS_REGISTRY.contains("DistributionFocalLoss")
    assert LOSS_REGISTRY.contains("CompositeDetectionLoss")
    assert OPTIMIZER_REGISTRY.contains("OptimizerBuilder")
    assert SCHEDULER_REGISTRY.contains("SchedulerBuilder")
    assert TRAINER_REGISTRY.contains("Trainer")


def test_iou_loss_variants():
    """Test IoU, GIoU, DIoU, and CIoU loss calculations."""
    if torch is None:
        return

    pred_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 80.0, 80.0]])
    target_boxes = torch.tensor([[12.0, 12.0, 48.0, 48.0], [20.0, 20.0, 80.0, 80.0]])

    for loss_type in ["iou", "giou", "diou", "ciou"]:
        loss_fn = BboxIoULoss(loss_type=loss_type)
        loss_val, breakdown = loss_fn(pred_boxes, target_boxes)
        assert hasattr(loss_val, "item")
        assert loss_val.item() >= 0.0
        assert f"{loss_type}_loss" in breakdown


def test_focal_loss():
    """Test FocalLoss forward propagation."""
    if torch is None:
        return

    pred_logits = torch.randn(4, 10)
    targets = torch.zeros(4, 10)
    targets[0, 2] = 1.0

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    loss_val, breakdown = loss_fn(pred_logits, targets)
    assert loss_val.item() > 0.0
    assert "focal_loss" in breakdown


def test_composite_loss():
    """Test CompositeDetectionLoss orchestrator."""
    if torch is None:
        return

    preds = {
        "cls_logits": torch.randn(2, 5),
        "pred_boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [0.0, 0.0, 20.0, 20.0]]),
        "pred_dfl": torch.randn(2, 4, 16),
    }
    targets = {
        "cls_targets": torch.zeros(2, 5),
        "target_boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [0.0, 0.0, 20.0, 20.0]]),
        "target_dfl": torch.tensor([[5.0, 5.0, 5.0, 5.0], [2.0, 2.0, 2.0, 2.0]]),
    }

    loss_fn = CompositeDetectionLoss()
    total_loss, breakdown = loss_fn(preds, targets)
    assert total_loss.item() >= 0.0
    assert "total_loss" in breakdown
    assert "cls_loss" in breakdown
    assert "box_loss" in breakdown
    assert "dfl_loss" in breakdown


def test_optimizer_and_scheduler_builders():
    """Test OptimizerBuilder and SchedulerBuilder."""
    if torch is None:
        return

    dummy_model = nn.Linear(10, 2)
    opt_builder = OptimizerBuilder(name="AdamW")
    optimizer = opt_builder.build(dummy_model.parameters(), lr=0.01)

    sched_builder = SchedulerBuilder(name="CosineAnnealingLR")
    scheduler = sched_builder.build(optimizer, epochs=10)

    assert optimizer is not None
    assert scheduler is not None


def test_checkpoint_save_and_load(tmp_path):
    """Test CheckpointManager saving and loading payloads with RNG state."""
    if torch is None:
        return

    set_seed(42)
    model = nn.Linear(5, 2)
    opt_builder = OptimizerBuilder(name="Adam")
    optimizer = opt_builder.build(model.parameters(), lr=0.01)

    manager = CheckpointManager(output_dir=tmp_path)
    saved_file = manager.save_checkpoint(model=model, optimizer=optimizer, epoch=5, is_best=True)

    assert saved_file.exists()
    assert (tmp_path / "best.pt").exists()

    model_new = nn.Linear(5, 2)
    opt_new = opt_builder.build(model_new.parameters(), lr=0.01)
    payload = manager.load_checkpoint(saved_file, model=model_new, optimizer=opt_new)

    assert payload["epoch"] == 5


def test_model_ema():
    """Test ModelEMA weight update step."""
    if torch is None:
        return

    model = nn.Linear(5, 2)
    ema = ModelEMA(model, decay=0.99)
    assert ema.ema is not None

    # Mutate model weights
    with torch.no_grad():
        model.weight.add_(1.0)

    ema.update(model)


def test_metrics_evaluator():
    """Test DetectionEvaluator and MetricAggregator."""
    evaluator = DetectionEvaluator(num_classes=5)
    metrics = evaluator.compute()

    assert "mAP@50" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics


class DummyDataset:
    """Mock PyTorch Dataset returning image tensors and targets."""

    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        if torch is None:
            return None, None
        img = torch.randn(3, 320, 320)
        target = {
            "cls_targets": torch.zeros(1, 80),
            "target_boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "target_dfl": torch.tensor([[5.0, 5.0, 5.0, 5.0]]),
        }
        return img, target


def dummy_collate(batch):
    if torch is None:
        return None, None
    imgs = torch.stack([b[0] for b in batch])
    targets = {
        "cls_targets": torch.cat([b[1]["cls_targets"] for b in batch]),
        "target_boxes": torch.cat([b[1]["target_boxes"] for b in batch]),
        "target_dfl": torch.cat([b[1]["target_dfl"] for b in batch]),
    }
    return imgs, targets


def test_trainer_end_to_end(tmp_path):
    """Test end-to-end Trainer fit lifecycle using AurelixDetector."""
    if torch is None or DataLoader is None:
        return

    model = AurelixDetector(num_classes=80, scale="tiny")
    dataset = DummyDataset(size=2)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dummy_collate)

    config = {
        "epochs": 2,
        "device": "cpu",
        "amp": False,
        "accumulate_steps": 1,
        "output_dir": str(tmp_path),
        "seed": 42,
        "num_classes": 80,
        "optimizer": {"name": "AdamW", "lr": 0.001},
        "scheduler": {"name": "CosineAnnealingLR"},
    }

    trainer = Trainer(config=config, model=model, train_dataloader=dataloader, val_dataloader=dataloader)
    summary = trainer.fit()

    assert summary is not None
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metrics.csv").exists()
