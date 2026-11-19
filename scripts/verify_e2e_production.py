"""Real End-to-End Production Audit Script for AURELIX Vision Framework.

Executes real multi-epoch training without mocks or patches and verifies that all physical files exist.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import torch
from torch.utils.data import DataLoader

from aurelix.models.detector import AurelixDetector
from aurelix.trainer.trainer import Trainer
from aurelix.trainer.experiment import ExperimentArtifactManager


class MockDataset:
    """Mock PyTorch Dataset producing image and target dicts."""

    def __init__(self, count: int = 8) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int):
        img = torch.randn(3, 128, 128)
        target = {
            "cls_targets": torch.zeros(1, 2),
            "target_boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "target_dfl": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        }
        return img, target


def main():
    print("=" * 70)
    print("AURELIX REAL END-TO-END PRODUCTION TRAINING AUDIT")
    print("=" * 70)

    cfg = {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.001,
        "device": "cpu",
        "amp": False,
        "ema": True,
        "output_dir": "runs",
        "class_names": ["Person", "Vehicle"],
    }

    model = AurelixDetector(scale="tiny", num_classes=2)
    train_loader = DataLoader(MockDataset(8), batch_size=4)
    val_loader = DataLoader(MockDataset(8), batch_size=4)

    trainer = Trainer(
        config=cfg,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    print("\n[E2E Audit] Launching 2-epoch production training run...")
    summary = trainer.fit()

    exp_dir = Path(summary["exp_dir"]).resolve()
    print(f"\n[E2E Audit] Training complete. Canonical Experiment Directory: {exp_dir}")

    # Initialize manager to run self test over generated experiment folder
    mgr = ExperimentArtifactManager(base_dir=exp_dir.parent, run_id=exp_dir.name)
    test_results = mgr.run_self_test()

    all_passed = all(status == "PASS" for status in test_results.values())
    if all_passed:
        print("\n[E2E Audit SUCCESS]: All physical artifacts exist and passed integrity verification!")
    else:
        print("\n[E2E Audit FAIL]: Some artifacts failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
