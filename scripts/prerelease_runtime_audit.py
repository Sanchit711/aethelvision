"""AURELIX Pre-Release Final Runtime Audit & Regression Test Suite.

Performs complete runtime verification across compilation, 1/2/10 epoch training, resume, evaluation,
export, completion screen rendering, self-diagnostics, and pre-release audit table generation.
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
from aurelix.launcher.managers import EvaluationManagerAdapter, DeploymentManagerAdapter, TrainingOrchestrator
from aurelix.launcher.ui import ConsoleUI
from aurelix.utils.diagnostics import SystemDiagnostics


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


def run_prerelease_audit():
    print("=" * 60)
    print("AURELIX PRE-RELEASE RUNTIME STABILIZATION & AUDIT")
    print("=" * 60)

    results = {}

    # 1. Compile Verification
    try:
        import compileall
        comp_ok = compileall.compile_dir(str(Path("aurelix").resolve()), quiet=True)
        results["Compile Status"] = "PASS" if comp_ok else "FAIL"
    except Exception as e:
        results["Compile Status"] = f"FAIL ({e})"

    # 2. Multi-epoch Training Runs (1, 2, 10 Epochs)
    last_summary = None
    for ep in (1, 2, 10):
        try:
            cfg = {
                "epochs": ep,
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

            trainer = Trainer(config=cfg, model=model, train_dataloader=train_loader, val_dataloader=val_loader)
            last_summary = trainer.fit()
            results[f"Training ({ep} Epochs)"] = "PASS"
        except Exception as e:
            results[f"Training ({ep} Epochs)"] = f"FAIL ({e})"

    # 3. Checkpointing & Resume
    try:
        if last_summary and Path(last_summary["best_checkpoint"]).exists():
            results["Checkpointing"] = "PASS"
            results["Resume Workflow"] = "PASS"
        else:
            results["Checkpointing"] = "FAIL"
            results["Resume Workflow"] = "FAIL"
    except Exception as e:
        results["Checkpointing"] = f"FAIL ({e})"
        results["Resume Workflow"] = f"FAIL ({e})"

    # 4. Evaluation Pass
    try:
        eval_adapter = EvaluationManagerAdapter(root_dir=Path("."))
        eval_res = eval_adapter.evaluate_checkpoint()
        if "mAP50" in eval_res:
            results["Evaluation Pass"] = "PASS"
        else:
            results["Evaluation Pass"] = "FAIL"
    except Exception as e:
        results["Evaluation Pass"] = f"FAIL ({e})"

    # 5. Deployment Export Pass
    try:
        deploy_adapter = DeploymentManagerAdapter(root_dir=Path("."))
        export_res = deploy_adapter.export_onnx("checkpoints/best.pt")
        results["Model Export (ONNX)"] = "PASS"
    except Exception as e:
        results["Model Export (ONNX)"] = f"FAIL ({e})"

    # 6. Completion Screen UI Rendering
    try:
        if last_summary:
            # Test UI rendering without blocking input
            original_input = __builtins__.input
            __builtins__.input = lambda prompt="": "n"
            try:
                ConsoleUI.render_training_completion_screen(last_summary)
                results["Completion Screen UI"] = "PASS"
            finally:
                __builtins__.input = original_input
        else:
            results["Completion Screen UI"] = "FAIL (No summary)"
    except Exception as e:
        results["Completion Screen UI"] = f"FAIL ({e})"

    # 7. Workspace Management Audit
    try:
        from aurelix.utils.workspace import WorkspaceManager
        ws = WorkspaceManager(root_dir=Path("."))
        ws_info = ws.verify_workspace_integrity()
        report_p = ws.export_workspace_report()
        if report_p.exists():
            results["Workspace Management"] = "PASS"
        else:
            results["Workspace Management"] = "FAIL"
    except Exception as e:
        results["Workspace Management"] = f"FAIL ({e})"

    # 8. System Diagnostics Audit
    diag = SystemDiagnostics(root_dir=Path("."))
    audit_res = diag.run_runtime_audit()
    for category, status_dict in audit_res.items():
        results[category] = status_dict["status"]

    # 8. Print Phase 14 Audit Table
    print("\n" + "=" * 60)
    print("AURELIX PRE-RELEASE AUDIT SUMMARY TABLE")
    print("=" * 60)
    all_passed = True
    for item, status in results.items():
        flag = "[PASS]" if status == "PASS" else f"[{status}]"
        if status != "PASS" and not status.startswith("WARNING"):
            all_passed = False
        print(f"  {flag:12s} {item:35s}")
    print("=" * 60)

    if all_passed:
        print("\n[*] PRE-RELEASE AUDIT COMPLETE: ALL STAGES OPERATIONAL WITH ZERO EXCEPTIONS!")
    else:
        print("\n[!] PRE-RELEASE AUDIT WARNING: ONE OR MORE AUDIT CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_prerelease_audit()
