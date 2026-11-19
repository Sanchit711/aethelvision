"""AETHEL Enterprise AI Vision Platform — Demo Mode & Tutorial Project Generator.

Provides safe simulation Demo Mode (zero filesystem mutations, simulated training/eval/export)
and Tutorial Project Generator for instant onboarding without bringing external data.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.box import ROUNDED, HEAVY


class DemoModeManager:
    """Safe Demonstration Mode Manager (Simulated Training, Benchmark, & Export)."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self.is_demo_active = False

    def toggle_demo_mode(self) -> bool:
        """Toggle active Demo Mode state."""
        self.is_demo_active = not self.is_demo_active
        return self.is_demo_active

    def run_simulated_training(self) -> None:
        """Run simulated training demonstration without file mutations."""
        self.console.clear()
        
        banner = Panel(
            Align.center(Text("AETHEL SAFE DEMO MODE — SIMULATED MODEL TRAINING", style="bold bright_yellow")),
            box=HEAVY,
            border_style="yellow",
        )
        self.console.print(banner)
        print("\n[DEMO MODE ACTIVE] Executing zero-side-effect training demonstration...\n")

        for epoch in range(1, 11):
            loss = max(0.01, 0.50 - (epoch * 0.045))
            val_loss = max(0.015, 0.52 - (epoch * 0.042))
            mAP50 = min(0.985, 0.50 + (epoch * 0.048))
            
            bar = "█" * (epoch * 3) + "░" * (30 - (epoch * 3))
            print(f"  [DEMO Epoch {epoch:2d}/10] [{bar}] Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | mAP50: {mAP50:.4f}")
            time.sleep(0.3)

        print("\n✔ [DEMO COMPLETE] Simulated model training finished cleanly with mAP50=98.5%!")
        input("\n  [Demo Mode] Press Enter to return...")

    def run_simulated_benchmark(self) -> None:
        """Run simulated benchmark demonstration."""
        self.console.clear()
        
        table = Table(title="[bold yellow]DEMO MODE — SIMULATED HARDWARE BENCHMARK[/bold yellow]", box=ROUNDED, border_style="yellow", expand=True)
        table.add_column("Engine / Backend", style="bold cyan")
        table.add_column("Precision", style="bold white")
        table.add_column("Latency (ms)", style="bold red")
        table.add_column("Throughput (FPS)", style="bold green")

        table.add_row("PyTorch Native Engine", "FP32", "7.02 ms", "142.5 FPS")
        table.add_row("ONNX Runtime Engine", "FP32", "3.12 ms", "320.5 FPS")
        table.add_row("NVIDIA TensorRT Engine", "FP16", "1.45 ms", "689.6 FPS")

        self.console.print(table)
        input("\n  [Demo Mode] Press Enter to return...")


class TutorialProjectGenerator:
    """Tutorial Project & Sample Dataset Generator for New User Onboarding."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or Path(".").resolve()
        self.tutorial_dir = self.root_dir / "projects" / "tutorial_project"

    def generate_tutorial_project(self) -> Dict[str, Any]:
        """Generate sample tutorial project with dataset, checkpoint, and reports."""
        self.tutorial_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["datasets", "models", "experiments", "reports", "exports", "plugins", "configs", "logs"]:
            (self.tutorial_dir / sub).mkdir(parents=True, exist_ok=True)

        # 1. Project Manifest
        manifest = {
            "name": "AETHEL Tutorial Project",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_opened": time.strftime("%Y-%m-%d %H:%M:%S"),
            "favorite": True,
            "version": "1.0.0",
            "is_tutorial": True,
            "description": "Interactive sample tutorial project for new user onboarding.",
        }
        with open(self.tutorial_dir / "project.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 2. Sample Dataset YAML
        ds_dir = self.tutorial_dir / "datasets" / "sample_person_dataset"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (ds_dir / "val" / "images").mkdir(parents=True, exist_ok=True)

        data_yaml = {
            "path": str(ds_dir),
            "train": "train/images",
            "val": "val/images",
            "names": {0: "person", 1: "vehicle"},
        }
        try:
            import yaml
            with open(ds_dir / "data.yaml", "w", encoding="utf-8") as f:
                yaml.dump(data_yaml, f)
        except Exception:
            pass

        # 3. Sample Model Meta
        model_meta = {
            "name": "AethelNet-SMALL-Tutorial.pt",
            "architecture": "CSPDarknet-Small",
            "params": "4,913,826",
            "mAP50": "98.2%",
            "status": "READY_FOR_INFERENCE",
        }
        with open(self.tutorial_dir / "models" / "tutorial_model_meta.json", "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2)

        return {
            "status": "SUCCESS",
            "project_name": "AETHEL Tutorial Project",
            "project_path": str(self.tutorial_dir),
        }


__all__ = ["DemoModeManager", "TutorialProjectGenerator"]
