"""AETHEL Enterprise AI Vision Platform — Onboarding & Guided Product Tour.

Provides first-run detection, tour state persistence, and an interactive 18-step
guided product tour explaining every framework subsystem and workflow.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.align import Align
from rich.box import DOUBLE, HEAVY, ROUNDED, SQUARE
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


class OnboardingManager:
    """Enterprise Interactive Guided Product Tour & Onboarding Manager."""

    def __init__(self, root_dir: Optional[Path] = None, console: Optional[Console] = None) -> None:
        self.root_dir = root_dir or Path(".").resolve()
        self.config_path = self.root_dir / "configs" / "onboarding.json"
        self.console = console or Console()
        self._tour_steps = self._build_tour_steps()

    def is_first_run(self) -> bool:
        """Check if onboarding has been completed or skipped previously."""
        if not self.config_path.exists():
            return True
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return not data.get("onboarding_completed", False)
        except Exception:
            return True

    def mark_completed(self, completed: bool = True) -> None:
        """Persist onboarding completion status."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "onboarding_completed": completed,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0.0",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def reset_onboarding(self) -> None:
        """Reset onboarding state to allow re-running the tour."""
        if self.config_path.exists():
            try:
                os.remove(self.config_path)
            except Exception:
                pass

    def prompt_first_run(self) -> str:
        """Prompt first-time user with Welcome dialog."""
        self.console.clear()
        
        banner_text = Text.assemble(
            ("AETHEL\n", "bold bright_cyan"),
            ("Enterprise AI Vision Platform\n", "bold white"),
            ("Advanced Computer Vision Framework v1.0.0\n\n", "italic dim white"),
            ("WELCOME TO AETHEL ENTERPRISE PLATFORM!\n", "bold bright_green"),
            ("Would you like to take a guided interactive product tour?", "bold white")
        )

        panel = Panel(
            Align.center(banner_text),
            box=ROUNDED,
            border_style="bright_blue",
            padding=(1, 4),
            title="[bold bright_white] FIRST-RUN ONBOARDING [/bold bright_white]",
        )
        self.console.print(panel)

        menu_t = Table(show_header=False, box=None, padding=(0, 2))
        menu_t.add_column("Key", style="bold yellow")
        menu_t.add_column("Label", style="bold white")
        menu_t.add_row("[1]", "Start Interactive Guided Tour (Recommended)")
        menu_t.add_row("[2]", "Skip Tour & Launch Dashboard")
        menu_t.add_row("[3]", "Open Documentation Suite")
        self.console.print(Align.center(menu_t))
        self.console.print()

        try:
            choice = input("  AETHEL Onboarding > Select option [1-3] (default 1): ").strip()
            return choice or "1"
        except (KeyboardInterrupt, EOFError):
            return "2"

    def _build_tour_steps(self) -> List[Dict[str, Any]]:
        """Construct the 18-step enterprise guided product tour pages."""
        return [
            {
                "step": 1,
                "title": "Welcome to AETHEL Enterprise AI Platform",
                "purpose": "Platform Introduction & Architectural Philosophy",
                "description": "AETHEL is a commercial-grade, indigenous PyTorch computer vision framework built for enterprise AI development, model training, and edge deployment.",
                "workflow": "Single entrypoint launcher -> First-class projects -> Automated training fit loop -> ONNX/TensorRT export -> Edge deployment.",
                "notes": "100% standalone indigenous PyTorch implementation without third-party framework dependencies.",
                "shortcuts": "Press [N] Next | [P] Previous | [J] Jump | [X] Skip | [0] Exit Tour",
            },
            {
                "step": 2,
                "title": "Platform Launcher Dashboard",
                "purpose": "Real-time System Overview & Health Monitoring",
                "description": "The central operational dashboard aggregates System Status, Build Metadata, Hardware Resources (CPU/GPU/VRAM), Active Project State, Training Status, and AI Metrics.",
                "workflow": "Inspect hardware badges -> Verify CUDA / TensorRT acceleration -> Check active project & dataset status.",
                "notes": "Updates metrics automatically on startup and during training runs.",
                "shortcuts": "Press [F1] for Context Help | [:p] for Command Palette",
            },
            {
                "step": 3,
                "title": "Project Manager Workspace",
                "purpose": "First-Class Enterprise Workspace Lifecycle",
                "description": "Projects are isolated self-contained environments managing datasets, models, experiments, audit reports, exports, plugins, and configurations.",
                "workflow": "Create Project -> Open Project -> Clone -> Archive -> Switch Active Workspace.",
                "notes": "Each project maintains a manifest (`project.json`) and standard subdirectories (`datasets/`, `models/`, `reports/`).",
                "shortcuts": "Menu [12] -> Project Manager",
            },
            {
                "step": 4,
                "title": "Dataset Center & Validation Engine",
                "purpose": "Data Ingestion, Verification, and Health Analytics",
                "description": "Manages computer vision datasets (YOLO, COCO format). Performs boundary validation, duplicate detection, class distribution analysis, and dataset health scoring.",
                "workflow": "Import Dataset -> Validate Structure -> Check Health Score -> Pass to Training Orchestrator.",
                "notes": "Dataset Validator ensures zero out-of-bounds bounding boxes or corrupted image files before training starts.",
                "shortcuts": "Menu [4] Dataset Manager | Menu [6] Dataset Validation",
            },
            {
                "step": 5,
                "title": "Deep Learning Training Center",
                "purpose": "End-to-End Model Training, Resuming, & Fine-Tuning",
                "description": "Orchestrates object detection training with automatic mixed precision (AMP fp16), EMA weights, Cosine Annealing learning rate schedules, and live Rich terminal progress bars.",
                "workflow": "Select Dataset -> Choose Model Preset (AethelNet-SMALL) -> Set Hyperparameters -> Launch Fit Loop -> Auto Save Checkpoints.",
                "notes": "Crash Recovery automatically saves session metadata (`crash_recovery_metadata.json`) allowing 1-click continuation after power loss.",
                "shortcuts": "Menu [1] Training Center | Menu [2] Resume Training | Menu [3] Fine Tune",
            },
            {
                "step": 6,
                "title": "Model Registry & Version History",
                "purpose": "Model Checkpoint Versioning & Promotion Lifecycle",
                "description": "Discovers, tracks, and manages candidate and master model checkpoints. Tracks parameter counts, file sizes, mAP@0.50 accuracy scores, and creation dates.",
                "workflow": "Discover Checkpoints -> Compare Performance -> Promote Candidate Model to Master Release -> Export to Production.",
                "notes": "Checkpoints store PyTorch `state_dict`, optimizer, scheduler, EMA, and RNG states for exact reproducibility.",
                "shortcuts": "Menu [7] Model Registry",
            },
            {
                "step": 7,
                "title": "Evaluation Center & Metric Analytics",
                "purpose": "Validation, Precision, Recall, & Loss Quantification",
                "description": "Computes object detection validation metrics: Precision, Recall, F1 Score, mAP@0.50, and mAP@0.50:0.95 across validation splits.",
                "workflow": "Select Model Checkpoint -> Select Target Validation Split -> Compute Metrics -> Inspect Confusion Matrix & PR Curves.",
                "notes": "Uses native Decoupled Head evaluation logic matching COCO metric standards.",
                "shortcuts": "Menu [8] Evaluation Center",
            },
            {
                "step": 8,
                "title": "Benchmark Center & Performance Profiling",
                "purpose": "Latency, FPS, Memory, & Throughput Profiling",
                "description": "Profiles model inference speed, per-image latency (ms), throughput (FPS), peak RAM usage, and VRAM footprint across CPU and GPU hardware backends.",
                "workflow": "Select Execution Device -> Define Batch Size & Warmup Iterations -> Run Benchmark -> Generate Performance Report.",
                "notes": "Compares raw PyTorch runtime speed against ONNX Runtime and TensorRT FP16 acceleration.",
                "shortcuts": "Menu [9] Benchmark Center",
            },
            {
                "step": 9,
                "title": "Deployment & Model Exporter",
                "purpose": "Cross-Platform Exporting & Edge AI Packaging",
                "description": "Exports PyTorch checkpoints into high-performance production formats: ONNX, TensorRT engines, TorchScript, OpenVINO, NCNN, and RKNN.",
                "workflow": "Select Checkpoint -> Choose Export Target (ONNX / TensorRT) -> Generate Compiled Engine -> Verify Export Output.",
                "notes": "ONNXExporter generates standalone `.onnx` files with dynamic batch size support.",
                "shortcuts": "Menu [10] Export Center | Menu [11] Deployment Center",
            },
            {
                "step": 10,
                "title": "Audit & Report Center",
                "purpose": "Automated Multi-Format Document Generation",
                "description": "Automatically generates comprehensive Markdown & HTML audit reports covering Training Runs, Evaluation Results, Benchmarks, Environment Profile, Hardware, and License status.",
                "workflow": "Run Experiment -> Select Report Center -> Generate Master Release Summary -> Export to Markdown / HTML.",
                "notes": "Reports are formatted for executive compliance and technical documentation.",
                "shortcuts": "Menu [13] Reports Center",
            },
            {
                "step": 11,
                "title": "Plugin Architecture & Extensions",
                "purpose": "Modular Feature Extensions across Vision Domains",
                "description": "Supports dynamic discovery and activation of plugins: Object Detection, ByteTrack Tracking, Paddle OCR, SAM Segmentation, Pose Keypoints, Classifiers, and Custom Extensions.",
                "workflow": "Place plugin folder in `plugins/` -> Discover Plugins -> Enable / Disable -> Execute in Pipeline.",
                "notes": "Plugins follow standard manifest specs (`plugin.json`) and extend core framework capabilities.",
                "shortcuts": "Menu [14] Diagnostics & Security | Plugin Registry",
            },
            {
                "step": 12,
                "title": "Offline Challenge-Response Licensing",
                "purpose": "Zero-Trust Cryptographic Machine Binding & Activation",
                "description": "Provides offline challenge-response security binding licenses to hardware IDs. Supports Trial, Developer, and Enterprise commercial tiers.",
                "workflow": "Generate Request Code -> Submit to Enterprise Admin -> Receive Signed Key -> Verify Signature -> Unlock Platform.",
                "notes": "No internet connection required. Key validation uses HMAC-SHA256 signature verification.",
                "shortcuts": "Menu [16] License Manager",
            },
            {
                "step": 13,
                "title": "Platform Settings & Configuration",
                "purpose": "Global & Project Environment Customization",
                "description": "Manages platform defaults: hardware device selection, CUDA AMP mixed precision, batch size, default directories, logging levels, and console UI themes.",
                "workflow": "Inspect Settings -> Modify Config -> Update `configs/train.yaml` -> Apply Platform Defaults.",
                "notes": "Configurations persist across application restarts.",
                "shortcuts": "Menu [15] Settings",
            },
            {
                "step": 14,
                "title": "Internal AI Vision Assistant",
                "purpose": "Intelligent Diagnostic Advisor & Hyperparameter Tuning",
                "description": "Provides error root-cause explanation, dataset health scoring, optimal batch size / image resolution advice, and augmentation recommendations.",
                "workflow": "Encounter Error or Prepare Training -> Open AI Assistant -> Receive Customized Advice -> Apply Recommended Fixes.",
                "notes": "Evaluates GPU VRAM and CPU capacity to recommend optimal training parameters.",
                "shortcuts": "Menu [17] AI Vision Assistant",
            },
            {
                "step": 15,
                "title": "Knowledge Graph & Code Intelligence",
                "purpose": "Graphify Codebase Mapping & Architectural Querying",
                "description": "Integrates `graphify` knowledge graph mapping code relationships, AST nodes, and architectural dependencies across 6,000+ code nodes.",
                "workflow": "Run `graphify query <question>` -> Inspect subgraphs -> Understand module interrelationships.",
                "notes": "Keeps codebase architecture transparent and maintainable.",
                "shortcuts": "CLI: `graphify query <concept>`",
            },
            {
                "step": 16,
                "title": "Developer Mode & SDK Architecture",
                "purpose": "API Integration & Custom Model Development",
                "description": "Allows developers to import AETHEL Python APIs directly into custom Python scripts or Jupyter Notebooks.",
                "workflow": "`from aethel.models import build_aethelnet_small` -> Initialize model -> Pass PyTorch tensor -> Execute forward pass.",
                "notes": "Complete API Reference and SDK documentation available in `docs/`.",
                "shortcuts": "Refer to `docs/Developer_Guide.md` & `docs/SDK_Guide.md`",
            },
            {
                "step": 17,
                "title": "Enterprise Best Practices",
                "purpose": "Operational Guidelines for Production Deployments",
                "description": "Provides industry best practices for project structure, dataset annotation splits, checkpoint retention, metric evaluation, and TensorRT optimization.",
                "workflow": "Organize Projects -> Validate Datasets -> Train with Cosine Schedules -> Evaluate mAP -> Export ONNX -> Deploy.",
                "notes": "Following best practices guarantees maximum detection precision and real-time inference speed.",
                "shortcuts": "Refer to `docs/User_Manual.md`",
            },
            {
                "step": 18,
                "title": "Onboarding Complete — Ready to Launch!",
                "purpose": "Product Tour Completion & Quick Actions",
                "description": "You have completed the AETHEL Enterprise AI Platform guided tour. You are now equipped to navigate, train, evaluate, and deploy enterprise computer vision models.",
                "workflow": "Choose a Quick Action below to start working with AETHEL Enterprise Platform.",
                "notes": "You can re-launch this guided tour anytime from the Main Console menu.",
                "shortcuts": "[1] Create Project | [2] Import Dataset | [3] Open Docs | [4] Training Center | [0] Main Console",
            },
        ]

    def run_tour(self) -> None:
        """Run interactive 18-step guided product tour."""
        current_step = 0
        total_steps = len(self._tour_steps)

        while True:
            step_data = self._tour_steps[current_step]
            self.console.clear()
            
            # Header Panel
            header_text = Text.assemble(
                ("AETHEL ENTERPRISE PLATFORM — GUIDED PRODUCT TOUR\n", "bold bright_cyan"),
                (f"Step {step_data['step']} of {total_steps}: ", "bold yellow"),
                (f"{step_data['title']}", "bold white")
            )
            self.console.print(Panel(Align.center(header_text), box=ROUNDED, border_style="cyan"))

            # Step Content Table
            content_table = Table(show_header=False, box=ROUNDED, border_style="bright_blue", expand=True)
            content_table.add_column("Field", style="bold cyan", width=18)
            content_table.add_column("Content", style="white")

            content_table.add_row("Module Purpose", f"[bold yellow]{step_data['purpose']}[/bold yellow]")
            content_table.add_row("Description", step_data['description'])
            content_table.add_row("Typical Workflow", f"[bold green]{step_data['workflow']}[/bold green]")
            content_table.add_row("Important Notes", f"[italic dim white]{step_data['notes']}[/italic dim white]")
            content_table.add_row("Shortcuts & Access", f"[bold magenta]{step_data['shortcuts']}[/bold magenta]")

            self.console.print(content_table)

            # Progress Bar
            pct = int((step_data['step'] / total_steps) * 100)
            bar = f"[{'█' * int(30 * step_data['step'] / total_steps)}{'░' * (30 - int(30 * step_data['step'] / total_steps))}] {pct}%"
            self.console.print(Align.center(Text(f"Progress: {bar}\n", style="bold yellow")))

            # Controls Prompt
            self.console.print("  Controls: [N] Next | [P] Previous | [J] Jump to Step | [S] Save & Exit | [0] Main Console")
            try:
                cmd = input(f"  Tour (Step {step_data['step']}/{total_steps}) > ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                break

            if cmd in ("n", "", "next"):
                if current_step < total_steps - 1:
                    current_step += 1
                else:
                    self.mark_completed(True)
                    print("\n[AETHEL] Tour completed! Returning to Main Console...")
                    time.sleep(1)
                    break
            elif cmd in ("p", "prev", "previous"):
                if current_step > 0:
                    current_step -= 1
            elif cmd in ("j", "jump"):
                try:
                    target = input(f"  Enter step number [1-{total_steps}]: ").strip()
                    if target.isdigit() and 1 <= int(target) <= total_steps:
                        current_step = int(target) - 1
                except Exception:
                    pass
            elif cmd in ("s", "0", "exit", "skip", "x"):
                self.mark_completed(True)
                print("\n[AETHEL] Tour paused/saved. Returning to Main Console...")
                time.sleep(0.5)
                break


__all__ = ["OnboardingManager"]
