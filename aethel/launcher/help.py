"""AETHEL Enterprise AI Vision Platform — Help & Command Palette System.

Provides Contextual Help (F1 / ?), Searchable Documentation & Workflows,
and Command Palette (:p / Ctrl+Shift+P) feature search.
"""

from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.box import ROUNDED, HEAVY


class HelpSystem:
    """Enterprise Contextual Help & Command Palette Engine."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self._command_index = self._build_command_index()

    def show_context_help(self, center_name: str = "Dashboard") -> None:
        """Display context-sensitive help for active screen."""
        self.console.clear()
        
        help_topics = {
            "Dashboard": {
                "title": "CONTEXT HELP — MAIN DASHBOARD",
                "description": "Displays live system health, hardware GPU memory, platform status, project state, and AI metrics.",
                "shortcuts": "[1-18] Select Menu Option | [F1 / ?] Context Help | [:p] Command Palette | [0] Exit",
                "workflow": "Review status cards -> Select action from 18 enterprise centers.",
            },
            "Training Center": {
                "title": "CONTEXT HELP — TRAINING CENTER",
                "description": "Orchestrates object detection fit loop with AMP fp16 precision, EMA weights, Cosine Annealing, and checkpointing.",
                "shortcuts": "[1] Train New Model | [2] Resume Training | [3] Fine Tune | [4] Interrupted Recovery",
                "workflow": "Select dataset -> Choose architecture -> Configure epochs/LR -> Start fit loop.",
            },
            "Dataset Center": {
                "title": "CONTEXT HELP — DATASET CENTER",
                "description": "Validates dataset directory structure, label bounding box coordinates, and train/val split ratios.",
                "shortcuts": "[1] Validate Dataset | [2] Summary Report | [3] AI Health Analysis",
                "workflow": "Select dataset -> Run validation -> Inspect class distribution.",
            },
            "Model Registry": {
                "title": "CONTEXT HELP — MODEL REGISTRY",
                "description": "Tracks candidate and master PyTorch model checkpoints, parameter counts, file sizes, and mAP metrics.",
                "shortcuts": "Inspect registered models -> Compare accuracy -> Export to ONNX.",
                "workflow": "Locate checkpoint in `checkpoints/` -> Export / Evaluate.",
            },
            "Evaluation Center": {
                "title": "CONTEXT HELP — EVALUATION CENTER",
                "description": "Calculates precision, recall, mAP@0.50, mAP@0.50:0.95, and F1 score on validation splits.",
                "shortcuts": "Select checkpoint -> Execute evaluation pass -> Inspect metrics table.",
                "workflow": "Choose checkpoint -> Run DetectionEvaluator -> View metrics summary.",
            },
            "Benchmark Center": {
                "title": "CONTEXT HELP — BENCHMARK CENTER",
                "description": "Profiles per-image latency (ms), throughput (FPS), RAM usage, and VRAM memory footprint.",
                "shortcuts": "Define batch size & iterations -> Run benchmark -> Inspect FPS.",
                "workflow": "Choose batch size -> Run inference loop -> View avg latency.",
            },
            "Deployment Center": {
                "title": "CONTEXT HELP — DEPLOYMENT CENTER",
                "description": "Exports models into production deployment targets: ONNX, TensorRT FP16 engines, and TorchScript.",
                "shortcuts": "[1] Export ONNX | [2] Export TensorRT | [3] Export TorchScript",
                "workflow": "Select target format -> Export binary -> Deploy to edge device / Docker.",
            },
            "License Manager": {
                "title": "CONTEXT HELP — LICENSE MANAGER",
                "description": "Manages offline challenge-response security binding licenses to hardware IDs.",
                "shortcuts": "[1] Generate Request Code | [2] Input Activation Key",
                "workflow": "Generate Request Code -> Submit to Admin -> Verify Signature.",
            },
        }

        topic = help_topics.get(center_name, help_topics["Dashboard"])
        
        table = Table(show_header=False, box=ROUNDED, border_style="cyan", expand=True)
        table.add_column("Key", style="bold cyan", width=18)
        table.add_column("Val", style="white")
        table.add_row("Overview", topic["description"])
        table.add_row("Key Shortcuts", f"[bold yellow]{topic['shortcuts']}[/bold yellow]")
        table.add_row("Recommended Flow", f"[bold green]{topic['workflow']}[/bold green]")

        self.console.print(Panel(table, title=f"[bold bright_white] {topic['title']} [/bold bright_white]", box=ROUNDED, border_style="bright_blue"))
        input("\n  [AETHEL Help] Press Enter to return...")

    def _build_command_index(self) -> List[Dict[str, str]]:
        """Index all platform commands for search and command palette."""
        return [
            {"cmd": "Train New Model", "center": "Training Center", "option": "1", "keywords": "train model fit hyperparameters epoch"},
            {"cmd": "Resume Training", "center": "Training Center", "option": "2", "keywords": "resume checkpoint continue interrupted"},
            {"cmd": "Fine Tune Model", "center": "Training Center", "option": "3", "keywords": "fine tune transfer learning transfer"},
            {"cmd": "Dataset Manager", "center": "Dataset Center", "option": "4", "keywords": "dataset data import coco yolo images labels"},
            {"cmd": "Annotation Manager", "center": "Dataset Center", "option": "5", "keywords": "annotation labels classes convert format"},
            {"cmd": "Dataset Validation", "center": "Dataset Center", "option": "6", "keywords": "validate check health split balance"},
            {"cmd": "Model Registry", "center": "Model Registry", "option": "7", "keywords": "model registry checkpoint weights history master"},
            {"cmd": "Evaluation Center", "center": "Evaluation Center", "option": "8", "keywords": "eval evaluation metric precision recall map f1"},
            {"cmd": "Benchmark Center", "center": "Benchmark Center", "option": "9", "keywords": "benchmark latency fps throughput memory latency speed"},
            {"cmd": "Export Models", "center": "Export Center", "option": "10", "keywords": "export onnx tensorrt torchscript openvino ncnn rknn"},
            {"cmd": "Deployment Center", "center": "Deployment Center", "option": "11", "keywords": "deploy deployment docker jetson edge package"},
            {"cmd": "Project Manager", "center": "Project Manager", "option": "12", "keywords": "project workspace create open clone archive workspace"},
            {"cmd": "Reports Center", "center": "Reports Center", "option": "13", "keywords": "report markdown html audit summary export"},
            {"cmd": "Security & Diagnostics", "center": "Diagnostics", "option": "14", "keywords": "security hash integrity tamper cache cleanup"},
            {"cmd": "Platform Settings", "center": "Settings", "option": "15", "keywords": "settings config hardware theme cuda mixed precision"},
            {"cmd": "License Manager", "center": "License Manager", "option": "16", "keywords": "license request key offline signature trial enterprise"},
            {"cmd": "AI Vision Assistant", "center": "AI Assistant", "option": "17", "keywords": "ai assistant advice diagnostic hyperparameter fix"},
            {"cmd": "Release Packaging", "center": "Packaging", "option": "18", "keywords": "package packaging build docker executable installer"},
        ]

    def open_command_palette(self) -> Optional[str]:
        """Interactive Command Palette search."""
        self.console.clear()
        self.console.print(Panel(Align.center(Text("AETHEL COMMAND PALETTE — SEARCH PLATFORM FEATURES", style="bold bright_cyan")), box=ROUNDED, border_style="cyan"))
        
        try:
            query = input("  Search Command / Feature (e.g. 'train', 'onnx', 'benchmark') > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return None

        if not query:
            return None

        matches = [
            item for item in self._command_index
            if query in item["cmd"].lower() or query in item["keywords"] or query in item["center"].lower()
        ]

        if not matches:
            print(f"\n[AETHEL] No command found matching '{query}'.")
            input("  Press Enter to return...")
            return None

        table = Table(title=f"[bold green]SEARCH RESULTS FOR '{query.upper()}'[/bold green]", box=ROUNDED, border_style="green", expand=True)
        table.add_column("Option", style="bold yellow", width=8)
        table.add_column("Feature / Command Name", style="bold white")
        table.add_column("Platform Center", style="bold cyan")

        for m in matches:
            table.add_row(f"[{m['option']}]", m['cmd'], m['center'])

        self.console.print(table)
        try:
            choice = input("\n  Select Option to Launch [Press Enter to cancel]: ").strip()
            return choice if choice else None
        except (KeyboardInterrupt, EOFError):
            return None

    def search_help(self, query: str) -> None:
        """Search documentation and workflows."""
        self.console.clear()
        self.console.print(Panel(Align.center(Text(f"SEARCHABLE HELP — '{query.upper()}'", style="bold bright_cyan")), box=ROUNDED, border_style="cyan"))
        
        matches = [
            item for item in self._command_index
            if query.lower() in item["cmd"].lower() or query.lower() in item["keywords"]
        ]

        table = Table(box=ROUNDED, border_style="bright_blue", expand=True)
        table.add_column("Feature", style="bold yellow")
        table.add_column("Menu Option", style="bold cyan")
        table.add_column("Search Keywords", style="white")

        for m in matches:
            table.add_row(m["cmd"], f"Menu [{m['option']}] ({m['center']})", m["keywords"])

        self.console.print(table)
        input("\n  [AETHEL Search] Press Enter to return...")


__all__ = ["HelpSystem"]
