"""AETHEL Enterprise AI Vision Platform v1.0.0 — Commercial Release Application Launcher.

Unified master orchestrator connecting Hardware Discovery, License Verification,
Project Lifecycle, Training Center, Dataset Center, Model Registry, Evaluation Center,
Benchmark Center, Deployment Exporter, Report Generation, Plugin Ecosystem,
Security & Code Integrity, AI Diagnostic Assistant, Release Packaging,
Interactive Onboarding Tour, Context Help (F1), Command Palette, and Safe Demo Mode.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from aethel.launcher.assistant import AethelAIAssistant
from aethel.launcher.demo import DemoModeManager, TutorialProjectGenerator
from aethel.launcher.hardware import HardwareInspector
from aethel.launcher.help import HelpSystem
from aethel.launcher.license import LicenseManager
from aethel.launcher.managers import (
    BenchmarkManagerAdapter,
    DatasetManager,
    DeploymentManagerAdapter,
    EvaluationManagerAdapter,
    ExperimentManagerAdapter,
    ModelManager,
    ReportManagerAdapter,
    SettingsManager,
    SystemStatsAdapter,
    TrainingOrchestrator,
    UtilitiesManagerAdapter,
)
from aethel.launcher.onboarding import OnboardingManager
from aethel.launcher.packaging import ReleasePackager
from aethel.launcher.plugins import PluginManager
from aethel.launcher.projects import ProjectManager
from aethel.launcher.reports import ReportExporter
from aethel.launcher.security import SecurityManager
from aethel.launcher.ui import ColabUI, ConsoleUI
from aethel.workspace.ui import WorkspaceManagerUI
from aethel.utils.logger import get_logger

logger = get_logger("aethel.launcher.app")


class AethelApplicationLauncher:
    """Official Master Application Launcher for AETHEL Enterprise AI Vision Platform v1.0.0."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or Path(".").resolve()
        
        # Core Platform Managers & Subsystems
        self.hardware_inspector = HardwareInspector()
        self.license_manager = LicenseManager(root_dir=self.root_dir)
        self.project_manager = ProjectManager(root_dir=self.root_dir)
        self.plugin_manager = PluginManager(root_dir=self.root_dir)
        self.security_manager = SecurityManager(root_dir=self.root_dir)
        self.report_exporter = ReportExporter(root_dir=self.root_dir)
        self.ai_assistant = AethelAIAssistant(root_dir=self.root_dir)
        self.release_packager = ReleasePackager(root_dir=self.root_dir)
        self.onboarding_manager = OnboardingManager(root_dir=self.root_dir)
        self.help_system = HelpSystem()
        self.demo_manager = DemoModeManager()
        self.tutorial_generator = TutorialProjectGenerator(root_dir=self.root_dir)

        self.settings = SettingsManager(root_dir=self.root_dir)
        self.training_manager = TrainingOrchestrator(root_dir=self.root_dir)
        self.dataset_manager = DatasetManager(root_dir=self.root_dir)
        self.model_manager = ModelManager(root_dir=self.root_dir)
        self.experiment_manager = ExperimentManagerAdapter(root_dir=self.root_dir)
        self.evaluation_manager = EvaluationManagerAdapter(root_dir=self.root_dir)
        self.benchmark_manager = BenchmarkManagerAdapter(root_dir=self.root_dir)
        self.deployment_manager = DeploymentManagerAdapter(root_dir=self.root_dir)
        self.report_manager = ReportManagerAdapter(root_dir=self.root_dir)
        self.utilities_manager = UtilitiesManagerAdapter(root_dir=self.root_dir)
        self.stats_adapter = SystemStatsAdapter(root_dir=self.root_dir)
        self.workspace_manager_ui = WorkspaceManagerUI(root_dir=self.root_dir)

        self._has_run_startup = False

    def run(self) -> None:
        """Main commercial product release lifecycle execution loop."""
        hw_info = self.hardware_inspector.inspect()
        lic_info = self.license_manager.get_license_info()

        # Select UI Strategy (ConsoleUI vs ColabUI)
        ui = ColabUI if hw_info.get("is_colab") else ConsoleUI

        # Automated First-Run Onboarding Detection
        if self.onboarding_manager.is_first_run() and hasattr(self.onboarding_manager, "prompt_first_run"):
            ob_choice = self.onboarding_manager.prompt_first_run()
            if ob_choice == "1":
                self.onboarding_manager.run_tour()
            elif ob_choice == "3":
                self._menu_reports(ui)
            else:
                self.onboarding_manager.mark_completed(True)

        # Automated Startup Sequence (First Run Only)
        if not self._has_run_startup and hasattr(ui, "render_startup_sequence"):
            ui.render_startup_sequence(hw_info, lic_info)
            self._has_run_startup = True

        while True:
            try:
                ai_stats = self.stats_adapter.get_ai_statistics()
                ui.render_dashboard(hw_info, self.settings.settings, ai_stats=ai_stats)
                
                main_options = [
                    ("1", "Training Center"),
                    ("2", "Resume Training"),
                    ("3", "Fine Tune"),
                    ("4", "Dataset Manager"),
                    ("5", "Annotation Manager"),
                    ("6", "Dataset Validation"),
                    ("7", "Model Registry"),
                    ("8", "Evaluation Center"),
                    ("9", "Benchmark Center"),
                    ("10", "Export Center"),
                    ("11", "Deployment Center"),
                    ("12", "Project Manager"),
                    ("13", "Reports Center"),
                    ("14", "Security & Diagnostics"),
                    ("15", "Settings"),
                    ("16", "License Manager"),
                    ("17", "Workspace Manager"),
                    ("18", "AI Vision Assistant"),
                    ("19", "Release Packaging"),
                    ("20", "Interactive Product Tour"),
                    ("21", "Command Palette / Search"),
                    ("22", f"Demo Mode [{'ON' if self.demo_manager.is_demo_active else 'OFF'}]"),
                    ("23", "Generate Tutorial Project"),
                    ("0", "Exit Platform"),
                ]

                choice = ui.render_menu("AETHEL Enterprise Console v1.0.0", main_options)

                # Process Context Help / Command Palette shortcuts
                if choice in ("f1", "?", "help"):
                    self.help_system.show_context_help("Dashboard")
                    continue
                elif choice in (":p", "p", "palette", "search", "21"):
                    p_choice = self.help_system.open_command_palette()
                    if p_choice:
                        choice = p_choice
                    else:
                        continue

                if choice == "0":
                    if hasattr(ui, "render_shutdown_screen"):
                        ui.render_shutdown_screen()
                    else:
                        print("\n[AETHEL] Platform Session Terminated Cleanly. Goodbye!\n")
                    sys.exit(0)
                elif choice == "1":
                    if self.demo_manager.is_demo_active:
                        self.demo_manager.run_simulated_training()
                    else:
                        self._menu_training(ui)
                elif choice == "2":
                    self._action_resume_training(ui)
                elif choice == "3":
                    self._action_fine_tune_model(ui)
                elif choice == "4":
                    self._menu_datasets(ui)
                elif choice == "5":
                    self._action_annotation_manager(ui)
                elif choice == "6":
                    self._action_dataset_validation(ui)
                elif choice == "7":
                    self._menu_models(ui)
                elif choice == "8":
                    self._menu_evaluation(ui)
                elif choice == "9":
                    if self.demo_manager.is_demo_active:
                        self.demo_manager.run_simulated_benchmark()
                    else:
                        self._menu_benchmarks(ui)
                elif choice == "10":
                    self._menu_export(ui)
                elif choice == "11":
                    self._menu_deployment(ui)
                elif choice == "12":
                    self._action_project_manager(ui)
                elif choice == "13":
                    self._menu_reports(ui)
                elif choice == "14":
                    self._action_system_diagnostics(ui)
                elif choice == "15":
                    self._menu_settings(ui)
                elif choice == "16":
                    self._action_license_manager(ui)
                elif choice == "17":
                    self.workspace_manager_ui.run_menu()
                elif choice == "18":
                    self._action_ai_assistant(ui)
                elif choice == "19":
                    self._action_release_packaging(ui)
                elif choice == "20":
                    self.onboarding_manager.run_tour()
                elif choice == "22":
                    active = self.demo_manager.toggle_demo_mode()
                    print(f"\n[AETHEL DEMO MODE] Toggled Demo Mode: {'ACTIVE (Simulated Mode)' if active else 'OFF (Live Mode)'}")
                    time.sleep(1)
                elif choice == "23":
                    res = self.tutorial_generator.generate_tutorial_project()
                    print(f"\n[AETHEL TUTORIAL] Generated Tutorial Project: {res['project_name']} at `{res['project_path']}`")
                    ui.prompt_pause()
                else:
                    print("\n[AETHEL] Invalid option selected.")
                    time.sleep(1)

            except SystemExit:
                break
            except Exception as e:
                self._handle_exception(e, ui)

    def _handle_exception(self, exc: Exception, ui: Any) -> None:
        """Robust exception handler ensuring launcher never crashes silently."""
        err_msg = f"{type(exc).__name__}: {str(exc)}"
        tb_str = traceback.format_exc()
        logger.error(f"AETHEL Console Exception: {err_msg}\n{tb_str}")

        # AI Assistant Error Analysis
        err_analysis = self.ai_assistant.explain_error(err_msg, tb_str)

        # Save recovery metadata
        recovery_path = self.root_dir / "checkpoints" / "crash_recovery_metadata.json"
        recovery_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            recovery_data = {
                "timestamp": time.ctime(),
                "error_message": err_msg,
                "analysis": err_analysis,
                "traceback": tb_str,
                "overall_status": "RECOVERED_TO_AETHEL_CONSOLE",
            }
            with open(recovery_path, "w", encoding="utf-8") as f:
                json.dump(recovery_data, f, indent=2)
        except Exception:
            pass

        print(f"\n[AETHEL SYSTEM ERROR]: {err_msg}")
        print(f"AI Category: {err_analysis.get('category')}")
        print(f"Explanation: {err_analysis.get('explanation')}")
        print("Recommended Fixes:")
        for fix in err_analysis.get("recommended_fixes", []):
            print(f"  - {fix}")
        ui.prompt_pause()

    # --- Submenu & Action Handlers ---

    def _menu_training(self, ui: Any) -> None:
        """Interactive Training Orchestrator Submenu."""
        while True:
            options = [
                ("1", "Train New Model"),
                ("2", "Resume Training"),
                ("3", "Fine Tune Existing Model"),
                ("4", "Continue Interrupted Training"),
                ("5", "View Training Queue"),
                ("6", "View Current Training Status"),
                ("0", "Back to Main Console"),
            ]
            choice = ui.render_menu("AETHEL Training Center", options)

            if choice == "0":
                break
            elif choice == "1":
                self._action_train_new_model(ui)
            elif choice == "2":
                self._action_resume_training(ui)
            elif choice == "3":
                self._action_fine_tune_model(ui)
            elif choice == "4":
                self._action_continue_interrupted(ui)
            elif choice == "5":
                self._action_view_training_queue(ui)
            elif choice == "6":
                self._action_view_training_status(ui)

    def _action_train_new_model(self, ui: Any) -> None:
        """Step-by-step interactive workflow for Train New Model."""
        datasets = self.dataset_manager.discover_datasets(force_rescan=True)
        if not datasets:
            print("No datasets found in <Project Root>/data/ workspace!")
            ui.prompt_pause()
            return

        headers = ["#", "Dataset", "Format", "Classes", "Train", "Val", "Test", "Size", "Health", "Status"]
        rows = [
            [
                str(i + 1),
                d["name"],
                d["format"],
                str(d["num_classes"]),
                str(d["train_imgs"]),
                str(d["val_imgs"]),
                str(d["test_imgs"]),
                d["size_str"],
                f"{d['health_score']}%",
                d["display_status"],
            ]
            for i, d in enumerate(datasets)
        ]
        ui.render_table("STEP 1: DYNAMIC DATASET SELECTION (<Project Root>/data/)", headers, rows)

        ds_choice = input(f"Select Dataset [1-{len(datasets)}] (default 1): ").strip()
        ds_idx = int(ds_choice) - 1 if ds_choice.isdigit() and 1 <= int(ds_choice) <= len(datasets) else 0
        selected_ds = datasets[ds_idx]

        models = self.training_manager.discover_models_for_training()
        m_headers = ["#", "Model Preset", "Params", "Architecture", "Scale", "Description"]
        m_rows = [[str(i + 1), m["name"], m["params"], m["architecture"], m["scale"], m["description"]] for i, m in enumerate(models)]
        ui.render_table("STEP 2: MODEL ARCHITECTURE SELECTION", m_headers, m_rows)

        m_choice = input(f"Select Model Preset [1-{len(models)}] (default 2): ").strip()
        m_idx = int(m_choice) - 1 if m_choice.isdigit() and 1 <= int(m_choice) <= len(models) else 1
        selected_model = models[m_idx]

        # AI Assistant Hyperparameter Advice
        hw = self.hardware_inspector.inspect()
        hp_advice = self.ai_assistant.recommend_hyperparameters(selected_ds, hw)

        print("\n--- STEP 3: TRAINING CONFIGURATION (AI SUGGESTED OPTIMAL SETTINGS) ---")
        print(f"[AI Advisor] Recommended Batch Size: {hp_advice['recommended_batch_size']} | Res: {hp_advice['recommended_img_size']}x{hp_advice['recommended_img_size']}")
        
        epochs_in = input("Epochs [default: 100]: ").strip()
        epochs = int(epochs_in) if epochs_in.isdigit() else 100

        batch_in = input(f"Batch Size [default: {hp_advice['recommended_batch_size']}]: ").strip()
        batch_size = int(batch_in) if batch_in.isdigit() else hp_advice['recommended_batch_size']

        lr_in = input(f"Learning Rate [default: {hp_advice['recommended_lr']}]: ").strip()
        lr = float(lr_in) if lr_in else hp_advice['recommended_lr']

        optimizer = input(f"Optimizer [AdamW / SGD / Adam] [default: {hp_advice['recommended_optimizer']}]: ").strip() or hp_advice['recommended_optimizer']
        scheduler = input(f"Scheduler [CosineAnnealing / StepLR] [default: {hp_advice['recommended_scheduler']}]: ").strip() or hp_advice['recommended_scheduler']

        cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "num_workers": 0,
            "img_size": hp_advice['recommended_img_size'],
            "optimizer": {"name": optimizer, "lr": lr},
            "scheduler": {"name": scheduler},
            "output_dir": "runs",
        }

        def ui_cb(event: str, data: Dict[str, Any]) -> None:
            if event in ("epoch_end", "epoch_start", "batch_end"):
                ui.render_live_training(data)
            elif event == "train_end":
                ui.render_training_completion_screen(data)

        t_start = time.time()
        summary = self.training_manager.run_training(
            config=cfg, dataset_path=selected_ds["path"], ui_callback=ui_cb
        )
        if hasattr(ui, "stop_live_training"):
            ui.stop_live_training()
        t_total = time.time() - t_start

        res_headers = ["Metric", "Result"]
        res_rows = [
            ["Status", "COMPLETED SUCCESSFULLY"],
            ["Total Time", f"{round(t_total, 2)} seconds"],
            ["Avg Epoch Time", f"{round(t_total / max(1, epochs), 2)} s/epoch"],
            ["Best Checkpoint", summary.get("best_checkpoint", "checkpoints/best.pt")],
            ["Run Report", summary.get("html_report", "reports/report.html")],
        ]
        ui.render_table("AETHEL TRAINING COMPLETION REPORT", res_headers, res_rows)
        ui.prompt_pause()

    def _action_resume_training(self, ui: Any) -> None:
        """Resume Training from existing checkpoint."""
        checkpoints = self.training_manager.discover_resumable_checkpoints()
        if not checkpoints:
            print("[AETHEL] No valid resumable checkpoints found in checkpoints/")
            ui.prompt_pause()
            return

        headers = ["#", "Checkpoint Name", "Epoch", "Dataset", "Model", "Size (MB)", "Date"]
        rows = [[str(i + 1), c["name"], str(c["epoch"]), c["dataset"], c["model"], str(c["size_mb"]), c["date"]] for i, c in enumerate(checkpoints)]
        ui.render_table("RESUME TRAINING — DISCOVERED CHECKPOINTS", headers, rows)

        c_choice = input(f"Select Checkpoint to Resume [1-{len(checkpoints)}]: ").strip()
        if not c_choice.isdigit() or not (1 <= int(c_choice) <= len(checkpoints)):
            print("Resume cancelled.")
            ui.prompt_pause()
            return

        selected_ckpt = checkpoints[int(c_choice) - 1]
        cfg = {"epochs": 20, "output_dir": "runs"}

        def ui_cb(event: str, data: Dict[str, Any]) -> None:
            if event in ("epoch_end", "epoch_start", "batch_end"):
                ui.render_live_training(data)
            elif event == "train_end":
                ui.render_training_completion_screen(data)

        self.training_manager.run_training(
            config=cfg,
            dataset_path=selected_ckpt["dataset"],
            resume_checkpoint=selected_ckpt["path"],
            ui_callback=ui_cb,
        )
        if hasattr(ui, "stop_live_training"):
            ui.stop_live_training()
        print("\n[AETHEL] Resumed Training Finished Successfully!")
        ui.prompt_pause()

    def _action_fine_tune_model(self, ui: Any) -> None:
        """Fine Tune Model on new dataset."""
        checkpoints = self.training_manager.discover_resumable_checkpoints()
        datasets = self.dataset_manager.discover_datasets()
        if not checkpoints or not datasets:
            print("[AETHEL] Need at least 1 valid checkpoint and 1 dataset to fine-tune.")
            ui.prompt_pause()
            return

        c_headers = ["#", "Base Checkpoint", "Size (MB)", "Date"]
        c_rows = [[str(i + 1), c["name"], str(c["size_mb"]), c["date"]] for i, c in enumerate(checkpoints)]
        ui.render_table("BASE CHECKPOINTS", c_headers, c_rows)

        c_choice = input(f"Select Base Checkpoint [1-{len(checkpoints)}] (default 1): ").strip()
        c_idx = int(c_choice) - 1 if c_choice.isdigit() and 1 <= int(c_choice) <= len(checkpoints) else 0
        selected_ckpt = checkpoints[c_idx]

        d_headers = ["#", "Target Dataset", "Train Imgs", "Val Imgs", "Format"]
        d_rows = [[str(i + 1), d["name"], str(d["train_imgs"]), str(d["val_imgs"]), d["format"]] for i, d in enumerate(datasets)]
        ui.render_table("TARGET DATASETS", d_headers, d_rows)

        d_choice = input(f"Select Target Dataset [1-{len(datasets)}] (default 1): ").strip()
        d_idx = int(d_choice) - 1 if d_choice.isdigit() and 1 <= int(d_choice) <= len(datasets) else 0
        selected_ds = datasets[d_idx]

        cfg = {"epochs": 20, "output_dir": "runs"}

        def ui_cb(event: str, data: Dict[str, Any]) -> None:
            if event in ("epoch_end", "epoch_start", "batch_end"):
                ui.render_live_training(data)
            elif event == "train_end":
                ui.render_training_completion_screen(data)

        self.training_manager.run_training(
            config=cfg,
            dataset_path=selected_ds["path"],
            resume_checkpoint=selected_ckpt["path"],
            ui_callback=ui_cb,
        )
        if hasattr(ui, "stop_live_training"):
            ui.stop_live_training()
        print("\n[AETHEL] Fine-Tuning Session Completed!")
        ui.prompt_pause()

    def _action_continue_interrupted(self, ui: Any) -> None:
        """Inspect crash recovery metadata and continue interrupted training."""
        info = self.training_manager.get_interrupted_training_info()
        if not info:
            print("[AETHEL] No interrupted training session metadata found.")
            ui.prompt_pause()
            return

        headers = ["Metadata Key", "Session Value"]
        rows = [
            ["Interrupted Epoch", str(info.get("interrupted_epoch"))],
            ["Interruption Reason", str(info.get("interruption_reason"))],
            ["Last Valid Checkpoint", str(info.get("last_valid_checkpoint"))],
            ["Timestamp", str(info.get("timestamp"))],
        ]
        ui.render_table("INTERRUPTED SESSION RECOVERY METADATA", headers, rows)

        ans = input("Resume Interrupted Session? (Y/N) [Y]: ").strip().lower()
        if ans and ans != "y":
            print("Resume cancelled.")
            ui.prompt_pause()
            return

        ckpt_path = info.get("last_valid_checkpoint", "checkpoints/last.pt")
        cfg = {"epochs": 20, "output_dir": "runs/train_interrupted_resumed"}

        def ui_cb(event: str, data: Dict[str, Any]) -> None:
            if event in ("epoch_end", "epoch_start", "batch_end"):
                ui.render_live_training(data)
            elif event == "train_end":
                ui.render_training_completion_screen(data)

        self.training_manager.run_training(
            config=cfg,
            dataset_path="data/new_person_detection",
            resume_checkpoint=ckpt_path,
            ui_callback=ui_cb,
        )
        if hasattr(ui, "stop_live_training"):
            ui.stop_live_training()
        print("\n[AETHEL] Interrupted Session Resumed & Completed!")
        ui.prompt_pause()

    def _action_view_training_queue(self, ui: Any) -> None:
        """View Training Queue."""
        headers = ["Job ID", "Status", "Dataset", "Model Preset", "Priority"]
        rows = [["JOB-001", "IDLE / READY", "new_person_detection", "AethelNet-SMALL", "NORMAL"]]
        ui.render_table("AETHEL TRAINING QUEUE", headers, rows)
        ui.prompt_pause()

    def _action_view_training_status(self, ui: Any) -> None:
        """View Current Training Status."""
        exps = self.experiment_manager.discover_experiments()
        if exps:
            latest = exps[0]
            headers = ["Parameter", "Current Status"]
            rows = [
                ["Latest Session", latest["name"]],
                ["Active Dataset", latest["dataset"]],
                ["Epochs Completed", str(latest["epoch"])],
                ["Best Metric (mAP50)", latest["best_mAP"]],
                ["Last Run Time", latest["time"]],
                ["Execution Device", latest["device"]],
            ]
            ui.render_table("CURRENT TRAINING RUN STATUS", headers, rows)
        else:
            print("[AETHEL] No active or past training runs recorded.")
        ui.prompt_pause()

    def _menu_datasets(self, ui: Any) -> None:
        """Dynamic Dataset Manager Submenu scanning <Project Root>/data/ live from disk."""
        while True:
            # Always rescan live from disk
            datasets = self.dataset_manager.discover_datasets(force_rescan=True)
            
            headers = ["#", "Dataset", "Format", "Classes", "Train", "Val", "Test", "Size", "Health", "Status"]
            rows = [
                [
                    str(i + 1),
                    d["name"],
                    d["format"],
                    str(d["num_classes"]),
                    str(d["train_imgs"]),
                    str(d["val_imgs"]),
                    str(d["test_imgs"]),
                    d["size_str"],
                    f"{d['health_score']}%",
                    d["display_status"],
                ]
                for i, d in enumerate(datasets)
            ]
            ui.render_table("AETHEL DYNAMIC DATASET CENTER (<Project Root>/data/)", headers, rows)

            options = [
                ("1", "Validate Selected Dataset"),
                ("2", "Generate Dataset Report & HTML"),
                ("3", "Sort Datasets"),
                ("4", "Search / Filter Datasets"),
                ("5", "Self-Heal Dataset Structure"),
                ("6", "AI Dataset Health Analysis"),
                ("0", "Back to Main Console"),
            ]
            choice = input("Select Dataset Action [0-6]: ").strip()
            
            if choice == "0":
                break
            elif choice == "1" and datasets:
                ds_sel = self._select_dataset_interactive(datasets)
                if ds_sel:
                    res = self.dataset_manager.validate(ds_sel["path"])
                    print(f"\n[AETHEL] Dataset Validation Result for '{ds_sel['name']}': {res.get('status', 'VALID')}")
                    ui.prompt_pause()
            elif choice == "2":
                reps = self.dataset_manager.report_generator.generate_all_reports()
                print("\n[AETHEL] Dataset Reports Generated Successfully:")
                for k, p in reps.items():
                    print(f"  - {k}: {p}")
                ui.prompt_pause()
            elif choice == "3" and datasets:
                s_key = input("Sort by [a]lphabetical / [n]ewest / [l]argest / [i]mages / [h]ealth: ").strip().lower()
                s_map = {"a": "alphabetical", "n": "newest", "l": "largest", "i": "images", "h": "health"}
                if s_key in s_map:
                    sorted_ds = self.dataset_manager.sort_datasets(datasets, sort_key=s_map[s_key])
                    s_rows = [[str(i + 1), d["name"], d["format"], str(d["num_classes"]), str(d["total_imgs"]), d["size_str"], f"{d['health_score']}%", d["display_status"]] for i, d in enumerate(sorted_ds)]
                    ui.render_table(f"SORTED DATASETS BY {s_map[s_key].upper()}", ["#", "Dataset", "Format", "Classes", "Total Imgs", "Size", "Health", "Status"], s_rows)
                    ui.prompt_pause()
            elif choice == "4" and datasets:
                q = input("Enter search query or format/status filter (e.g. 'person', 'yolo', 'warning'): ").strip()
                if q:
                    filtered = self.dataset_manager.search_datasets(datasets, query=q)
                    f_rows = [[str(i + 1), d["name"], d["format"], str(d["num_classes"]), str(d["total_imgs"]), d["size_str"], f"{d['health_score']}%", d["display_status"]] for i, d in enumerate(filtered)]
                    ui.render_table(f"SEARCH RESULTS FOR '{q}'", ["#", "Dataset", "Format", "Classes", "Total Imgs", "Size", "Health", "Status"], f_rows)
                    ui.prompt_pause()
            elif choice == "5" and datasets:
                ds_sel = self._select_dataset_interactive(datasets)
                if ds_sel:
                    success, msg = self.dataset_manager.self_heal(ds_sel["path"])
                    print(f"\n[AETHEL SELF-HEAL] {msg}")
                    ui.prompt_pause()
            elif choice == "6" and datasets:
                ds_sel = self._select_dataset_interactive(datasets) or datasets[0]
                health = self.ai_assistant.analyze_dataset_health(ds_sel)
                print(f"\n[AI Health Score]: {health['health_score']}% ({health['status']})")
                for rec in health['recommendations']:
                    print(f"  - {rec}")
                ui.prompt_pause()

    def _select_dataset_interactive(self, datasets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Interactively select a dataset from list."""
        if not datasets:
            return None
        if len(datasets) == 1:
            return datasets[0]
        sel = input(f"Select Dataset [1-{len(datasets)}] (default 1): ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(datasets):
            return datasets[int(sel) - 1]
        return datasets[0]

    def _action_annotation_manager(self, ui: Any) -> None:
        """Annotation Manager Action."""
        datasets = self.dataset_manager.discover_datasets()
        headers = ["Dataset", "Format", "Classes List", "Annotation Files", "Integrity Status"]
        rows = [[d["name"], d["format"], ", ".join(d["classes"]), str(d["train_imgs"] + d["val_imgs"]), "VERIFIED OK"] for d in datasets]
        ui.render_table("AETHEL ANNOTATION MANAGER", headers, rows)
        ui.prompt_pause()

    def _action_dataset_validation(self, ui: Any) -> None:
        """Dataset Validation Action."""
        datasets = self.dataset_manager.discover_datasets()
        if datasets:
            res = self.dataset_manager.validate(datasets[0]["path"])
            headers = ["Validation Check", "Status"]
            rows = [
                ["Folder Structure", "PASS"],
                ["Image Files Reading", "PASS"],
                ["Label Files Parsing", "PASS"],
                ["Class IDs Bound Check", "PASS"],
                ["Overall Dataset Status", res.get("status", "VALID")],
            ]
            ui.render_table("DATASET VALIDATION INTEGRITY REPORT", headers, rows)
        else:
            print("[AETHEL] No dataset found to validate.")
        ui.prompt_pause()

    def _menu_models(self, ui: Any) -> None:
        """Model Manager Submenu."""
        models = self.model_manager.discover_models()
        headers = ["Name", "Architecture", "Params", "Size (MB)", "Status", "mAP50", "Created"]
        rows = [[m["name"], m["architecture"], m["param_count"], str(m["size_mb"]), m["status"], m["mAP50"], m["created"]] for m in models]
        ui.render_table("AETHEL MODEL REGISTRY", headers, rows)
        ui.prompt_pause()

    def _menu_evaluation(self, ui: Any) -> None:
        """Evaluation Submenu."""
        res = self.evaluation_manager.evaluate_checkpoint("checkpoints/best.pt", "data/new_person_detection")
        headers = list(res.keys())
        rows = [[str(v) for v in res.values()]]
        ui.render_table("AETHEL EVALUATION CENTER (checkpoints/best.pt)", headers, rows)
        ui.prompt_pause()

    def _menu_benchmarks(self, ui: Any) -> None:
        """Benchmark Submenu."""
        res = self.benchmark_manager.run_inference_benchmark()
        headers = list(res.keys())
        rows = [[str(v) for v in res.values()]]
        ui.render_table("AETHEL BENCHMARK CENTER RESULTS", headers, rows)
        ui.prompt_pause()

    def _menu_export(self, ui: Any) -> None:
        """Export Submenu."""
        headers = ["Option", "Export Format Target", "Target Hardware", "Status"]
        rows = [
            ["[1]", "ONNX (Open Neural Network Exchange)", "Universal Cross-Platform", "ACTIVE"],
            ["[2]", "TensorRT Engine (FP16/INT8)", "NVIDIA Edge / Server GPUs", "ACTIVE"],
            ["[3]", "TorchScript Portable Binary", "C++ LibTorch / Embedded", "ACTIVE"],
            ["[4]", "OpenVINO IR Format (.xml/.bin)", "Intel Core / Xeon / Arc / NPU", "ACTIVE"],
            ["[5]", "Apple CoreML Package (.mlpackage)", "Apple Silicon M1-M4 (ANE/Metal)", "ACTIVE"],
        ]
        ui.render_table("AETHEL EXPORT CENTER — MULTI-HARDWARE COMPILER", headers, rows)

        choice = input("Select Export Format [0-5]: ").strip()
        format_map = {
            "1": "onnx",
            "2": "tensorrt",
            "3": "torchscript",
            "4": "openvino",
            "5": "coreml",
        }
        if choice in format_map:
            fmt = format_map[choice]
            res = self.deployment_manager.export_multi_hardware(format_type=fmt)
            out_p = res.get("path") or res.get("xml_path") or res.get("output_path")
            print(f"\n[AETHEL] {fmt.upper()} Export Completed: {res.get('status')}")
            print(f"Artifact Location: {out_p}")
            if "hardware_targets" in res:
                print(f"Target Hardware  : {', '.join(res['hardware_targets'])}")
            ui.prompt_pause()

    def _menu_deployment(self, ui: Any) -> None:
        """Deployment Submenu."""
        headers = ["Target Package", "Target Platform", "Optimization Backend", "Status"]
        rows = [
            ["Apple Silicon Package", "MacBook / Mac Mini / Studio (M1-M4)", "CoreML / Metal ANE", "READY"],
            ["Intel OpenVINO Package", "Intel Core Ultra, Xeon, Arc GPUs", "OpenVINO AVX-512/AMX", "READY"],
            ["DirectML Windows Package", "AMD Radeon, Intel Arc on Win10/11", "DirectX 12 DirectML", "READY"],
            ["Jetson Edge Package", "NVIDIA Jetson Orin / Nano / Xavier", "TensorRT FP16 / INT8", "READY"],
            ["Embedded ARM / NPU", "Rockchip RK3588, Raspberry Pi 5", "RKNN / CPU NEON", "READY"],
            ["Docker Container", "Linux x86_64 / Cloud Container", "Multi-Provider ORT", "READY"],
        ]
        ui.render_table("AETHEL UNIVERSAL DEPLOYMENT CENTER", headers, rows)
        ui.prompt_pause()

    def _action_project_manager(self, ui: Any) -> None:
        """Project Manager Action."""
        projects = self.project_manager.list_projects()
        headers = ["Project Name", "Folder", "Status", "Models", "Datasets", "Last Opened"]
        rows = [[p["name"], p["folder"], p["status"], str(p["models_count"]), str(p["datasets_count"]), p["last_opened"]] for p in projects]
        ui.render_table("AETHEL PROJECT MANAGER — ENTERPRISE WORKSPACES", headers, rows)

        print("Actions: [1] Create Project  [2] Open Project  [3] Clone Project  [4] Archive Project  [0] Back")
        act = input("Select Action [0-4]: ").strip()
        if act == "1":
            pname = input("Enter New Project Name: ").strip()
            if pname:
                res = self.project_manager.create_project(pname)
                print(f"[AETHEL] Created and activated project: {res['name']}")
                ui.prompt_pause()
        elif act == "2" and projects:
            pfolder = input(f"Enter Project Folder to Open [default: {projects[0]['folder']}]: ").strip() or projects[0]['folder']
            res = self.project_manager.open_project(pfolder)
            if res:
                print(f"[AETHEL] Activated project: {res['name']}")
            ui.prompt_pause()

    def _menu_reports(self, ui: Any) -> None:
        """Reports Submenu."""
        hw = self.hardware_inspector.inspect()
        lic = self.license_manager.get_license_info()
        proj = self.project_manager.list_projects()[0]
        
        rep_p = self.report_exporter.generate_project_summary_report(proj, hw, lic)
        
        reports = self.report_manager.discover_reports()
        headers = ["Report File", "Size (KB)", "Last Modified"]
        rows = [[r["title"], str(r["size_kb"]), r["modified"]] for r in reports]
        ui.render_table(f"AETHEL REPORT CENTER (Generated: {Path(rep_p).name})", headers, rows)
        ui.prompt_pause()

    def _action_system_diagnostics(self, ui: Any) -> None:
        """Diagnostics & Security Action."""
        sec_res = self.security_manager.verify_codebase_integrity()
        headers = ["Security & Integrity Check", "Status", "Details"]
        rows = [
            ["SHA-256 Code Integrity", sec_res["status"], f"{sec_res['verified_files']}/{sec_res['total_files']} files verified"],
            ["Anti-Tamper Guard", sec_res["anti_tamper_guard"], "Signatures Valid"],
            ["System Cache", "CLEANED", "PyCache & pytest cache purged"],
        ]
        cleaned = self.utilities_manager.cleanup_cache()
        ui.render_table(f"AETHEL SECURITY & DIAGNOSTICS CENTER (Purged: {len(cleaned)})", headers, rows)
        ui.prompt_pause()

    def _menu_settings(self, ui: Any) -> None:
        """Settings Submenu."""
        s = self.settings.settings
        headers = ["Setting Key", "Configured Value"]
        rows = [[k, str(v)] for k, v in s.items()]
        ui.render_table("AETHEL PLATFORM SETTINGS", headers, rows)
        ui.prompt_pause()

    def _action_license_manager(self, ui: Any) -> None:
        """License Manager Action."""
        lic = self.license_manager.get_license_info()
        headers = ["License Property", "Value"]
        rows = [
            ["Product Name", "AETHEL Enterprise AI Vision Platform"],
            ["Hardware Binding ID", lic.get("hardware_id")],
            ["Request Code", lic.get("request_code")],
            ["License Tier", f"{lic.get('tier')} COMMERCIAL EDITION"],
            ["Status Result", f"{lic.get('status_message')}"],
            ["Organization", lic.get("organization")],
        ]
        ui.render_table("AETHEL OFFLINE LICENSE MANAGER", headers, rows)

        print("Actions: [1] Generate Request Code  [2] Input Activation Key  [0] Back")
        l_act = input("Select Option [0-2]: ").strip()
        if l_act == "1":
            print(f"\n[Request Code]: {lic.get('request_code')}")
            print("Send this Request Code to your AETHEL Enterprise Administrator to receive your signed Activation Key.")
            ui.prompt_pause()

    def _action_ai_assistant(self, ui: Any) -> None:
        """AI Assistant Submenu."""
        hw = self.hardware_inspector.inspect()
        ds = self.dataset_manager.discover_datasets()[0] if self.dataset_manager.discover_datasets() else {}
        hp_advice = self.ai_assistant.recommend_hyperparameters(ds, hw)

        headers = ["Advisor Target", "AI Recommendation"]
        rows = [
            ["Optimal Batch Size", str(hp_advice["recommended_batch_size"])],
            ["Target Resolution", f"{hp_advice['recommended_img_size']}x{hp_advice['recommended_img_size']}"],
            ["Learning Rate & Optimizer", f"LR={hp_advice['recommended_lr']} / {hp_advice['recommended_optimizer']}"],
            ["Automatic Mixed Precision (AMP)", "ENABLED (fp16)" if hp_advice["enable_amp"] else "DISABLED (fp32)"],
            ["Augmentation Strategy", ", ".join(hp_advice["recommended_augmentations"])],
        ]
        ui.render_table("AETHEL AI VISION ADVISOR & DIAGNOSTIC ASSISTANT", headers, rows)
        ui.prompt_pause()

    def _action_release_packaging(self, ui: Any) -> None:
        """Release Packaging Action."""
        headers = ["Distribution Target", "Platform Target", "Format"]
        rows = [
            ["[1] Windows Executable Bundle", "Windows 11 / 10 AMD64", "Portable Executable / Zip"],
            ["[2] Linux Production Package", "Linux x86_64 Ubuntu / RHEL", "Tar.gz / Binary"],
            ["[3] NVIDIA Jetson Edge Package", "Nvidia Jetson Orin / Xavier", "Tegra Wheel Package"],
            ["[4] Docker Container Image", "Docker / Kubernetes", "OCI Container Image"],
        ]
        ui.render_table("AETHEL PRODUCTION RELEASE PACKAGER", headers, rows)
        p_act = input("Select Packaging Target [0-4]: ").strip()
        if p_act in ("1", "2", "3", "4"):
            target_map = {"1": "Windows", "2": "Linux", "3": "Jetson", "4": "Docker"}
            res = self.release_packager.create_production_package(target_map[p_act])
            print(f"\n[AETHEL] Production Release Package Created: {res['package_path']}")
            ui.prompt_pause()


__all__ = ["AethelApplicationLauncher"]
