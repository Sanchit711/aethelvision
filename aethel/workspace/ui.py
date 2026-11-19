"""AETHEL Workspace Manager Terminal UI Module (Phases 1, 2, 14).

Renders Rich tables, submenus, storage analyzer dashboards, health reports,
and interactive prompts for the Workspace Manager subsystem.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.align import Align
from rich.box import ROUNDED, SQUARE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aethel.workspace.analyzer import format_bytes
from aethel.workspace.manager import WorkspaceManager, open_folder_in_explorer


class WorkspaceManagerUI:
    """Rich interactive Console UI handler for Workspace Manager subsystem."""

    console = Console()

    def __init__(self, root_dir: Path = Path("."), ui_provider: Optional[Any] = None) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.manager = WorkspaceManager(root_dir=self.root_dir)
        self.ui_provider = ui_provider

    def run_menu(self) -> None:
        """Execute interactive Workspace Manager main loop (Phase 1)."""
        while True:
            self._clear_screen()
            self._render_workspace_banner()

            options = [
                ("1", "Show Experiments"),
                ("2", "Open Experiment Folder"),
                ("3", "Delete Experiment"),
                ("4", "Delete Multiple Experiments"),
                ("5", "Delete Failed Experiments"),
                ("6", "Delete Empty Experiments"),
                ("7", "Delete Old Checkpoints"),
                ("8", "Delete Validation Images"),
                ("9", "Delete Reports"),
                ("10", "Delete Exports"),
                ("11", "Archive Experiment"),
                ("12", "Restore Archived Experiment"),
                ("13", "Compress Experiment"),
                ("14", "Storage Analyzer"),
                ("15", "Workspace Health"),
                ("16", "Cleanup Cache"),
                ("17", "Vacuum Workspace"),
                ("18", "Auto Cleanup Policy"),
                ("19", "Recover Deleted Experiments"),
                ("20", "Search Experiments"),
                ("0", "Back"),
            ]

            # Render options table in 2 columns
            menu_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
            menu_table.add_column("Col1", style="bold yellow", width=35)
            menu_table.add_column("Col2", style="bold yellow", width=35)

            half = (len(options) - 1 + 1) // 2
            for i in range(half):
                k1, v1 = options[i]
                r2 = options[i + half] if i + half < len(options) - 1 else None
                c1_str = f"[{k1}] {v1}"
                c2_str = f"[{r2[0]}] {r2[1]}" if r2 else ""
                menu_table.add_row(c1_str, c2_str)

            menu_panel = Panel(
                menu_table,
                title="[bold bright_cyan] WORKSPACE MANAGER [/bold bright_cyan]",
                box=ROUNDED,
                border_style="bright_blue",
                padding=(1, 2),
            )
            self.console.print(menu_panel)
            self.console.print(Align.center(Text("  [0] Back to Main Menu", style="bold red")))
            self.console.print()

            try:
                choice = input("Select Workspace Action [0-20]: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if choice == "0":
                break
            elif choice == "1":
                self._action_show_experiments()
            elif choice == "2":
                self._action_open_experiment_folder()
            elif choice == "3":
                self._action_delete_experiment()
            elif choice == "4":
                self._action_delete_multiple()
            elif choice == "5":
                self._action_delete_by_type("failed")
            elif choice == "6":
                self._action_delete_by_type("empty")
            elif choice == "7":
                self._action_delete_old_checkpoints()
            elif choice == "8":
                self._action_delete_asset("validation_images")
            elif choice == "9":
                self._action_delete_asset("reports")
            elif choice == "10":
                self._action_delete_asset("exports")
            elif choice == "11":
                self._action_archive_experiment()
            elif choice == "12":
                self._action_restore_archive()
            elif choice == "13":
                self._action_compress_experiment()
            elif choice == "14":
                self._action_storage_analyzer()
            elif choice == "15":
                self._action_workspace_health()
            elif choice == "16":
                self._action_cleanup_cache()
            elif choice == "17":
                self._action_vacuum_workspace()
            elif choice == "18":
                self._action_auto_cleanup_policy()
            elif choice == "19":
                self._action_recover_deleted()
            elif choice == "20":
                self._action_search_experiments()

    def _clear_screen(self) -> None:
        if hasattr(self.ui_provider, "clear"):
            self.ui_provider.clear()
        else:
            self.console.clear()

    def _pause(self) -> None:
        if hasattr(self.ui_provider, "prompt_pause"):
            self.ui_provider.prompt_pause()
        else:
            input("\nPress Enter to continue...")

    def _render_workspace_banner(self) -> None:
        banner_text = Text("AETHEL ENTERPRISE WORKSPACE MANAGER", style="bold bright_cyan center")
        self.console.print(Panel(Align.center(banner_text), box=SQUARE, style="on blue"))

    def _action_show_experiments(self) -> None:
        """[1] Show Experiments with sorting options."""
        exps = self.manager.get_experiments()
        if not exps:
            self.console.print("[bold yellow]No experiments found in workspace database.[/bold yellow]")
            self._pause()
            return

        table = Table(title="[bold green] WORKSPACE EXPERIMENT INDEX [/bold green]", box=ROUNDED, border_style="green", expand=True)
        table.add_column("#", style="bold yellow", width=4)
        table.add_column("Run ID", style="bold cyan")
        table.add_column("Model", style="white")
        table.add_column("Dataset", style="white")
        table.add_column("Epoch", style="cyan")
        table.add_column("Best mAP", style="bold green")
        table.add_column("Status", style="bold white")
        table.add_column("Disk Usage", style="bold yellow")
        table.add_column("Date", style="dim white")

        for i, e in enumerate(exps):
            size_str = format_bytes(e.get("disk_usage", 0))
            table.add_row(
                str(i + 1),
                e.get("run_id", "N/A"),
                e.get("model", "N/A"),
                e.get("dataset", "N/A"),
                str(e.get("epoch", 0)),
                str(e.get("best_mAP", "N/A")),
                f"[bold green]{e.get('status')}[/bold green]" if e.get("status") == "Completed" else f"[bold yellow]{e.get('status')}[/bold yellow]",
                size_str,
                str(e.get("timestamp", "N/A")),
            )

        self.console.print(table)
        
        # Optional sorting prompt
        sort_choice = input("\nSort Options: [d]ate  [s]ize  [m]AP  [e]poch  [b]ack: ").strip().lower()
        if sort_choice in ("d", "s", "m", "e"):
            sort_map = {"d": "date", "s": "size", "m": "map", "e": "epoch"}
            sorted_exps = self.manager.sort_experiments(exps, sort_key=sort_map[sort_choice])
            
            s_table = Table(title=f"[bold green] SORTED BY {sort_map[sort_choice].upper()} [/bold green]", box=ROUNDED, border_style="green", expand=True)
            s_table.add_column("Run ID", style="bold cyan")
            s_table.add_column("Model", style="white")
            s_table.add_column("Best mAP", style="bold green")
            s_table.add_column("Disk Usage", style="bold yellow")
            s_table.add_column("Date", style="dim white")

            for e in sorted_exps:
                s_table.add_row(
                    e.get("run_id", "N/A"),
                    e.get("model", "N/A"),
                    str(e.get("best_mAP", "N/A")),
                    format_bytes(e.get("disk_usage", 0)),
                    str(e.get("timestamp", "N/A")),
                )
            self.console.print(s_table)
            self._pause()

    def _action_open_experiment_folder(self) -> None:
        """[2] Open Experiment Folder in native OS explorer."""
        exps = self.manager.get_experiments()
        if not exps:
            self.console.print("No experiments found.")
            self._pause()
            return

        run_id = input(f"Enter Run ID to open [default: {exps[0]['run_id']}]: ").strip() or exps[0]['run_id']
        success, msg = self.manager.open_experiment_folder(run_id)
        self.console.print(f"[bold green]{msg}[/bold green]" if success else f"[bold red]{msg}[/bold red]")
        self._pause()

    def _action_delete_experiment(self) -> None:
        """[3] Delete Experiment (Phase 4 requirement: display metadata, type DELETE)."""
        exps = [e for e in self.manager.get_experiments() if e.get("status") != "Deleted"]
        if not exps:
            self.console.print("No active experiments to delete.")
            self._pause()
            return

        run_id = input("Enter Run ID to delete: ").strip()
        target_exp = next((e for e in exps if e["run_id"] == run_id), None)
        if not target_exp:
            self.console.print(f"[bold red]Experiment '{run_id}' not found.[/bold red]")
            self._pause()
            return

        # Display pre-deletion metadata card required by Phase 4
        confirm_table = Table(title="[bold red] CONFIRM DELETION — EXPERIMENT METADATA [/bold red]", box=ROUNDED, border_style="red")
        confirm_table.add_column("Property", style="bold yellow")
        confirm_table.add_column("Value", style="bold white")
        confirm_table.add_row("Experiment Name", target_exp["run_id"])
        confirm_table.add_row("Dataset", target_exp.get("dataset", "Unknown"))
        confirm_table.add_row("Model", target_exp.get("model", "Unknown"))
        confirm_table.add_row("Epoch", str(target_exp.get("epoch", 0)))
        confirm_table.add_row("Disk Size", format_bytes(target_exp.get("disk_usage", 0)))
        confirm_table.add_row("Creation Date", target_exp.get("timestamp", "Unknown"))
        confirm_table.add_row("Last Modified", target_exp.get("timestamp", "Unknown"))

        self.console.print(confirm_table)
        confirm_in = input("\nType DELETE to continue: ").strip()
        if confirm_in == "DELETE":
            success, msg, reclaimed = self.manager.delete_experiment(run_id, confirmed=True)
            self.console.print(f"[bold green]{msg} Reclaimed: {format_bytes(reclaimed)}[/bold green]" if success else f"[bold red]{msg}[/bold red]")
        else:
            self.console.print("[bold yellow]Deletion cancelled.[/bold yellow]")
        self._pause()

    def _action_delete_multiple(self) -> None:
        """[4] Delete Multiple Experiments (Phase 5 query parser)."""
        self.console.print("[bold cyan]Multi-Select Delete Examples:[/bold cyan]")
        self.console.print("  - Space/Comma IDs: `3 7 12 18` or `exp_001, exp_002`")
        self.console.print("  - Month/Date: `2026-08`")
        self.console.print("  - Status: `failed` or `empty`")
        self.console.print("  - Relative Age: `older than 90 days`")

        query = input("\nEnter multi-select query: ").strip()
        if not query:
            return

        selected = self.manager.deleter.parse_multi_select_query(query)
        if not selected:
            self.console.print("[bold yellow]No experiments matched the query.[/bold yellow]")
            self._pause()
            return

        total_bytes = sum(int(e.get("disk_usage", 0)) for e in selected)
        self.console.print(f"\n[bold red]Selected {len(selected)} experiments for deletion (Reclaimable: {format_bytes(total_bytes)})[/bold red]")
        for e in selected:
            self.console.print(f"  - {e.get('run_id')} ({format_bytes(e.get('disk_usage', 0))})")

        confirm_in = input("\nType DELETE to continue: ").strip()
        if confirm_in == "DELETE":
            count, reclaimed = self.manager.delete_multiple_experiments(query, confirmed=True)
            self.console.print(f"[bold green]Successfully moved {count} experiments to trash. Reclaimed: {format_bytes(reclaimed)}[/bold green]")
        else:
            self.console.print("[bold yellow]Multi-delete cancelled.[/bold yellow]")
        self._pause()

    def _action_delete_by_type(self, type_key: str) -> None:
        """[5] Delete Failed Experiments & [6] Delete Empty Experiments."""
        count, reclaimed = self.manager.delete_multiple_experiments(type_key, confirmed=True)
        self.console.print(f"[bold green]Deleted {count} '{type_key}' experiments. Reclaimed: {format_bytes(reclaimed)}[/bold green]")
        self._pause()

    def _action_delete_old_checkpoints(self) -> None:
        """[7] Delete Old Checkpoints."""
        policy = {"delete_epoch_checkpoints": True}
        res = self.manager.policy_engine.execute_policy(policy, confirmed=True)
        self.console.print(f"[bold green]Deleted {res.get('deleted_files_count', 0)} epoch checkpoints. Reclaimed: {format_bytes(res.get('reclaimed_bytes', 0))}[/bold green]")
        self._pause()

    def _action_delete_asset(self, asset_key: str) -> None:
        """[8] Delete Validation Images, [9] Reports, [10] Exports."""
        policy = {f"delete_{asset_key}": True}
        res = self.manager.policy_engine.execute_policy(policy, confirmed=True)
        self.console.print(f"[bold green]Deleted {res.get('deleted_files_count', 0)} {asset_key}. Reclaimed: {format_bytes(res.get('reclaimed_bytes', 0))}[/bold green]")
        self._pause()

    def _action_archive_experiment(self) -> None:
        """[11] Archive Experiment into ZIP with SHA-256 checksum."""
        exps = [e for e in self.manager.get_experiments() if e.get("status") not in ("Deleted", "Archived")]
        if not exps:
            self.console.print("No active experiments available to archive.")
            self._pause()
            return

        run_id = input(f"Enter Run ID to archive [default: {exps[0]['run_id']}]: ").strip() or exps[0]['run_id']
        success, msg, sha = self.manager.archive_experiment(run_id)
        self.console.print(f"[bold green]{msg}[/bold green]" if success else f"[bold red]{msg}[/bold red]")
        self._pause()

    def _action_restore_archive(self) -> None:
        """[12] Restore Archived Experiment."""
        archives = self.manager.archiver.list_archived_experiments()
        if not archives:
            self.console.print("No archives found in archive/ directory.")
            self._pause()
            return

        table = Table(title="ARCHIVED EXPERIMENTS IN ARCHIVE/", box=ROUNDED, border_style="cyan")
        table.add_column("#", style="bold yellow")
        table.add_column("Filename", style="bold white")
        table.add_column("Size", style="cyan")
        table.add_column("SHA-256", style="dim white")

        for i, a in enumerate(archives):
            table.add_row(str(i + 1), a["filename"], format_bytes(a["size_bytes"]), a["sha256"][:16])
        self.console.print(table)

        fn = input("Select archive filename to restore: ").strip()
        if fn:
            success, msg = self.manager.restore_archived_experiment(fn)
            self.console.print(f"[bold green]{msg}[/bold green]" if success else f"[bold red]{msg}[/bold red]")
        self._pause()

    def _action_compress_experiment(self) -> None:
        """[13] Compress Experiment."""
        exps = self.manager.get_experiments()
        if not exps:
            self.console.print("No experiments found.")
            self._pause()
            return

        run_id = input(f"Enter Run ID to compress [default: {exps[0]['run_id']}]: ").strip() or exps[0]['run_id']
        success, msg, sha = self.manager.archiver.compress_experiment_in_place(run_id)
        self.console.print(f"[bold green]{msg}[/bold green]" if success else f"[bold red]{msg}[/bold red]")
        self._pause()

    def _action_storage_analyzer(self) -> None:
        """[14] Storage Analyzer Dashboard (Phase 2)."""
        stats = self.manager.storage_analysis()

        counts_table = Table(title="[bold green] EXPERIMENT COUNT BREAKDOWN [/bold green]", box=ROUNDED, border_style="green", expand=True)
        counts_table.add_column("Total Exps", style="bold yellow")
        counts_table.add_column("Running", style="bold cyan")
        counts_table.add_column("Completed", style="bold green")
        counts_table.add_column("Failed", style="bold red")
        counts_table.add_column("Interrupted", style="bold red")
        counts_table.add_column("Archived", style="bold blue")
        counts_table.add_column("Deleted (Trash)", style="dim white")

        c = stats["counts"]
        counts_table.add_row(
            str(c["total"]), str(c["running"]), str(c["completed"]),
            str(c["failed"]), str(c["interrupted"]), str(c["archived"]), str(c["deleted"])
        )

        size_table = Table(title="[bold green] DISK USAGE BREAKDOWN BY ASSET TYPE [/bold green]", box=ROUNDED, border_style="green", expand=True)
        size_table.add_column("Asset Type", style="bold white")
        size_table.add_column("Disk Space", style="bold yellow")

        size_table.add_row("Total Workspace Usage", stats["disk_usage_total_str"])
        size_table.add_row("Checkpoints Size (.pt, .onnx)", stats["checkpoints_size_str"])
        size_table.add_row("Validation & Overlay Images", stats["validation_images_size_str"])
        size_table.add_row("Exports Folder", stats["exports_size_str"])
        size_table.add_row("Reports Folder", stats["reports_size_str"])
        size_table.add_row("System & Execution Logs", stats["logs_size_str"])
        size_table.add_row("PyCache & Temp Caches", stats["cache_size_str"])
        size_table.add_row("[bold cyan]Estimated Recoverable Space[/bold cyan]", f"[bold green]{stats['recoverable_space_str']}[/bold green]")

        metrics_table = Table(title="[bold green] EXPERIMENT METRICS & EXTREMES [/bold green]", box=ROUNDED, border_style="green", expand=True)
        metrics_table.add_column("Metric Target", style="bold white")
        metrics_table.add_column("Details", style="bold cyan")

        metrics_table.add_row("Average Experiment Size", stats["avg_exp_size_str"])
        metrics_table.add_row("Largest Experiment", f"{stats['largest_exp']['name']} ({stats['largest_exp']['size_str']})")
        metrics_table.add_row("Oldest Experiment", f"{stats['oldest_exp']['name']} ({stats['oldest_exp']['date']})")
        metrics_table.add_row("Newest Experiment", f"{stats['newest_exp']['name']} ({stats['newest_exp']['date']})")

        self.console.print(counts_table)
        self.console.print(size_table)
        self.console.print(metrics_table)
        self._pause()

    def _action_workspace_health(self) -> None:
        """[15] Workspace Health Diagnostics (Phase 10)."""
        health = self.manager.inspect_health()

        status_color = "green" if health["overall_status"] == "PASS" else ("yellow" if health["overall_status"] == "WARNING" else "red")
        
        table = Table(title=f"[bold {status_color}] WORKSPACE HEALTH DIAGNOSTIC RESULTS (STATUS: {health['overall_status']}) [/bold {status_color}]", box=ROUNDED, border_style=status_color, expand=True)
        table.add_column("Check Target", style="bold white")
        table.add_column("Discovered Issues", style="bold yellow")
        table.add_column("Status Result", style=f"bold {status_color}")

        checks = [
            ("Broken Folders", health["broken_folders"]),
            ("Missing Checkpoints", health["missing_checkpoints"]),
            ("Missing Reports", health["missing_reports"]),
            ("Missing Metrics", health["missing_metrics"]),
            ("Corrupt JSON", health["corrupt_json"]),
            ("Corrupt CSV", health["corrupt_csv"]),
            ("Corrupt HTML", health["corrupt_html"]),
            ("Broken Symlinks", health["broken_symlinks"]),
            ("Duplicate Checkpoints", health["duplicate_checkpoints"]),
            ("Orphan Folders", health["orphan_folders"]),
            ("Orphan Logs", health["orphan_logs"]),
            ("Orphan Exports", health["orphan_exports"]),
            ("Zero-Byte Files", health["zero_byte_files"]),
        ]

        for target, count in checks:
            st = "[bold green]PASS[/bold green]" if count == 0 else "[bold yellow]WARNING[/bold yellow]"
            table.add_row(target, str(count), st)

        self.console.print(table)
        self._pause()

    def _action_cleanup_cache(self) -> None:
        """[16] Cleanup Cache."""
        policy = {"delete_cache": True}
        res = self.manager.vacuum_workspace()
        self.console.print(f"[bold green]Cleaned PyCache and temporary files. Reclaimed: {format_bytes(res.get('bytes_reclaimed', 0))}[/bold green]")
        self._pause()

    def _action_vacuum_workspace(self) -> None:
        """[17] Vacuum Workspace (Phase 11)."""
        self.console.print("[bold cyan]Executing Vacuum Engine... Purging junk files & rebuilding databases...[/bold cyan]")
        res = self.manager.vacuum_workspace()

        self.console.print(f"\n[bold green]VACUUM COMPLETED SUCCESSFULLY![/bold green]")
        self.console.print(f"  - Purged Items Count: {res.get('items_removed_count', 0)}")
        self.console.print(f"  - Reclaimed Space: {format_bytes(res.get('bytes_reclaimed', 0))}")
        self.console.print(f"  - Rebuilt Database Index Runs: {res.get('rebuilt_experiments_count', 0)}")
        self.console.print(f"  - Rebuilt Files: runs/index.json, runs/history.csv, runs/latest.json, runs/workspace.db")
        self._pause()

    def _action_auto_cleanup_policy(self) -> None:
        """[18] Auto Cleanup Policy (Phase 12)."""
        self.console.print("[bold cyan]Auto Cleanup Policy Configuration:[/bold cyan]")
        self.console.print("  [1] Keep last 10 experiments")
        self.console.print("  [2] Delete experiments older than 30 days")
        self.console.print("  [3] Keep only best.pt (Delete epoch checkpoints)")
        self.console.print("  [4] Delete validation images & reports")

        choice = input("Select policy to preview [1-4]: ").strip()
        policy = {}
        if choice == "1":
            policy = {"keep_last_n": 10}
        elif choice == "2":
            policy = {"older_than_days": 30}
        elif choice == "3":
            policy = {"delete_epoch_checkpoints": True}
        elif choice == "4":
            policy = {"delete_validation_images": True, "delete_reports": True}
        else:
            return

        preview = self.manager.policy_engine.preview_policy(policy)
        self.console.print(f"\n[bold yellow]POLICY PREVIEW:[/bold yellow]")
        self.console.print(f"  - Targeted Experiments: {preview['experiments_to_delete_count']}")
        self.console.print(f"  - Targeted Files: {preview['files_to_delete_count']}")
        self.console.print(f"  - Estimated Reclaimed Space: {format_bytes(preview['estimated_reclaimed_bytes'])}")

        ans = input("\nExecute Policy Now? (Y/N): ").strip().lower()
        if ans == "y":
            res = self.manager.policy_engine.execute_policy(policy, confirmed=True)
            self.console.print(f"[bold green]Policy executed successfully. Reclaimed: {format_bytes(res.get('reclaimed_bytes', 0))}[/bold green]")
        self._pause()

    def _action_recover_deleted(self) -> None:
        """[19] Recover Deleted Experiments (Phase 13)."""
        deleted = self.manager.deleter.list_deleted_experiments()
        if not deleted:
            self.console.print("[bold yellow]No deleted experiments in runs/.trash/[/bold yellow]")
            self._pause()
            return

        table = Table(title="[bold green] RECOVERABLE EXPERIMENTS IN TRASH [/bold green]", box=ROUNDED, border_style="green", expand=True)
        table.add_column("#", style="bold yellow")
        table.add_column("Run ID", style="bold cyan")
        table.add_column("Deleted Date", style="dim white")
        table.add_column("Original Location", style="white")
        table.add_column("Disk Size", style="bold yellow")

        for i, d in enumerate(deleted):
            table.add_row(str(i + 1), d["run_id"], d["deleted_date"], d["original_location"], format_bytes(d["disk_size"]))

        self.console.print(table)
        sel = input("Select number or Trash Folder Name to restore: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(deleted):
            target_folder = deleted[int(sel) - 1]["trash_folder"]
        else:
            target_folder = sel

        if target_folder:
            success, msg = self.manager.recover_deleted_experiment(target_folder)
            self.console.print(f"[bold green]{msg}[/bold green]" if success else f"[bold red]{msg}[/bold red]")
        self._pause()

    def _action_search_experiments(self) -> None:
        """[20] Search Experiments (Phase 7)."""
        query = input("Enter search query (Run ID, Dataset, Model, Date, Status, mAP, Epoch, Commit): ").strip()
        if not query:
            return

        results = self.manager.search_experiments(query)
        self.console.print(f"\n[bold green]Discovered {len(results)} matching experiments:[/bold green]")

        table = Table(box=ROUNDED, border_style="cyan", expand=True)
        table.add_column("Run ID", style="bold cyan")
        table.add_column("Model", style="white")
        table.add_column("Dataset", style="white")
        table.add_column("Best mAP", style="bold green")
        table.add_column("Status", style="bold yellow")
        table.add_column("Date", style="dim white")

        for e in results:
            table.add_row(e["run_id"], e["model"], e["dataset"], str(e["best_mAP"]), e["status"], str(e["timestamp"]))

        self.console.print(table)
        self._pause()

    def render_terminal_completion_screen(self, summary_data: Dict[str, Any]) -> None:
        """Render Phase 14 Terminal Completion Screen after training."""
        run_id = summary_data.get("run_id", "exp_run")
        exp_folder = summary_data.get("output_dir", f"runs/{run_id}")
        exp_path = Path(exp_folder).resolve()

        val_folder = exp_path / "prediction_overlays"
        ckpt_folder = exp_path / "weights" if (exp_path / "weights").exists() else exp_path
        reports_folder = self.root_dir / "reports"
        export_folder = self.root_dir / "exports"

        disk_size = sum(f.stat().st_size for f in exp_path.glob("**/*") if f.is_file()) if exp_path.exists() else 0

        comp_table = Table(title="[bold green] AETHEL TRAINING SESSION TERMINAL COMPLETION SCREEN [/bold green]", box=ROUNDED, border_style="green", expand=True)
        comp_table.add_column("Target Workspace Location", style="bold cyan")
        comp_table.add_column("Path / Details", style="bold white")

        comp_table.add_row("Run ID", run_id)
        comp_table.add_row("Experiment Folder", str(exp_path))
        comp_table.add_row("Disk Usage", format_bytes(disk_size))
        comp_table.add_row("Validation Folder", str(val_folder))
        comp_table.add_row("Checkpoint Folder", str(ckpt_folder))
        comp_table.add_row("Reports Folder", str(reports_folder))
        comp_table.add_row("Export Folder", str(export_folder))

        self.console.print(comp_table)

        # Interactive prompts as per Phase 14
        ans_open = input("\nOpen Experiment Folder in Explorer? (Y/N) [N]: ").strip().lower()
        if ans_open == "y":
            open_folder_in_explorer(exp_path)

        ans_arch = input("Archive Experiment to ZIP? (Y/N) [N]: ").strip().lower()
        if ans_arch == "y":
            success, msg, sha = self.manager.archive_experiment(run_id)
            self.console.print(f"[bold green]{msg}[/bold green]" if success else f"[bold red]{msg}[/bold red]")

        ans_del = input("Delete Experiment to Trash? (Y/N) [N]: ").strip().lower()
        if ans_del == "y":
            confirm = input("Type DELETE to confirm: ").strip()
            if confirm == "DELETE":
                success, msg, reclaimed = self.manager.delete_experiment(run_id, confirmed=True)
                self.console.print(f"[bold green]{msg}[/bold green]" if success else f"[bold red]{msg}[/bold red]")
