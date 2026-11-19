"""AETHEL Enterprise Workspace Manager Master Module.

Orchestrates all workspace lifecycle operations: database indexing, recoverable deletion,
archiving, storage analysis, health diagnostics, vacuum engine, auto-cleanup policies,
cross-platform folder explorer launching, and report generation.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from aethel.utils.logger import get_logger
from aethel.workspace.analyzer import StorageAnalyzer
from aethel.workspace.archiver import WorkspaceArchiver
from aethel.workspace.database import ExperimentDatabase
from aethel.workspace.deleter import RecoverableDeleter
from aethel.workspace.health import WorkspaceHealthChecker
from aethel.workspace.logger import WorkspaceLogger
from aethel.workspace.policy import AutoCleanupPolicyEngine
from aethel.workspace.reports import WorkspaceReportGenerator
from aethel.workspace.vacuum import VacuumEngine

logger = get_logger("aethel.workspace.manager")


def open_folder_in_explorer(folder_path: Union[str, Path]) -> Tuple[bool, str]:
    """Cross-platform folder opening (Windows: explorer.exe, Linux/Jetson: xdg-open, macOS: open)."""
    p = Path(folder_path).resolve()
    if not p.exists():
        return False, f"Path '{p}' does not exist on disk."

    abs_path_str = str(p)
    logger.info(f"Opening folder in native OS explorer: {abs_path_str}")

    try:
        sys_name = platform.system().lower()
        if "windows" in sys_name or os.name == "nt":
            if hasattr(os, "startfile"):
                os.startfile(abs_path_str)
            else:
                subprocess.Popen(["explorer.exe", abs_path_str])
        elif "darwin" in sys_name:
            subprocess.Popen(["open", abs_path_str])
        else:  # Linux, Jetson, WSL
            subprocess.Popen(["xdg-open", abs_path_str])
        return True, f"Opened explorer at '{abs_path_str}'."
    except Exception as e:
        logger.warning(f"Could not launch native file explorer: {e}")
        return False, f"Folder path is '{abs_path_str}' (OS Explorer launch suppressed: {e})"


class WorkspaceManager:
    """Master Workspace Manager Engine for AETHEL Enterprise Platform."""

    def __init__(self, root_dir: Path = Path(".")) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.db = ExperimentDatabase(root_dir=self.root_dir)
        self.analyzer = StorageAnalyzer(root_dir=self.root_dir, db=self.db)
        self.deleter = RecoverableDeleter(root_dir=self.root_dir, db=self.db)
        self.archiver = WorkspaceArchiver(root_dir=self.root_dir, db=self.db)
        self.health_checker = WorkspaceHealthChecker(root_dir=self.root_dir, db=self.db)
        self.vacuum_engine = VacuumEngine(root_dir=self.root_dir, db=self.db)
        self.policy_engine = AutoCleanupPolicyEngine(root_dir=self.root_dir, db=self.db, deleter=self.deleter)
        self.reporter = WorkspaceReportGenerator(root_dir=self.root_dir)
        self.ws_logger = WorkspaceLogger(root_dir=self.root_dir)

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Return list of all experiments from database."""
        return self.db.get_all_experiments()

    def search_experiments(self, query: str) -> List[Dict[str, Any]]:
        """Search experiments by run ID, dataset, model, date, status, mAP, epoch, loss, notes, commit."""
        exps = self.db.get_all_experiments()
        if not query.strip():
            return exps

        q = query.strip().lower()
        matched = []
        for e in exps:
            fields = [
                str(e.get("run_id", "")),
                str(e.get("model", "")),
                str(e.get("dataset", "")),
                str(e.get("timestamp", "")),
                str(e.get("status", "")),
                str(e.get("best_mAP", "")),
                str(e.get("epoch", "")),
                str(e.get("loss", "")),
                str(e.get("user_notes", "")),
                str(e.get("git_commit", "")),
            ]
            if any(q in f.lower() for f in fields):
                matched.append(e)
        return matched

    def sort_experiments(self, exps: List[Dict[str, Any]], sort_key: str = "date", descending: bool = True) -> List[Dict[str, Any]]:
        """Sort experiment records by key."""
        key_map = {
            "date": lambda x: str(x.get("timestamp", "")),
            "size": lambda x: int(x.get("disk_usage", 0)),
            "map": lambda x: float(x.get("best_mAP", 0.0)),
            "loss": lambda x: float(x.get("loss", 0.0)),
            "epoch": lambda x: int(x.get("epoch", 0)),
            "dataset": lambda x: str(x.get("dataset", "")),
            "model": lambda x: str(x.get("model", "")),
            "duration": lambda x: float(x.get("duration", 0.0)),
            "disk_usage": lambda x: int(x.get("disk_usage", 0)),
        }
        fn = key_map.get(sort_key.lower(), key_map["date"])
        return sorted(exps, key=fn, reverse=descending)

    def delete_experiment(self, run_id: str, confirmed: bool = False) -> Tuple[bool, str, int]:
        """Delete an experiment (move to runs/.trash/)."""
        return self.deleter.delete_experiment(run_id, confirmed=confirmed)

    def delete_multiple_experiments(self, query: str, confirmed: bool = False) -> Tuple[int, int]:
        """Delete multiple experiments matching query."""
        selected = self.deleter.parse_multi_select_query(query)
        deleted_count = 0
        total_reclaimed = 0
        if confirmed:
            for e in selected:
                success, msg, size = self.deleter.delete_experiment(e["run_id"], confirmed=True)
                if success:
                    deleted_count += 1
                    total_reclaimed += size
        return deleted_count, total_reclaimed

    def archive_experiment(self, run_id: str) -> Tuple[bool, str, str]:
        """Archive experiment into a ZIP archive."""
        return self.archiver.archive_experiment(run_id)

    def restore_archived_experiment(self, zip_filename: str) -> Tuple[bool, str]:
        """Restore an experiment from archive ZIP."""
        return self.archiver.restore_archived_experiment(zip_filename)

    def recover_deleted_experiment(self, trash_folder_name: str) -> Tuple[bool, str]:
        """Recover a deleted experiment from trash."""
        return self.deleter.restore_deleted_experiment(trash_folder_name)

    def vacuum_workspace(self) -> Dict[str, Any]:
        """Vacuum workspace and rebuild database."""
        res = self.vacuum_engine.vacuum()
        self.reporter.generate_all_reports()
        return res

    def inspect_health(self) -> Dict[str, Any]:
        """Perform workspace health inspection."""
        return self.health_checker.inspect_health()

    def storage_analysis(self) -> Dict[str, Any]:
        """Return workspace storage analysis."""
        return self.analyzer.analyze()

    def open_experiment_folder(self, run_id_or_path: str) -> Tuple[bool, str]:
        """Open experiment folder in native file explorer."""
        exps = self.get_experiments()
        folder_p = None
        for e in exps:
            if e["run_id"] == run_id_or_path or e["folder"] == run_id_or_path:
                folder_p = e["folder"]
                break
        if not folder_p:
            folder_p = run_id_or_path
        return open_folder_in_explorer(folder_p)

    def generate_reports(self) -> Dict[str, str]:
        """Generate all 5 workspace reports."""
        return self.reporter.generate_all_reports()
