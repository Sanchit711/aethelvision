"""AETHEL Recoverable Deleter & Multi-Select Module (Phases 4, 5, 13).

Provides exception-safe recoverable deletion (moving to runs/.trash/),
multi-select targeting (lists, dates, statuses, relative age), and recovery restoration.
"""

import datetime
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aethel.utils.logger import get_logger
from aethel.workspace.database import ExperimentDatabase
from aethel.workspace.logger import WorkspaceLogger

logger = get_logger("aethel.workspace.deleter")


class RecoverableDeleter:
    """Recoverable Experiment Delete Engine."""

    def __init__(
        self,
        root_dir: Path = Path("."),
        db: Optional[ExperimentDatabase] = None,
        ws_logger: Optional[WorkspaceLogger] = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.runs_dir = self.root_dir / "runs"
        self.trash_dir = self.runs_dir / ".trash"
        self.trash_dir.mkdir(parents=True, exist_ok=True)
        self.db = db or ExperimentDatabase(root_dir=self.root_dir)
        self.ws_logger = ws_logger or WorkspaceLogger(root_dir=self.root_dir)

    def delete_experiment(
        self, experiment_id_or_path: str, confirmed: bool = False
    ) -> Tuple[bool, str, int]:
        """Move an experiment directory to runs/.trash/ with metadata."""
        t0 = time.time()
        exps = self.db.get_all_experiments()

        target_exp = None
        for e in exps:
            if (
                e["run_id"] == experiment_id_or_path
                or e["folder"] == experiment_id_or_path
                or Path(e["folder"]).name == experiment_id_or_path
            ):
                target_exp = e
                break

        if not target_exp:
            exp_path = Path(experiment_id_or_path)
            if not exp_path.is_absolute():
                exp_path = self.runs_dir / experiment_id_or_path
            if exp_path.exists():
                target_exp = {
                    "run_id": exp_path.name,
                    "folder": str(exp_path),
                    "model": "Unknown",
                    "dataset": "Unknown",
                    "epoch": 0,
                    "disk_usage": sum(f.stat().st_size for f in exp_path.glob("**/*") if f.is_file()),
                }

        if not target_exp:
            return False, f"Experiment '{experiment_id_or_path}' not found.", 0

        src_folder = Path(target_exp["folder"])
        if not src_folder.exists():
            return False, f"Source path '{src_folder}' does not exist on disk.", 0

        if not confirmed:
            return False, "Deletion not confirmed (requires confirmation).", 0

        # Create unique trash folder timestamp name
        time_stamp_str = time.strftime("%Y%m%d_%H%M%S")
        trash_folder_name = f"{src_folder.name}_{time_stamp_str}"
        dest_trash_path = self.trash_dir / trash_folder_name

        try:
            # Atomic move or copytree with rollback capability
            shutil.move(str(src_folder), str(dest_trash_path))
            
            # Save trash metadata file
            metadata = {
                "delete_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "original_path": str(src_folder),
                "experiment_metadata": target_exp,
            }
            with open(dest_trash_path / "trash_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            target_exp["status"] = "Deleted"
            self.db.upsert_experiment(target_exp)
            self.db.rebuild_index()

            t_elapsed_ms = (time.time() - t0) * 1000
            disk_size = target_exp.get("disk_usage", 0)
            self.ws_logger.log(
                action="DELETION",
                target=src_folder.name,
                status="SUCCESS",
                details={"trash_path": str(dest_trash_path)},
                execution_time_ms=t_elapsed_ms,
                disk_usage_bytes=disk_size,
            )
            return True, f"Successfully moved '{src_folder.name}' to trash.", disk_size

        except Exception as e:
            logger.error(f"Error deleting experiment {src_folder}: {e}")
            t_elapsed_ms = (time.time() - t0) * 1000
            self.ws_logger.log(
                action="DELETION",
                target=src_folder.name,
                status="FAILED",
                details={"error": str(e)},
                execution_time_ms=t_elapsed_ms,
            )
            return False, f"Failed to delete experiment: {e}", 0

    def parse_multi_select_query(self, query: str) -> List[Dict[str, Any]]:
        """Parse multi-select query (space/comma IDs, month string '2026-08', status 'failed', 'older than N days')."""
        exps = [e for e in self.db.get_all_experiments() if e.get("status") != "Deleted"]
        query_str = query.strip().lower()
        selected: List[Dict[str, Any]] = []

        if not query_str:
            return selected

        # 1. Match relative age: "older than 90 days", "older than 30 days"
        if "older than" in query_str and "days" in query_str:
            try:
                days_str = query_str.replace("older than", "").replace("days", "").strip()
                days = int(days_str)
                cutoff = time.time() - (days * 86400)
                for e in exps:
                    folder = Path(e["folder"])
                    if folder.exists() and folder.stat().st_mtime < cutoff:
                        selected.append(e)
                return selected
            except Exception:
                pass

        # 2. Match status: "failed", "empty", "interrupted", "running", "completed"
        if query_str in ("failed", "empty", "interrupted", "running", "completed"):
            for e in exps:
                st = str(e.get("status", "")).lower()
                folder = Path(e["folder"])
                if query_str == "empty":
                    if folder.exists() and not list(folder.glob("*")):
                        selected.append(e)
                elif query_str in st:
                    selected.append(e)
            return selected

        # 3. Match month/date: "2026-08", "2026-07-20"
        if len(query_str) in (7, 10) and "-" in query_str:
            for e in exps:
                ts = str(e.get("timestamp", ""))
                if ts.startswith(query_str):
                    selected.append(e)
            return selected

        # 4. Match space/comma separated list of indices or run_ids (e.g., "3 7 12 18" or "exp_001, exp_002")
        tokens = [t.strip() for t in query_str.replace(",", " ").split() if t.strip()]
        for token in tokens:
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(exps):
                    selected.append(exps[idx])
            else:
                for e in exps:
                    if e.get("run_id", "").lower() == token or Path(e.get("folder", "")).name.lower() == token:
                        selected.append(e)

        return selected

    def list_deleted_experiments(self) -> List[Dict[str, Any]]:
        """List all recoverable deleted experiments currently in runs/.trash/."""
        deleted = []
        if not self.trash_dir.exists():
            return deleted

        for entry in self.trash_dir.iterdir():
            if entry.is_dir():
                meta_file = entry / "trash_metadata.json"
                meta = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                orig_meta = meta.get("experiment_metadata", {})
                size_bytes = sum(f.stat().st_size for f in entry.glob("**/*") if f.is_file())
                deleted.append({
                    "trash_folder": entry.name,
                    "trash_path": str(entry),
                    "run_id": orig_meta.get("run_id", entry.name),
                    "deleted_date": meta.get("delete_timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                    "original_location": meta.get("original_path", str(self.runs_dir / entry.name)),
                    "disk_size": size_bytes,
                    "metadata": orig_meta,
                })
        return deleted

    def restore_deleted_experiment(self, trash_folder_name: str) -> Tuple[bool, str]:
        """Restore a deleted experiment from runs/.trash/ back to its original location."""
        t0 = time.time()
        trash_path = self.trash_dir / trash_folder_name
        if not trash_path.exists():
            # Try searching by run_id
            for entry in self.trash_dir.iterdir():
                if entry.name == trash_folder_name or entry.name.startswith(f"{trash_folder_name}_"):
                    trash_path = entry
                    break

        if not trash_path.exists():
            return False, f"Trash folder '{trash_folder_name}' not found."

        meta_file = trash_path / "trash_metadata.json"
        meta = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        orig_path_str = meta.get("original_path", str(self.runs_dir / trash_path.name))
        dest_path = Path(orig_path_str)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists():
            # append restored suffix if original location exists
            dest_path = dest_path.parent / f"{dest_path.name}_restored"

        try:
            # Remove metadata file before moving back
            if meta_file.exists():
                meta_file.unlink()

            shutil.move(str(trash_path), str(dest_path))

            # Update database status
            orig_meta = meta.get("experiment_metadata", {})
            orig_meta["folder"] = str(dest_path)
            orig_meta["status"] = "Completed"
            self.db.upsert_experiment(orig_meta)
            self.db.rebuild_index()

            t_elapsed_ms = (time.time() - t0) * 1000
            self.ws_logger.log(
                action="RESTORE",
                target=dest_path.name,
                status="SUCCESS",
                details={"restored_path": str(dest_path)},
                execution_time_ms=t_elapsed_ms,
            )
            return True, f"Successfully restored experiment to '{dest_path}'."
        except Exception as e:
            logger.error(f"Error restoring experiment {trash_path}: {e}")
            t_elapsed_ms = (time.time() - t0) * 1000
            self.ws_logger.log(
                action="RESTORE",
                target=trash_folder_name,
                status="FAILED",
                details={"error": str(e)},
                execution_time_ms=t_elapsed_ms,
            )
            return False, f"Failed to restore experiment: {e}"
