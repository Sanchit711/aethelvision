"""AETHEL Auto Cleanup Policy Engine Module (Phase 12).

Applies configurable auto cleanup policies (keep last N runs, delete older than N days,
keep only best.pt checkpoint, delete validation images/reports/exports/logs),
previews reclaimed space, and executes policies safely.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aethel.utils.logger import get_logger
from aethel.workspace.database import ExperimentDatabase
from aethel.workspace.deleter import RecoverableDeleter
from aethel.workspace.logger import WorkspaceLogger

logger = get_logger("aethel.workspace.policy")


class AutoCleanupPolicyEngine:
    """Configurable Auto Cleanup Policy Engine."""

    def __init__(
        self,
        root_dir: Path = Path("."),
        db: Optional[ExperimentDatabase] = None,
        deleter: Optional[RecoverableDeleter] = None,
        ws_logger: Optional[WorkspaceLogger] = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.db = db or ExperimentDatabase(root_dir=self.root_dir)
        self.deleter = deleter or RecoverableDeleter(root_dir=self.root_dir, db=self.db)
        self.ws_logger = ws_logger or WorkspaceLogger(root_dir=self.root_dir)

    def preview_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Preview items that would be affected by the policy and calculate estimated reclaimed space."""
        exps = [e for e in self.db.get_all_experiments() if e.get("status") not in ("Deleted", "Archived")]

        target_experiments_to_delete: List[Dict[str, Any]] = []
        target_files_to_delete: List[Path] = []
        estimated_bytes = 0

        # 1. Keep last N experiments policy
        keep_last_n = policy.get("keep_last_n")
        if keep_last_n is not None and keep_last_n > 0:
            if len(exps) > keep_last_n:
                # Sort by timestamp ascending (oldest first)
                sorted_exps = sorted(exps, key=lambda x: str(x.get("timestamp", "")))
                to_delete = sorted_exps[: len(exps) - keep_last_n]
                target_experiments_to_delete.extend(to_delete)

        # 2. Delete older than N days policy
        older_than_days = policy.get("older_than_days")
        if older_than_days is not None and older_than_days > 0:
            cutoff = time.time() - (older_than_days * 86400)
            for e in exps:
                folder = Path(e["folder"])
                if folder.exists() and folder.stat().st_mtime < cutoff:
                    if e not in target_experiments_to_delete:
                        target_experiments_to_delete.append(e)

        for e in target_experiments_to_delete:
            estimated_bytes += int(e.get("disk_usage", 0))

        # 3. Keep only best.pt (delete epoch checkpoints)
        if policy.get("delete_epoch_checkpoints", False):
            for e in exps:
                folder = Path(e["folder"])
                if folder.exists():
                    for ckpt in folder.glob("**/epoch_*.pt"):
                        if ckpt.name != "best.pt":
                            target_files_to_delete.append(ckpt)
                            estimated_bytes += ckpt.stat().st_size

        # 4. Delete validation images
        if policy.get("delete_validation_images", False):
            for e in exps:
                folder = Path(e["folder"])
                if folder.exists():
                    for img in folder.glob("**/*.png"):
                        target_files_to_delete.append(img)
                        estimated_bytes += img.stat().st_size

        # 5. Delete reports
        if policy.get("delete_reports", False):
            for e in exps:
                folder = Path(e["folder"])
                if folder.exists():
                    for rep in folder.glob("**/*.html"):
                        target_files_to_delete.append(rep)
                        estimated_bytes += rep.stat().st_size

        # 6. Delete exports
        if policy.get("delete_exports", False):
            exports_dir = self.root_dir / "exports"
            if exports_dir.exists():
                for exp_f in exports_dir.glob("*.*"):
                    target_files_to_delete.append(exp_f)
                    estimated_bytes += exp_f.stat().st_size

        # 7. Delete logs
        if policy.get("delete_logs", False):
            logs_dir = self.root_dir / "logs"
            if logs_dir.exists():
                for log_f in logs_dir.glob("*.log"):
                    if log_f.name not in ("workspace.log", "aethel.log"):
                        target_files_to_delete.append(log_f)
                        estimated_bytes += log_f.stat().st_size

        return {
            "experiments_to_delete_count": len(target_experiments_to_delete),
            "files_to_delete_count": len(target_files_to_delete),
            "target_experiments": [e.get("run_id") for e in target_experiments_to_delete],
            "estimated_reclaimed_bytes": estimated_bytes,
        }

    def execute_policy(self, policy: Dict[str, Any], confirmed: bool = False) -> Dict[str, Any]:
        """Execute cleanup policy after confirmation."""
        if not confirmed:
            return {"status": "FAILED", "reason": "Policy execution requires confirmation."}

        t0 = time.time()
        preview = self.preview_policy(policy)

        deleted_exps = 0
        deleted_files = 0
        reclaimed_bytes = 0

        # Delete targeted experiments via RecoverableDeleter
        for run_id in preview["target_experiments"]:
            success, msg, size = self.deleter.delete_experiment(run_id, confirmed=True)
            if success:
                deleted_exps += 1
                reclaimed_bytes += size

        # Delete individual target files
        exps = [e for e in self.db.get_all_experiments() if e.get("status") not in ("Deleted", "Archived")]

        if policy.get("delete_epoch_checkpoints", False):
            for e in exps:
                folder = Path(e["folder"])
                if folder.exists():
                    for ckpt in list(folder.glob("**/epoch_*.pt")):
                        if ckpt.name != "best.pt":
                            try:
                                size = ckpt.stat().st_size
                                ckpt.unlink()
                                deleted_files += 1
                                reclaimed_bytes += size
                            except Exception:
                                pass

        if policy.get("delete_validation_images", False):
            for e in exps:
                folder = Path(e["folder"])
                if folder.exists():
                    for img in list(folder.glob("**/*.png")):
                        try:
                            size = img.stat().st_size
                            img.unlink()
                            deleted_files += 1
                            reclaimed_bytes += size
                        except Exception:
                            pass

        if policy.get("delete_reports", False):
            for e in exps:
                folder = Path(e["folder"])
                if folder.exists():
                    for rep in list(folder.glob("**/*.html")):
                        try:
                            size = rep.stat().st_size
                            rep.unlink()
                            deleted_files += 1
                            reclaimed_bytes += size
                        except Exception:
                            pass

        if policy.get("delete_exports", False):
            exports_dir = self.root_dir / "exports"
            if exports_dir.exists():
                for exp_f in list(exports_dir.glob("*.*")):
                    try:
                        size = exp_f.stat().st_size
                        exp_f.unlink()
                        deleted_files += 1
                        reclaimed_bytes += size
                    except Exception:
                        pass

        if policy.get("delete_logs", False):
            logs_dir = self.root_dir / "logs"
            if logs_dir.exists():
                for log_f in list(logs_dir.glob("*.log")):
                    if log_f.name not in ("workspace.log", "aethel.log"):
                        try:
                            size = log_f.stat().st_size
                            log_f.unlink()
                            deleted_files += 1
                            reclaimed_bytes += size
                        except Exception:
                            pass

        self.db.rebuild_index()

        t_elapsed_ms = (time.time() - t0) * 1000
        self.ws_logger.log(
            action="AUTO_CLEANUP_POLICY",
            target="workspace",
            status="SUCCESS",
            details={
                "deleted_exps": deleted_exps,
                "deleted_files": deleted_files,
            },
            execution_time_ms=t_elapsed_ms,
            disk_usage_bytes=reclaimed_bytes,
        )

        return {
            "status": "SUCCESS",
            "deleted_experiments_count": deleted_exps,
            "deleted_files_count": deleted_files,
            "reclaimed_bytes": reclaimed_bytes,
            "execution_time_ms": t_elapsed_ms,
        }
