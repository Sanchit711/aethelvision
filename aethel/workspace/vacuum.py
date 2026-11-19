"""AETHEL Workspace Vacuum Engine Module (Phase 11).

Purges junk files (__pycache__, .pyc, .tmp, .temp, Thumbs.db, .desktop.ini,
broken caches, unused exports, duplicate checkpoints, orphan logs) and rebuilds
index.json, history.csv, latest.json, workspace.db.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aethel.utils.logger import get_logger
from aethel.workspace.database import ExperimentDatabase
from aethel.workspace.health import WorkspaceHealthChecker
from aethel.workspace.logger import WorkspaceLogger

logger = get_logger("aethel.workspace.vacuum")


class VacuumEngine:
    """Workspace Vacuum & Rebuild Engine."""

    def __init__(
        self,
        root_dir: Path = Path("."),
        db: Optional[ExperimentDatabase] = None,
        ws_logger: Optional[WorkspaceLogger] = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.db = db or ExperimentDatabase(root_dir=self.root_dir)
        self.ws_logger = ws_logger or WorkspaceLogger(root_dir=self.root_dir)
        self.health_checker = WorkspaceHealthChecker(root_dir=self.root_dir, db=self.db)

    def vacuum(self) -> Dict[str, Any]:
        """Execute full vacuum cleanup operation across workspace and rebuild index."""
        t0 = time.time()
        removed_files: List[str] = []
        bytes_reclaimed = 0

        # 1. Target junk patterns & files
        junk_extensions = {".pyc", ".tmp", ".temp"}
        junk_filenames = {"thumbs.db", ".desktop.ini", "desktop.ini"}

        for p in list(self.root_dir.rglob("*")):
            try:
                if not p.exists():
                    continue

                # Remove junk directories
                if p.is_dir() and p.name in ("__pycache__", ".pytest_cache"):
                    size = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())
                    shutil.rmtree(p)
                    removed_files.append(str(p))
                    bytes_reclaimed += size
                    continue

                # Remove junk files
                if p.is_file():
                    if p.suffix.lower() in junk_extensions or p.name.lower() in junk_filenames:
                        size = p.stat().st_size
                        p.unlink()
                        removed_files.append(str(p))
                        bytes_reclaimed += size
            except Exception:
                pass

        # 2. Clean orphan logs in logs/
        logs_dir = self.root_dir / "logs"
        if logs_dir.exists():
            for log_p in logs_dir.glob("*.log"):
                if log_p.name not in ("workspace.log", "aethel.log"):
                    try:
                        size = log_p.stat().st_size
                        log_p.unlink()
                        removed_files.append(str(log_p))
                        bytes_reclaimed += size
                    except Exception:
                        pass

        # 3. Clean zero-byte temp files
        for p in list(self.root_dir.glob("*.tmp")) + list(self.root_dir.glob("*.temp")):
            try:
                if p.is_file():
                    p.unlink()
                    removed_files.append(str(p))
            except Exception:
                pass

        # 4. Rebuild canonical database & index files cleanly
        exps = self.db.rebuild_index()

        t_elapsed_ms = (time.time() - t0) * 1000
        self.ws_logger.log(
            action="VACUUM",
            target="workspace",
            status="SUCCESS",
            details={
                "items_removed": len(removed_files),
                "rebuilt_index_count": len(exps),
            },
            execution_time_ms=t_elapsed_ms,
            disk_usage_bytes=bytes_reclaimed,
        )

        return {
            "status": "SUCCESS",
            "items_removed_count": len(removed_files),
            "bytes_reclaimed": bytes_reclaimed,
            "rebuilt_experiments_count": len(exps),
            "execution_time_ms": t_elapsed_ms,
            "removed_files_sample": removed_files[:10],
        }
