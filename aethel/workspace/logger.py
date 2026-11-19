"""AETHEL Workspace Logger Module.

Maintains workspace.log recording all workspace operations: creation, deletion,
restore, archive, cleanup, vacuum, storage analysis, errors, disk usage, and execution time.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class WorkspaceLogger:
    """Exception-safe workspace operation logging engine."""

    def __init__(self, root_dir: Path = Path(".")) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.log_file = self.root_dir / "workspace.log"
        self.secondary_log_file = self.root_dir / "logs" / "workspace.log"
        self._ensure_log_files()

    def _ensure_log_files(self) -> None:
        """Ensure log directory exists and log files are writable."""
        try:
            self.secondary_log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def log(
        self,
        action: str,
        target: str,
        status: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None,
        execution_time_ms: float = 0.0,
        disk_usage_bytes: int = 0,
    ) -> None:
        """Log workspace operation with timestamp and metadata."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        details_str = ""
        if details:
            details_str = " | " + " ".join(f"{k}={v}" for k, v in details.items())

        log_entry = (
            f"[{timestamp}] [{action.upper()}] [{status}] Target: {target} | "
            f"Time: {execution_time_ms:.2f}ms | DiskUsage: {disk_usage_bytes} bytes{details_str}\n"
        )

        for path in [self.log_file, self.secondary_log_file]:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception:
                pass
