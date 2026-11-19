"""AETHEL Workspace Health Inspector Module (Phase 10).

Detects workspace anomalies: broken folders, missing checkpoints/reports/metrics,
corrupt JSON/CSV/HTML, broken symlinks, duplicate checkpoints, orphan folders/logs/exports,
zero-byte files. Categorizes overall status as PASS, WARNING, or FAIL.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from aethel.utils.logger import get_logger
from aethel.workspace.database import ExperimentDatabase

logger = get_logger("aethel.workspace.health")


class WorkspaceHealthChecker:
    """Workspace Health Diagnostics Inspector Engine."""

    def __init__(self, root_dir: Path = Path("."), db: Optional[ExperimentDatabase] = None) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.runs_dir = self.root_dir / "runs"
        self.db = db or ExperimentDatabase(root_dir=self.root_dir)

    def inspect_health(self) -> Dict[str, Any]:
        """Perform comprehensive 13-point workspace health diagnostic audit."""
        issues: List[Dict[str, Any]] = []

        broken_folders = 0
        missing_checkpoints = 0
        missing_reports = 0
        missing_metrics = 0
        corrupt_json = 0
        corrupt_csv = 0
        corrupt_html = 0
        broken_symlinks = 0
        duplicate_checkpoints = 0
        orphan_folders = 0
        orphan_logs = 0
        orphan_exports = 0
        zero_byte_files = 0

        # Track checksums for duplicate checkpoint detection
        ckpt_checksums: Dict[str, List[Path]] = {}

        exps = self.db.get_all_experiments()
        tracked_folders = {Path(e["folder"]).resolve() for e in exps if "folder" in e}

        # 1. Scan runs/ directory
        if self.runs_dir.exists():
            for p in self.runs_dir.rglob("*"):
                # Check broken symlinks
                if p.is_symlink() and not p.exists():
                    broken_symlinks += 1
                    issues.append({"type": "Broken Symlink", "path": str(p), "severity": "WARNING"})
                    continue

                if not p.exists():
                    continue

                # Check zero-byte files
                if p.is_file() and p.stat().st_size == 0:
                    zero_byte_files += 1
                    issues.append({"type": "Zero-Byte File", "path": str(p), "severity": "WARNING"})

                # Check corrupt JSON files
                if p.is_file() and p.suffix.lower() == ".json":
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            json.load(f)
                    except Exception as e:
                        corrupt_json += 1
                        issues.append({"type": "Corrupt JSON", "path": str(p), "severity": "FAIL", "error": str(e)})

                # Check corrupt CSV files
                if p.is_file() and p.suffix.lower() == ".csv":
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            reader = csv.reader(f)
                            list(reader)
                    except Exception as e:
                        corrupt_csv += 1
                        issues.append({"type": "Corrupt CSV", "path": str(p), "severity": "FAIL", "error": str(e)})

                # Check corrupt HTML files
                if p.is_file() and p.suffix.lower() == ".html":
                    if p.stat().st_size < 10:
                        corrupt_html += 1
                        issues.append({"type": "Corrupt HTML", "path": str(p), "severity": "WARNING"})

                # Check duplicate checkpoints
                if p.is_file() and p.suffix.lower() in (".pt", ".pth", ".onnx", ".engine") and p.stat().st_size > 1024:
                    try:
                        hasher = hashlib.sha256()
                        with open(p, "rb") as f:
                            hasher.update(f.read(65536))
                        h = hasher.hexdigest()
                        if h not in ckpt_checksums:
                            ckpt_checksums[h] = []
                        ckpt_checksums[h].append(p)
                    except Exception:
                        pass

            # 2. Check orphan folders in runs/
            for d in self.runs_dir.iterdir():
                if d.is_dir() and d.name not in (".trash", "__pycache__"):
                    if d.resolve() not in tracked_folders:
                        orphan_folders += 1
                        issues.append({"type": "Orphan Folder", "path": str(d), "severity": "WARNING"})

                    # Check broken/empty folder
                    if not list(d.glob("*")):
                        broken_folders += 1
                        issues.append({"type": "Broken Folder (Empty)", "path": str(d), "severity": "WARNING"})

                    # Check missing metrics or summary in active experiment run
                    if not (d / "summary.json").exists() and not (d / "metrics.csv").exists():
                        missing_metrics += 1
                        issues.append({"type": "Missing Metrics", "path": str(d), "severity": "WARNING"})

                    # Check missing reports
                    if not list(d.glob("*.html")) and not (self.root_dir / "reports").exists():
                        missing_reports += 1
                        issues.append({"type": "Missing Reports", "path": str(d), "severity": "WARNING"})

        # Count duplicate checkpoints
        for h, paths in ckpt_checksums.items():
            if len(paths) > 1:
                duplicate_checkpoints += (len(paths) - 1)
                for dup_p in paths[1:]:
                    issues.append({"type": "Duplicate Checkpoint", "path": str(dup_p), "severity": "WARNING"})

        # 3. Check orphan logs in logs/
        logs_dir = self.root_dir / "logs"
        if logs_dir.exists():
            for log_p in logs_dir.glob("*.log"):
                if log_p.name != "workspace.log" and log_p.name != "aethel.log":
                    orphan_logs += 1
                    issues.append({"type": "Orphan Log", "path": str(log_p), "severity": "WARNING"})

        # 4. Check orphan exports
        exports_dir = self.root_dir / "exports"
        if exports_dir.exists():
            for exp_p in exports_dir.glob("*.*"):
                orphan_exports += 1
                issues.append({"type": "Unassigned Export", "path": str(exp_p), "severity": "WARNING"})

        # Determine overall workspace health status
        fail_count = sum(1 for i in issues if i.get("severity") == "FAIL")
        warning_count = sum(1 for i in issues if i.get("severity") == "WARNING")

        if fail_count > 0:
            overall_status = "FAIL"
        elif warning_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"

        return {
            "overall_status": overall_status,
            "total_issues": len(issues),
            "fail_count": fail_count,
            "warning_count": warning_count,
            "broken_folders": broken_folders,
            "missing_checkpoints": missing_checkpoints,
            "missing_reports": missing_reports,
            "missing_metrics": missing_metrics,
            "corrupt_json": corrupt_json,
            "corrupt_csv": corrupt_csv,
            "corrupt_html": corrupt_html,
            "broken_symlinks": broken_symlinks,
            "duplicate_checkpoints": duplicate_checkpoints,
            "orphan_folders": orphan_folders,
            "orphan_logs": orphan_logs,
            "orphan_exports": orphan_exports,
            "zero_byte_files": zero_byte_files,
            "issues": issues,
        }
