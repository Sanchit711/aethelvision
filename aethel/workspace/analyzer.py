"""AETHEL Storage Analyzer Module (Phase 2).

Scans workspace to calculate experiment statistics, disk usage breakdown by asset type
(checkpoints, validation images, exports, reports, logs, cache), average/largest/oldest/newest
experiment metrics, and estimated recoverable space.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aethel.workspace.database import ExperimentDatabase


def format_bytes(size_bytes: float) -> str:
    """Format byte size into human readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{int(size_bytes)} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class StorageAnalyzer:
    """Enterprise Storage Analyzer Engine."""

    def __init__(self, root_dir: Path = Path("."), db: Optional[ExperimentDatabase] = None) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.db = db or ExperimentDatabase(root_dir=self.root_dir)

    def analyze(self) -> Dict[str, Any]:
        """Perform full workspace storage analysis."""
        exps = self.db.get_all_experiments()

        counts = {
            "total": len(exps),
            "running": 0,
            "completed": 0,
            "failed": 0,
            "interrupted": 0,
            "archived": 0,
            "deleted": 0,
        }

        for e in exps:
            st = str(e.get("status", "Completed")).lower()
            if "run" in st:
                counts["running"] += 1
            elif "fail" in st:
                counts["failed"] += 1
            elif "interrupt" in st:
                counts["interrupted"] += 1
            elif "archive" in st:
                counts["archived"] += 1
            elif "delete" in st:
                counts["deleted"] += 1
            else:
                counts["completed"] += 1

        # Calculate sizes by category across the filesystem
        checkpoints_size = self._calc_size_by_ext(
            [".pt", ".pth", ".ckpt", ".onnx", ".engine", ".tflite"],
            dirs=[self.root_dir / "runs", self.root_dir / "checkpoints", self.root_dir / "weights"]
        )

        validation_images_size = self._calc_size_by_ext(
            [".png", ".jpg", ".jpeg", ".bmp"],
            dirs=[self.root_dir / "runs", self.root_dir / "prediction_overlays", self.root_dir / "ground_truth_overlays", self.root_dir / "comparison_overlays"]
        )

        exports_size = self._calc_dir_size(self.root_dir / "exports") + self._calc_dir_size(self.root_dir / "export")
        reports_size = self._calc_dir_size(self.root_dir / "reports") + self._calc_size_by_ext([".html", ".md"], dirs=[self.root_dir / "runs"])
        logs_size = self._calc_dir_size(self.root_dir / "logs") + self._calc_size_by_ext([".log"], dirs=[self.root_dir])
        cache_size = (
            self._calc_dir_size(self.root_dir / "__pycache__")
            + self._calc_dir_size(self.root_dir / ".pytest_cache")
            + self._calc_size_by_glob("__pycache__", root=self.root_dir)
            + self._calc_size_by_ext([".pyc", ".tmp", ".temp"])
        )

        trash_size = self._calc_dir_size(self.root_dir / "runs" / ".trash")
        total_runs_size = self._calc_dir_size(self.root_dir / "runs")
        total_archive_size = self._calc_dir_size(self.root_dir / "archive")
        total_workspace_size = total_runs_size + total_archive_size + exports_size + reports_size + logs_size + cache_size

        # Find largest, oldest, newest experiment
        active_exps = [e for e in exps if e.get("status") not in ("Deleted", "Archived")]
        target_exps = active_exps if active_exps else exps

        avg_size = total_runs_size / max(1, len(target_exps))

        largest_exp = {"name": "N/A", "size_bytes": 0, "size_str": "0 B"}
        oldest_exp = {"name": "N/A", "date": "N/A"}
        newest_exp = {"name": "N/A", "date": "N/A"}

        if target_exps:
            sorted_by_size = sorted(target_exps, key=lambda x: int(x.get("disk_usage", 0)), reverse=True)
            largest_exp = {
                "name": sorted_by_size[0].get("run_id", "N/A"),
                "size_bytes": sorted_by_size[0].get("disk_usage", 0),
                "size_str": format_bytes(sorted_by_size[0].get("disk_usage", 0)),
            }

            sorted_by_date = sorted(target_exps, key=lambda x: str(x.get("timestamp", "")))
            oldest_exp = {
                "name": sorted_by_date[0].get("run_id", "N/A"),
                "date": sorted_by_date[0].get("timestamp", "N/A"),
            }
            newest_exp = {
                "name": sorted_by_date[-1].get("run_id", "N/A"),
                "date": sorted_by_date[-1].get("timestamp", "N/A"),
            }

        # Recoverable space calculation (trash + cache + failed experiments + orphan files)
        failed_exps_size = sum(int(e.get("disk_usage", 0)) for e in exps if str(e.get("status")).lower() in ("failed", "interrupted"))
        recoverable_space = trash_size + cache_size + failed_exps_size

        return {
            "counts": counts,
            "disk_usage_total_bytes": total_workspace_size,
            "disk_usage_total_str": format_bytes(total_workspace_size),
            "checkpoints_size_bytes": checkpoints_size,
            "checkpoints_size_str": format_bytes(checkpoints_size),
            "validation_images_size_bytes": validation_images_size,
            "validation_images_size_str": format_bytes(validation_images_size),
            "exports_size_bytes": exports_size,
            "exports_size_str": format_bytes(exports_size),
            "reports_size_bytes": reports_size,
            "reports_size_str": format_bytes(reports_size),
            "logs_size_bytes": logs_size,
            "logs_size_str": format_bytes(logs_size),
            "cache_size_bytes": cache_size,
            "cache_size_str": format_bytes(cache_size),
            "avg_exp_size_bytes": avg_size,
            "avg_exp_size_str": format_bytes(avg_size),
            "largest_exp": largest_exp,
            "oldest_exp": oldest_exp,
            "newest_exp": newest_exp,
            "recoverable_space_bytes": recoverable_space,
            "recoverable_space_str": format_bytes(recoverable_space),
        }

    def _calc_dir_size(self, path: Path) -> int:
        """Calculate total directory size safely."""
        total = 0
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        for p in path.glob("**/*"):
            if p.is_file() and not p.is_symlink():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
        return total

    def _calc_size_by_ext(self, extensions: List[str], dirs: Optional[List[Path]] = None) -> int:
        """Calculate total size of files matching given extensions within specified directories."""
        total = 0
        if dirs is None:
            dirs = [self.root_dir]
        ext_set = {e.lower() for e in extensions}
        for d in dirs:
            if d.exists():
                for p in d.glob("**/*"):
                    if p.is_file() and p.suffix.lower() in ext_set:
                        try:
                            total += p.stat().st_size
                        except Exception:
                            pass
        return total

    def _calc_size_by_glob(self, pattern: str, root: Path) -> int:
        """Calculate size of matching directories or files matching glob pattern."""
        total = 0
        if root.exists():
            for p in root.glob(f"**/{pattern}"):
                if p.is_file():
                    total += p.stat().st_size
                elif p.is_dir():
                    total += self._calc_dir_size(p)
        return total
