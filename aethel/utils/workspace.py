"""Canonical Workspace Manager for Aethel Vision Framework.

Provides enterprise-grade workspace management, experiment lifecycle tracking,
safe cleanup, health monitoring, disk usage inspection, global run indexing, and migration utilities.
"""

import csv
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aethel.utils.exceptions import CleanupException, DiskSpaceException, WorkspaceException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.utils.workspace")


class WorkspaceManager:
    """Enterprise Workspace Manager orchestrating experiment lifecycles, health verification, and resource safety."""

    SUBFOLDERS = [
        "checkpoints",
        "validation",
        "reports",
        "metrics",
        "exports",
        "logs",
        "cache",
        "temp",
    ]

    STALE_FILE_PATTERNS = [
        "__pycache__",
        "*.pyc",
        "Thumbs.db",
        "desktop.ini",
        "*.tmp",
        "*.temp",
        ".DS_Store",
    ]

    def __init__(self, root_dir: Union[str, Path] = ".") -> None:
        self.root_dir = Path(root_dir).resolve()
        self.runs_dir = self.root_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.index_json = self.runs_dir / "index.json"
        self.history_csv = self.runs_dir / "history.csv"
        self.latest_json = self.runs_dir / "latest.json"

    # --- Disk Usage & Health Monitoring ---

    def get_disk_usage(self) -> Dict[str, Any]:
        """Compute disk usage metrics for workspace and host disk."""
        total_bytes = 0
        exp_count = 0
        healthy_count = 0
        corrupt_count = 0

        for exp_path in self.runs_dir.glob("exp_*"):
            if exp_path.is_dir():
                exp_count += 1
                size = sum(f.stat().st_size for f in exp_path.rglob("*") if f.is_file())
                total_bytes += size
                # Quick health check
                if (exp_path / "checkpoints" / "best.pt").exists() or (exp_path / "summary.json").exists():
                    healthy_count += 1
                else:
                    corrupt_count += 1

        # System disk free space
        try:
            stat = shutil.disk_usage(str(self.runs_dir))
            free_gb = round(stat.free / (1024 ** 3), 2)
            total_gb = round(stat.total / (1024 ** 3), 2)
        except Exception:
            free_gb, total_gb = 0.0, 0.0

        return {
            "total_workspace_mb": round(total_bytes / (1024 ** 2), 2),
            "total_workspace_gb": round(total_bytes / (1024 ** 3), 2),
            "experiment_count": exp_count,
            "healthy_experiments": healthy_count,
            "failed_experiments": corrupt_count,
            "system_disk_free_gb": free_gb,
            "system_disk_total_gb": total_gb,
        }

    def check_disk_space_threshold(self, min_free_gb: float = 1.0) -> bool:
        """Verify host disk space exceeds safety threshold."""
        try:
            stat = shutil.disk_usage(str(self.runs_dir))
            free_gb = stat.free / (1024 ** 3)
            if free_gb < min_free_gb:
                raise DiskSpaceException(
                    message=f"Critically low disk space: {free_gb:.2f} GB free (threshold: {min_free_gb} GB).",
                    details=f"Directory '{self.runs_dir}' has insufficient free disk space.",
                    recommended_fix="Purge old experiments or clean workspace cache.",
                )
            return True
        except Exception as e:
            if isinstance(e, DiskSpaceException):
                raise e
            return True

    # --- Folder & Experiment Lifecycle ---

    def create_canonical_experiment_dir(self, run_id: Optional[str] = None) -> Path:
        """Create a unique timestamped canonical experiment directory under runs/."""
        if not run_id:
            run_id = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"

        exp_dir = self.runs_dir / run_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        for sub in self.SUBFOLDERS:
            (exp_dir / sub).mkdir(parents=True, exist_ok=True)

        return exp_dir

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Discover and return summary metadata for all local experiment runs."""
        experiments = []
        for exp_path in sorted(self.runs_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not exp_path.is_dir():
                continue

            summary_file = exp_path / "summary.json"
            best_ckpt = exp_path / "checkpoints" / "best.pt"
            last_ckpt = exp_path / "checkpoints" / "last.pt"
            report_html = exp_path / "reports" / "report.html"

            is_healthy = best_ckpt.exists() or summary_file.exists()
            size_mb = round(sum(f.stat().st_size for f in exp_path.rglob("*") if f.is_file()) / (1024 ** 2), 2)

            meta: Dict[str, Any] = {
                "run_id": exp_path.name,
                "path": str(exp_path),
                "is_healthy": is_healthy,
                "size_mb": size_mb,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_path.stat().st_mtime)),
                "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
                "last_checkpoint": str(last_ckpt) if last_ckpt.exists() else None,
                "report_html": str(report_html) if report_html.exists() else None,
            }

            if summary_file.exists():
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        summary_data = json.load(f)
                    meta.update({
                        "epochs": summary_data.get("total_epochs", summary_data.get("epochs", "N/A")),
                        "best_mAP50": summary_data.get("best_mAP50", summary_data.get("mAP50", "N/A")),
                        "final_loss": summary_data.get("final_loss", summary_data.get("loss", "N/A")),
                    })
                except Exception:
                    pass

            experiments.append(meta)

        return experiments

    def rebuild_run_index(self) -> Tuple[int, Path]:
        """Rebuild global runs/index.json, runs/history.csv, and runs/latest.json."""
        exps = self.list_experiments()

        # 1. Update index.json
        index_payload = {
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runs": len(exps),
            "runs": exps,
        }
        with open(self.index_json, "w", encoding="utf-8") as f:
            json.dump(index_payload, f, indent=2)

        # 2. Update history.csv
        fieldnames = ["run_id", "modified", "size_mb", "is_healthy", "epochs", "best_mAP50", "final_loss", "path"]
        with open(self.history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in exps:
                writer.writerow(row)

        # 3. Update latest.json
        if exps:
            latest = exps[0]
            with open(self.latest_json, "w", encoding="utf-8") as f:
                json.dump(latest, f, indent=2)

        logger.info(f"Rebuilt run index with {len(exps)} experiment runs at {self.runs_dir}")
        return len(exps), self.index_json

    # --- Workspace Cleanup & Sanitization ---

    def sanitize_stale_files(self) -> int:
        """Purge temporary files, __pycache__, .pyc, Thumbs.db, desktop.ini, and 0-byte temporary files."""
        removed = 0
        for root, dirs, files in os.walk(self.root_dir):
            # Remove pycache directories
            for d in list(dirs):
                if d == "__pycache__":
                    pyc_path = Path(root) / d
                    try:
                        shutil.rmtree(pyc_path)
                        dirs.remove(d)
                        removed += 1
                    except Exception:
                        pass

            for f in files:
                f_path = Path(root) / f
                if f in ("Thumbs.db", "desktop.ini", ".DS_Store") or f.endswith(".pyc") or f.endswith(".tmp"):
                    try:
                        f_path.unlink()
                        removed += 1
                    except Exception:
                        pass

                # Purge 0-byte temporary files inside temp/ or cache/
                if ("temp" in root.lower() or "cache" in root.lower()) and f_path.is_file() and f_path.stat().st_size == 0:
                    try:
                        f_path.unlink()
                        removed += 1
                    except Exception:
                        pass

        return removed

    def delete_failed_runs(self) -> int:
        """Purge empty, interrupted, or corrupted experiment directories."""
        deleted = 0
        for exp_path in self.runs_dir.glob("exp_*"):
            if not exp_path.is_dir():
                continue

            summary_file = exp_path / "summary.json"
            best_ckpt = exp_path / "checkpoints" / "best.pt"

            # Failed run if missing checkpoints and summary metadata
            if not best_ckpt.exists() and not summary_file.exists():
                try:
                    shutil.rmtree(exp_path)
                    deleted += 1
                    logger.info(f"Purged corrupted/failed experiment directory: {exp_path}")
                except Exception as e:
                    logger.warning(f"Could not remove {exp_path}: {e}")

        self.rebuild_run_index()
        return deleted

    def clear_workspace_cache(self) -> int:
        """Clear cache subdirectories and PyTorch CUDA allocator memory."""
        cleared_files = 0
        for cache_dir in self.root_dir.rglob("cache"):
            if cache_dir.is_dir():
                for item in cache_dir.glob("*"):
                    try:
                        if item.is_file():
                            item.unlink()
                            cleared_files += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            cleared_files += 1
                    except Exception:
                        pass

        # Perform CUDA & GC RAM cleanup
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return cleared_files

    def verify_workspace_integrity(self) -> Dict[str, Any]:
        """Perform full integrity audit of workspace structure and runs/ directory."""
        stale_cleaned = self.sanitize_stale_files()
        exps = self.list_experiments()
        usage = self.get_disk_usage()

        return {
            "status": "PASS" if usage["system_disk_free_gb"] > 0.5 else "WARNING",
            "experiments_found": len(exps),
            "healthy_runs": usage["healthy_experiments"],
            "failed_runs": usage["failed_experiments"],
            "stale_files_purged": stale_cleaned,
            "workspace_mb": usage["total_workspace_mb"],
            "disk_free_gb": usage["system_disk_free_gb"],
        }

    def export_workspace_report(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Generate and save Markdown workspace audit and health report."""
        path = Path(output_path) if output_path else self.runs_dir / "workspace_health_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        info = self.verify_workspace_integrity()
        exps = self.list_experiments()

        lines = [
            "# AETHEL Workspace Health & Audit Report",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Workspace Directory:** `{self.root_dir}`",
            "",
            "## Workspace Summary",
            f"- **Overall Health Status:** {info['status']}",
            f"- **Total Experiments:** {info['experiments_found']}",
            f"- **Healthy Experiments:** {info['healthy_runs']}",
            f"- **Failed / Corrupted Runs:** {info['failed_runs']}",
            f"- **Total Workspace Size:** {info['workspace_mb']} MB",
            f"- **Host Disk Free Space:** {info['disk_free_gb']} GB",
            "",
            "## Experiments Index",
            "| Run ID | Status | Epochs | mAP@50 | Size (MB) | Modified | Path |",
            "|---|---|---|---|---|---|---|",
        ]

        for e in exps:
            status_flag = "Healthy" if e["is_healthy"] else "Incomplete/Failed"
            lines.append(
                f"| `{e['run_id']}` | {status_flag} | {e.get('epochs', 'N/A')} | {e.get('best_mAP50', 'N/A')} | {e['size_mb']} | {e['modified']} | `{e['path']}` |"
            )

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Exported workspace health report to {path}")
        return path


__all__ = ["WorkspaceManager"]
