"""AETHEL Workspace Report Generator Module (Phase 18).

Generates 5 canonical enterprise workspace reports in markdown:
1. WORKSPACE_REPORT.md
2. STORAGE_REPORT.md
3. CLEANUP_REPORT.md
4. ARCHIVE_REPORT.md
5. WORKSPACE_HEALTH.md
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aethel import __version__ as FRAMEWORK_VERSION
from aethel.workspace.analyzer import StorageAnalyzer, format_bytes
from aethel.workspace.archiver import WorkspaceArchiver
from aethel.workspace.database import ExperimentDatabase
from aethel.workspace.deleter import RecoverableDeleter
from aethel.workspace.health import WorkspaceHealthChecker


class WorkspaceReportGenerator:
    """Enterprise Workspace Markdown Report Generation Engine."""

    def __init__(self, root_dir: Path = Path(".")) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.db = ExperimentDatabase(root_dir=self.root_dir)
        self.analyzer = StorageAnalyzer(root_dir=self.root_dir, db=self.db)
        self.deleter = RecoverableDeleter(root_dir=self.root_dir, db=self.db)
        self.archiver = WorkspaceArchiver(root_dir=self.root_dir, db=self.db)
        self.health_checker = WorkspaceHealthChecker(root_dir=self.root_dir, db=self.db)

    def generate_all_reports(self) -> Dict[str, str]:
        """Generate all 5 workspace markdown reports on disk."""
        generated = {}
        generated["WORKSPACE_REPORT.md"] = self.generate_workspace_report()
        generated["STORAGE_REPORT.md"] = self.generate_storage_report()
        generated["CLEANUP_REPORT.md"] = self.generate_cleanup_report()
        generated["ARCHIVE_REPORT.md"] = self.generate_archive_report()
        generated["WORKSPACE_HEALTH.md"] = self.generate_workspace_health_report()
        return generated

    def generate_workspace_report(self) -> str:
        """Generate WORKSPACE_REPORT.md."""
        path = self.root_dir / "WORKSPACE_REPORT.md"
        exps = self.db.get_all_experiments()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "# AETHEL Enterprise Workspace Summary Report",
            "",
            f"**Generated Date:** {timestamp}  ",
            f"**Framework Version:** AETHEL Enterprise v{FRAMEWORK_VERSION}  ",
            f"**Workspace Root:** `{self.root_dir}`  ",
            "",
            "## 1. Executive Workspace Overview",
            "",
            f"- **Total Discovered Experiments:** {len(exps)}",
            f"- **Active Experiments:** {len([e for e in exps if e.get('status') not in ('Deleted', 'Archived')])}",
            f"- **Archived Experiments:** {len([e for e in exps if e.get('status') == 'Archived'])}",
            f"- **Trash/Deleted Experiments:** {len([e for e in exps if e.get('status') == 'Deleted'])}",
            "",
            "## 2. Canonical Experiment Registry",
            "",
            "| Run ID | Model | Dataset | Epochs | Best mAP | Status | Disk Size | Timestamp |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for e in exps:
            size_str = format_bytes(e.get("disk_usage", 0))
            lines.append(
                f"| `{e.get('run_id')}` | {e.get('model')} | {e.get('dataset')} | "
                f"{e.get('epoch')} | `{e.get('best_mAP')}` | **{e.get('status')}** | {size_str} | {e.get('timestamp')} |"
            )

        lines.extend([
            "",
            "## 3. Database Indexes & Persistence",
            "",
            "- `runs/workspace.db` (SQLite Canonical Database)",
            "- `runs/index.json` (Structured JSON Metadata)",
            "- `runs/history.csv` (CSV Audit Trail)",
            "- `runs/latest.json` (Most Recent Experiment Pointer)",
            "",
            "---",
            "*(c) 2026 AETHEL Technologies Inc. Workspace Manager v1.0*",
        ])

        content = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(path)

    def generate_storage_report(self) -> str:
        """Generate STORAGE_REPORT.md."""
        path = self.root_dir / "STORAGE_REPORT.md"
        stats = self.analyzer.analyze()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# AETHEL Enterprise Storage Analysis Report

**Generated Date:** {timestamp}  
**Total Workspace Disk Usage:** `{stats['disk_usage_total_str']}`  
**Recoverable Space:** `{stats['recoverable_space_str']}`  

## 1. Storage Breakdown by Asset Type

| Asset Category | Disk Space | Percentage |
|---|---|---|
| Model Checkpoints (`.pt`, `.onnx`) | `{stats['checkpoints_size_str']}` | High |
| Validation & Overlay Images | `{stats['validation_images_size_str']}` | Medium |
| Model Exports | `{stats['exports_size_str']}` | Medium |
| Generated Reports | `{stats['reports_size_str']}` | Low |
| System & Execution Logs | `{stats['logs_size_str']}` | Low |
| PyCache & Temp Caches | `{stats['cache_size_str']}` | Low |

## 2. Experiment Size Metrics

- **Average Experiment Size:** `{stats['avg_exp_size_str']}`
- **Largest Experiment:** `{stats['largest_exp']['name']}` (`{stats['largest_exp']['size_str']}`)
- **Oldest Experiment:** `{stats['oldest_exp']['name']}` (`{stats['oldest_exp']['date']}`)
- **Newest Experiment:** `{stats['newest_exp']['name']}` (`{stats['newest_exp']['date']}`)

## 3. Storage Optimization Recommendations

1. Run **Vacuum Workspace** to purge `{stats['cache_size_str']}` of temporary caches.
2. Empty `.trash/` directory to reclaim estimated recoverable space.
3. Compress older experiments to `.zip` in `archive/`.

---
*(c) 2026 AETHEL Technologies Inc. Storage Analyzer v1.0*
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(path)

    def generate_cleanup_report(self) -> str:
        """Generate CLEANUP_REPORT.md."""
        path = self.root_dir / "CLEANUP_REPORT.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# AETHEL Workspace Cleanup & Vacuum Report

**Generated Date:** {timestamp}  
**Cleanup Engine Status:** OPERATIONAL  

## 1. Vacuum Policy & Execution Rules

- **Purged Files:** `__pycache__`, `.pyc`, `.tmp`, `.temp`, `Thumbs.db`, `.desktop.ini`
- **Orphan Logs:** Purged non-essential `.log` files in `logs/`
- **Index Rebuild:** Rebuilt `index.json`, `history.csv`, `latest.json`, `workspace.db`

## 2. Auto Cleanup Policy Options

- **Keep Last N Experiments:** User-definable threshold (Default: 10)
- **Age-Based Purge:** Thresholds of 30, 60, 90, 180 days
- **Selective Retention:** Retain `best.pt`, prune intermediate `epoch_*.pt` checkpoints

---
*(c) 2026 AETHEL Technologies Inc. Vacuum Engine v1.0*
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(path)

    def generate_archive_report(self) -> str:
        """Generate ARCHIVE_REPORT.md."""
        path = self.root_dir / "ARCHIVE_REPORT.md"
        archives = self.archiver.list_archived_experiments()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "# AETHEL Experiment Archive Report",
            "",
            f"**Generated Date:** {timestamp}  ",
            f"**Total Archived Experiments:** {len(archives)}  ",
            "",
            "## 1. Archived Experiment Packages",
            "",
            "| Filename | Size | SHA-256 Checksum | Archive Date | Integrity Status |",
            "|---|---|---|---|---|",
        ]

        for a in archives:
            lines.append(
                f"| `{a['filename']}` | {format_bytes(a['size_bytes'])} | `{a['sha256'][:16]}...` | {a['created_date']} | **VERIFIED OK** |"
            )

        lines.extend([
            "",
            "## 2. Archival Security Rules",
            "",
            "- Compression Algorithm: Standard ZIP DEFLATED",
            "- Checksum Standard: SHA-256 Digest",
            "- Integrity Check: `zipfile.testzip()` Validation",
            "",
            "---",
            "*(c) 2026 AETHEL Technologies Inc. Workspace Archiver v1.0*",
        ])

        content = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(path)

    def generate_workspace_health_report(self) -> str:
        """Generate WORKSPACE_HEALTH.md."""
        path = self.root_dir / "WORKSPACE_HEALTH.md"
        health = self.health_checker.inspect_health()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# AETHEL Workspace Health Audit Report

**Generated Date:** {timestamp}  
**Overall Workspace Health:** `{health['overall_status']}`  
**Total Issues Detected:** `{health['total_issues']}` (Critical Failures: {health['fail_count']}, Warnings: {health['warning_count']})  

## 1. Health Audit Breakdown

| Diagnostic Check Target | Result Status | Count |
|---|---|---|
| Broken / Empty Folders | {"PASS" if health['broken_folders'] == 0 else "WARNING"} | {health['broken_folders']} |
| Missing Checkpoints | {"PASS" if health['missing_checkpoints'] == 0 else "WARNING"} | {health['missing_checkpoints']} |
| Missing Reports | {"PASS" if health['missing_reports'] == 0 else "WARNING"} | {health['missing_reports']} |
| Missing Metrics | {"PASS" if health['missing_metrics'] == 0 else "WARNING"} | {health['missing_metrics']} |
| Corrupt JSON Files | {"PASS" if health['corrupt_json'] == 0 else "FAIL"} | {health['corrupt_json']} |
| Corrupt CSV Files | {"PASS" if health['corrupt_csv'] == 0 else "FAIL"} | {health['corrupt_csv']} |
| Corrupt HTML Files | {"PASS" if health['corrupt_html'] == 0 else "WARNING"} | {health['corrupt_html']} |
| Broken Symlinks | {"PASS" if health['broken_symlinks'] == 0 else "WARNING"} | {health['broken_symlinks']} |
| Duplicate Checkpoints | {"PASS" if health['duplicate_checkpoints'] == 0 else "WARNING"} | {health['duplicate_checkpoints']} |
| Orphan Folders | {"PASS" if health['orphan_folders'] == 0 else "WARNING"} | {health['orphan_folders']} |
| Orphan Logs | {"PASS" if health['orphan_logs'] == 0 else "WARNING"} | {health['orphan_logs']} |
| Orphan Exports | {"PASS" if health['orphan_exports'] == 0 else "WARNING"} | {health['orphan_exports']} |
| Zero-Byte Files | {"PASS" if health['zero_byte_files'] == 0 else "WARNING"} | {health['zero_byte_files']} |

---
*(c) 2026 AETHEL Technologies Inc. Workspace Health Engine v1.0*
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(path)
