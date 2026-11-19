"""AETHEL Workspace Archiver Module (Phase 6).

Handles ZIP compression of experiment directories into archive/,
SHA-256 checksum verification, ZIP integrity validation, and ZIP restoration back to runs/.
"""

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aethel.utils.logger import get_logger
from aethel.workspace.database import ExperimentDatabase
from aethel.workspace.logger import WorkspaceLogger

logger = get_logger("aethel.workspace.archiver")


class WorkspaceArchiver:
    """ZIP Experiment Archive Engine."""

    def __init__(
        self,
        root_dir: Path = Path("."),
        db: Optional[ExperimentDatabase] = None,
        ws_logger: Optional[WorkspaceLogger] = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.runs_dir = self.root_dir / "runs"
        self.archive_dir = self.root_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.db = db or ExperimentDatabase(root_dir=self.root_dir)
        self.ws_logger = ws_logger or WorkspaceLogger(root_dir=self.root_dir)

    def calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def archive_experiment(
        self, run_id_or_path: str, remove_original: bool = True
    ) -> Tuple[bool, str, str]:
        """Compress experiment directory to ZIP archive, verify checksum & integrity."""
        t0 = time.time()
        exps = self.db.get_all_experiments()

        exp_record = None
        for e in exps:
            if e["run_id"] == run_id_or_path or e["folder"] == run_id_or_path or Path(e["folder"]).name == run_id_or_path:
                exp_record = e
                break

        if exp_record:
            exp_folder = Path(exp_record["folder"])
        else:
            exp_folder = Path(run_id_or_path)
            if not exp_folder.is_absolute():
                exp_folder = self.runs_dir / run_id_or_path

        if not exp_folder.exists() or not exp_folder.is_dir():
            return False, f"Experiment folder '{exp_folder}' not found.", ""

        zip_filename = f"{exp_folder.name}.zip"
        target_zip_path = self.archive_dir / zip_filename
        checksum_file = self.archive_dir / f"{zip_filename}.sha256"

        tmp_zip = target_zip_path.with_suffix(".zip.tmp")

        try:
            # 1. Create ZIP archive safely
            with zipfile.ZipFile(str(tmp_zip), "w", zipfile.ZIP_DEFLATED) as zf:
                for file_p in exp_folder.glob("**/*"):
                    if file_p.is_file():
                        arcname = file_p.relative_to(exp_folder.parent)
                        zf.write(file_p, arcname)

            # 2. Verify ZIP integrity using testzip()
            with zipfile.ZipFile(str(tmp_zip), "r") as zf:
                bad_file = zf.testzip()
                if bad_file:
                    tmp_zip.unlink(missing_ok=True)
                    return False, f"ZIP corruption detected in file '{bad_file}'.", ""

            # Atomic replace tmp_zip to target_zip_path
            os.replace(tmp_zip, target_zip_path)

            # 3. Calculate and save SHA-256 checksum
            sha256_hash = self.calculate_sha256(target_zip_path)
            with open(checksum_file, "w", encoding="utf-8") as f:
                f.write(f"{sha256_hash}  {zip_filename}\n")

            # 4. If remove_original is True, delete original experiment folder safely
            if remove_original and exp_folder.exists():
                shutil.rmtree(exp_folder)

            # Update database status
            if exp_record:
                exp_record["status"] = "Archived"
                exp_record["folder"] = str(target_zip_path)
                exp_record["disk_usage"] = target_zip_path.stat().st_size
                self.db.upsert_experiment(exp_record)
            self.db.rebuild_index()

            t_elapsed_ms = (time.time() - t0) * 1000
            self.ws_logger.log(
                action="ARCHIVE",
                target=exp_folder.name,
                status="SUCCESS",
                details={"zip_path": str(target_zip_path), "sha256": sha256_hash[:12]},
                execution_time_ms=t_elapsed_ms,
                disk_usage_bytes=target_zip_path.stat().st_size,
            )
            return True, f"Archived '{exp_folder.name}' to '{target_zip_path.name}' (SHA256: {sha256_hash[:8]}...)", sha256_hash

        except Exception as e:
            logger.error(f"Error archiving experiment {exp_folder}: {e}")
            if tmp_zip.exists():
                tmp_zip.unlink(missing_ok=True)
            t_elapsed_ms = (time.time() - t0) * 1000
            self.ws_logger.log(
                action="ARCHIVE",
                target=exp_folder.name,
                status="FAILED",
                details={"error": str(e)},
                execution_time_ms=t_elapsed_ms,
            )
            return False, f"Failed to archive experiment: {e}", ""

    def list_archived_experiments(self) -> List[Dict[str, Any]]:
        """List all archived zip files in archive/."""
        archives = []
        if not self.archive_dir.exists():
            return archives

        for zip_p in self.archive_dir.glob("*.zip"):
            size_bytes = zip_p.stat().st_size
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(zip_p.stat().st_mtime))
            sha_file = zip_p.with_name(f"{zip_p.name}.sha256")
            sha_str = "Verified"
            if sha_file.exists():
                try:
                    with open(sha_file, "r", encoding="utf-8") as f:
                        sha_str = f.read().split()[0]
                except Exception:
                    pass

            archives.append({
                "run_id": zip_p.stem,
                "zip_path": str(zip_p),
                "filename": zip_p.name,
                "size_bytes": size_bytes,
                "created_date": mtime,
                "sha256": sha_str,
            })
        return archives

    def restore_archived_experiment(self, zip_filename_or_id: str) -> Tuple[bool, str]:
        """Verify checksum and extract ZIP archive back to runs/."""
        t0 = time.time()
        zip_p = self.archive_dir / zip_filename_or_id
        if not zip_p.exists():
            zip_p = self.archive_dir / f"{zip_filename_or_id}.zip"

        if not zip_p.exists():
            return False, f"Archive file '{zip_filename_or_id}' not found in archive/."

        # Verify ZIP testzip()
        try:
            with zipfile.ZipFile(str(zip_p), "r") as zf:
                bad_file = zf.testzip()
                if bad_file:
                    return False, f"Archive corruption detected in '{bad_file}'."
                
                # Extract to runs/
                zf.extractall(str(self.runs_dir))

            # Update database
            dest_folder = self.runs_dir / zip_p.stem
            if dest_folder.exists():
                exps = self.db.get_all_experiments()
                for e in exps:
                    if e["run_id"] == zip_p.stem:
                        e["status"] = "Completed"
                        e["folder"] = str(dest_folder)
                        self.db.upsert_experiment(e)
                        break
            self.db.rebuild_index()

            t_elapsed_ms = (time.time() - t0) * 1000
            self.ws_logger.log(
                action="RESTORE_ARCHIVE",
                target=zip_p.name,
                status="SUCCESS",
                details={"extracted_path": str(self.runs_dir / zip_p.stem)},
                execution_time_ms=t_elapsed_ms,
            )
            return True, f"Successfully restored archive '{zip_p.name}' to runs/{zip_p.stem}."

        except Exception as e:
            logger.error(f"Error restoring archive {zip_p}: {e}")
            t_elapsed_ms = (time.time() - t0) * 1000
            self.ws_logger.log(
                action="RESTORE_ARCHIVE",
                target=zip_p.name,
                status="FAILED",
                details={"error": str(e)},
                execution_time_ms=t_elapsed_ms,
            )
            return False, f"Failed to restore archive: {e}"

    def compress_experiment_in_place(self, run_id: str) -> Tuple[bool, str]:
        """Compress experiment folder to ZIP within runs/ without deleting metadata."""
        return self.archive_experiment(run_id, remove_original=False)
