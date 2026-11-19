"""Unit & Regression Test Suite for AURELIX Enterprise Workspace Manager (Phase 17).

Verifies workspace indexing, database synchronization, recoverable deletion,
trash recovery, ZIP archiving with SHA-256 checksums, vacuum engine, auto-cleanup policies,
search filtering, sorting, storage analysis, and cross-platform pathlib handling.
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from aurelix.workspace.analyzer import StorageAnalyzer
from aurelix.workspace.archiver import WorkspaceArchiver
from aurelix.workspace.database import ExperimentDatabase
from aurelix.workspace.deleter import RecoverableDeleter
from aurelix.workspace.health import WorkspaceHealthChecker
from aurelix.workspace.logger import WorkspaceLogger
from aurelix.workspace.manager import WorkspaceManager, open_folder_in_explorer
from aurelix.workspace.policy import AutoCleanupPolicyEngine
from aurelix.workspace.reports import WorkspaceReportGenerator
from aurelix.workspace.vacuum import VacuumEngine


@pytest.fixture
def temp_workspace(tmp_path: Path):
    """Fixture initializing a clean temporary workspace structure."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create test experiment folders
    exp1 = runs_dir / "exp_001_test"
    exp1.mkdir(parents=True, exist_ok=True)
    with open(exp1 / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": "exp_001_test", "best_mAP": 0.95, "epoch": 20, "model": "AurelixNet-SMALL"}, f)
    (exp1 / "best.pt").write_bytes(b"0" * 1024)

    exp2 = runs_dir / "exp_002_failed"
    exp2.mkdir(parents=True, exist_ok=True)
    (exp2 / ".failed").touch()
    with open(exp2 / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": "exp_002_failed", "best_mAP": 0.50, "epoch": 5, "model": "AurelixNet-TINY"}, f)

    return tmp_path


def test_experiment_database_indexing(temp_workspace: Path):
    """Test SQLite DB, index.json, history.csv, and latest.json generation."""
    db = ExperimentDatabase(root_dir=temp_workspace)
    exps = db.get_all_experiments()

    assert len(exps) >= 2
    assert (temp_workspace / "runs" / "workspace.db").exists()
    assert (temp_workspace / "runs" / "index.json").exists()
    assert (temp_workspace / "runs" / "history.csv").exists()
    assert (temp_workspace / "runs" / "latest.json").exists()

    # Query SQLite database directly
    with sqlite3.connect(str(temp_workspace / "runs" / "workspace.db")) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiments")
        count = cursor.fetchone()[0]
        assert count >= 2


def test_recoverable_deleter(temp_workspace: Path):
    """Test moving experiment to runs/.trash/ and metadata creation."""
    deleter = RecoverableDeleter(root_dir=temp_workspace)
    
    # Pre-deletion check
    assert (temp_workspace / "runs" / "exp_001_test").exists()

    success, msg, reclaimed = deleter.delete_experiment("exp_001_test", confirmed=True)
    assert success
    assert not (temp_workspace / "runs" / "exp_001_test").exists()
    assert (temp_workspace / "runs" / ".trash").exists()

    # Verify trash metadata file
    trash_items = list((temp_workspace / "runs" / ".trash").iterdir())
    assert len(trash_items) > 0
    trash_folder = trash_items[0]
    meta_file = trash_folder / "trash_metadata.json"
    assert meta_file.exists()

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["experiment_metadata"]["run_id"] == "exp_001_test"


def test_trash_recovery(temp_workspace: Path):
    """Test restoring deleted experiment from runs/.trash/ back to runs/."""
    deleter = RecoverableDeleter(root_dir=temp_workspace)
    deleter.delete_experiment("exp_001_test", confirmed=True)

    deleted_list = deleter.list_deleted_experiments()
    assert len(deleted_list) > 0

    trash_name = deleted_list[0]["trash_folder"]
    res_success, res_msg = deleter.restore_deleted_experiment(trash_name)
    assert res_success
    assert (temp_workspace / "runs" / "exp_001_test").exists()


def test_zip_archiver_and_checksum(temp_workspace: Path):
    """Test compressing experiment to ZIP, generating SHA-256 checksum, and extracting."""
    archiver = WorkspaceArchiver(root_dir=temp_workspace)

    success, msg, sha256 = archiver.archive_experiment("exp_001_test", remove_original=True)
    assert success
    assert sha256 != ""
    assert (temp_workspace / "archive" / "exp_001_test.zip").exists()
    assert (temp_workspace / "archive" / "exp_001_test.zip.sha256").exists()

    # Restore archive
    r_success, r_msg = archiver.restore_archived_experiment("exp_001_test.zip")
    assert r_success
    assert (temp_workspace / "runs" / "exp_001_test").exists()


def test_vacuum_engine(temp_workspace: Path):
    """Test vacuuming junk files and rebuilding index."""
    # Create junk files
    pycache = temp_workspace / "__pycache__"
    pycache.mkdir(parents=True, exist_ok=True)
    (pycache / "test.pyc").write_bytes(b"junk")
    (temp_workspace / "junk.tmp").write_bytes(b"junk")

    vacuum = VacuumEngine(root_dir=temp_workspace)
    res = vacuum.vacuum()

    assert res["status"] == "SUCCESS"
    assert res["items_removed_count"] >= 2
    assert not pycache.exists()
    assert not (temp_workspace / "junk.tmp").exists()


def test_storage_analyzer(temp_workspace: Path):
    """Test storage analysis calculations."""
    analyzer = StorageAnalyzer(root_dir=temp_workspace)
    stats = analyzer.analyze()

    assert stats["counts"]["total"] >= 2
    assert "disk_usage_total_str" in stats
    assert "recoverable_space_str" in stats


def test_workspace_health_checker(temp_workspace: Path):
    """Test health diagnostic check."""
    checker = WorkspaceHealthChecker(root_dir=temp_workspace)
    health = checker.inspect_health()

    assert health["overall_status"] in ("PASS", "WARNING", "FAIL")
    assert "broken_folders" in health


def test_auto_cleanup_policy(temp_workspace: Path):
    """Test auto cleanup policy preview and execution."""
    policy_engine = AutoCleanupPolicyEngine(root_dir=temp_workspace)
    policy = {"keep_last_n": 1}

    preview = policy_engine.preview_policy(policy)
    assert "estimated_reclaimed_bytes" in preview

    res = policy_engine.execute_policy(policy, confirmed=True)
    assert res["status"] == "SUCCESS"


def test_search_and_sorting(temp_workspace: Path):
    """Test search partial matching and sorting algorithm."""
    manager = WorkspaceManager(root_dir=temp_workspace)
    
    search_res = manager.search_experiments("failed")
    assert len(search_res) >= 1

    all_exps = manager.get_experiments()
    sorted_by_map = manager.sort_experiments(all_exps, sort_key="map", descending=True)
    assert len(sorted_by_map) >= 2


def test_report_generator(temp_workspace: Path):
    """Test generating all 5 workspace reports."""
    reporter = WorkspaceReportGenerator(root_dir=temp_workspace)
    reports = reporter.generate_all_reports()

    assert len(reports) == 5
    for r_filename, r_path in reports.items():
        assert Path(r_path).exists()


def test_open_folder_cross_platform(temp_workspace: Path):
    """Test cross-platform folder opener wrapper."""
    success, msg = open_folder_in_explorer(temp_workspace)
    assert success or "suppressed" in msg.lower() or "opened" in msg.lower()
