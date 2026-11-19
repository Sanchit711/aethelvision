"""AETHEL Enterprise Workspace Manager Subsystem.

Provides comprehensive workspace lifecycle management:
experiment database indexing, recoverable deletion, multi-select batch operations,
checksummed zip archiving, storage analysis, workspace health diagnostics,
vacuum engine, auto cleanup policies, recovery, logging, and report generation.
"""

from aethel.workspace.analyzer import StorageAnalyzer
from aethel.workspace.archiver import WorkspaceArchiver
from aethel.workspace.database import ExperimentDatabase
from aethel.workspace.deleter import RecoverableDeleter
from aethel.workspace.health import WorkspaceHealthChecker
from aethel.workspace.logger import WorkspaceLogger
from aethel.workspace.manager import WorkspaceManager
from aethel.workspace.policy import AutoCleanupPolicyEngine
from aethel.workspace.reports import WorkspaceReportGenerator
from aethel.workspace.vacuum import VacuumEngine

__all__ = [
    "WorkspaceManager",
    "ExperimentDatabase",
    "StorageAnalyzer",
    "RecoverableDeleter",
    "WorkspaceArchiver",
    "WorkspaceHealthChecker",
    "VacuumEngine",
    "AutoCleanupPolicyEngine",
    "WorkspaceLogger",
    "WorkspaceReportGenerator",
]
