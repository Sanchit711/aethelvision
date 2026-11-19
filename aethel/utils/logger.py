"""Centralized Enterprise Logging Subsystem for Aethel Vision Framework.

Provides formatted console and file logging drivers with extensible hooks for
cloud telemetry and TensorBoard integration. Zero print statements allowed in production modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Global logger cache
_LOGGERS = {}


class AethelFormatter(logging.Formatter):
    """Custom structured log formatter for Aethel Vision Framework."""

    fmt_str = "%(asctime)s | %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d - %(message)s"
    date_str = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.fmt_str, datefmt=self.date_str)


def setup_logger(
    name: str = "aethel",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """Initialize or update a structured logger instance.

    Args:
        name: Unique logger namespace name (default 'aethel').
        log_file: Optional filepath to write log output.
        level: Logging verbosity level (default logging.INFO).
        console_output: Whether to attach stdout stream handler.

    Returns:
        Configured logging.Logger instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    formatter = AethelFormatter()

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = "aethel") -> logging.Logger:
    """Retrieve an existing logger or create default aethel logger.

    Args:
        name: Logger module name.

    Returns:
        Logger instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]
    return setup_logger(name=name)


import datetime
import traceback
from typing import Any, Dict, Optional, Union


class StructuredLogManager:
    """Multi-sink structured logger managing separate domain log files."""

    def __init__(self, log_dir: Union[str, Path] = "runs/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sinks: Dict[str, logging.Logger] = {}
        self._init_sinks()

    def _init_sinks(self) -> None:
        sink_names = [
            "training",
            "validation",
            "checkpoint",
            "gpu",
            "system",
            "launcher",
            "exceptions",
            "performance",
        ]
        for s in sink_names:
            p = self.log_dir / f"{s}.log"
            self.sinks[s] = setup_logger(name=f"aethel.{s}", log_file=p, console_output=False)

    def log(self, category: str, message: str, level: int = logging.INFO) -> None:
        """Route log message to specific domain sink."""
        if category in self.sinks:
            self.sinks[category].log(level, message)
        get_logger("aethel").log(level, f"[{category.upper()}] {message}")

    def log_exception(
        self,
        exc: Exception,
        epoch: int = 0,
        batch: int = 0,
        dataset: str = "Unknown",
        model: str = "Unknown",
        device: str = "Unknown",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log rich exception diagnostic payload to exceptions.log."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        root_cause = getattr(exc, "root_cause", str(exc))
        location = getattr(exc, "location", "Framework Execution Loop")
        recommended_fix = getattr(exc, "recommended_fix", "Inspect traceback log and execution hyperparameters.")

        from aethel.utils.telemetry import SystemResourceMonitor
        telemetry = SystemResourceMonitor().get_system_telemetry()
        gpu_mem = f"{telemetry.get('vram_used_mb', 0)} MB / {telemetry.get('vram_total_mb', 0)} MB"
        cpu_usage = f"{telemetry.get('cpu_usage', 0)}%"

        payload = (
            f"\n" + "=" * 80 + "\n"
            f"EXCEPTION REPORT | {now_str}\n"
            f"Exception Type:   {type(exc).__name__}\n"
            f"Message:          {exc}\n"
            f"Epoch / Batch:    Epoch {epoch} | Batch {batch}\n"
            f"Dataset / Model:  Dataset: {dataset} | Model: {model}\n"
            f"Device:           {device}\n"
            f"GPU Memory:       {gpu_mem}\n"
            f"CPU Usage:        {cpu_usage}\n"
            f"Location:         {location}\n"
            f"Root Cause:       {root_cause}\n"
            f"Recommended Fix:  {recommended_fix}\n"
            f"Traceback:\n{tb_str}"
            f"=" * 80 + "\n"
        )

        if "exceptions" in self.sinks:
            self.sinks["exceptions"].error(payload)
        get_logger("aethel").error(f"[{type(exc).__name__}] {exc} | Fix: {recommended_fix}")


_GLOBAL_STRUCTURED_LOG_MGR: Optional[StructuredLogManager] = None


def get_structured_log_manager(log_dir: Union[str, Path] = "runs/logs") -> StructuredLogManager:
    """Retrieve global structured log manager."""
    global _GLOBAL_STRUCTURED_LOG_MGR
    if _GLOBAL_STRUCTURED_LOG_MGR is None:
        _GLOBAL_STRUCTURED_LOG_MGR = StructuredLogManager(log_dir=log_dir)
    return _GLOBAL_STRUCTURED_LOG_MGR


__all__ = [
    "setup_logger",
    "get_logger",
    "AethelFormatter",
    "StructuredLogManager",
    "get_structured_log_manager",
]
