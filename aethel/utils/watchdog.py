"""Startup Timeout Watchdog Subsystem for Aethel Vision Framework.

Monitors training startup latency, detects hangs or deadlocks (> 60s total startup or > 20s per batch fetch),
dumps thread stack traces, RAM/CPU/GPU state, current sample details, and writes startup_diagnostics.log.
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

from aethel.launcher.hardware import GPUDiagnostics
from aethel.utils.exceptions import TrainingException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.utils.watchdog")


class StartupWatchdog:
    """Watchdog for detecting startup deadlocks and collecting diagnostic dumps."""

    def __init__(
        self,
        max_startup_sec: float = 60.0,
        max_batch_fetch_sec: float = 20.0,
        output_dir: Union[str, Path] = Path("runs"),
    ) -> None:
        self.max_startup_sec = max_startup_sec
        self.max_batch_fetch_sec = max_batch_fetch_sec
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.last_batch_time = time.time()

        # Telemetry & State Tracking
        self.current_stage = "Initializing"
        self.num_workers = 0
        self.dataset_size = 0
        self.batch_size = 0
        self.cache_status = "Disabled"
        self.current_filename = "N/A"
        self.current_annotation = "N/A"
        self.last_successful_sample = "N/A"

    def update_stage(self, stage_name: str, **kwargs: Any) -> None:
        """Update current startup stage and telemetry parameters."""
        self.current_stage = stage_name
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        logger.info(f"Startup Stage Update: [{stage_name}]")

    def touch_batch(self, sample_filename: Optional[str] = None) -> None:
        """Record successful batch fetch timestamp."""
        self.last_batch_time = time.time()
        if sample_filename:
            self.last_successful_sample = sample_filename

    def check_timeout(self) -> None:
        """Verify startup time and batch fetch latency bounds. Trigger diagnostic dump if exceeded."""
        now = time.time()
        elapsed_startup = now - self.start_time
        elapsed_batch = now - self.last_batch_time

        if elapsed_startup > self.max_startup_sec:
            self.dump_diagnostics(reason=f"Startup timeout exceeded ({elapsed_startup:.1f}s > {self.max_startup_sec:.1f}s)")
            raise TrainingException(
                message=f"Training startup timed out after {elapsed_startup:.1f} seconds at stage '{self.current_stage}'",
                details="Detailed thread stack traces and GPU state dumped to startup_diagnostics.log",
            )

        if elapsed_batch > self.max_batch_fetch_sec:
            self.dump_diagnostics(reason=f"Batch fetch timeout exceeded ({elapsed_batch:.1f}s > {self.max_batch_fetch_sec:.1f}s)")
            raise TrainingException(
                message=f"DataLoader batch fetch hung for {elapsed_batch:.1f} seconds at sample '{self.current_filename}'",
                details="Detailed thread stack traces and GPU state dumped to startup_diagnostics.log",
            )

    def dump_diagnostics(self, reason: str = "Manual Trigger") -> Path:
        """Dump thread stack traces, system RAM/CPU, GPU state, and telemetry to startup_diagnostics.log."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "startup_diagnostics.log"

        gpu_snap = GPUDiagnostics.snapshot()

        lines = [
            "=" * 70,
            "AETHEL VISION - STARTUP TIMEOUT DIAGNOSTIC DUMP",
            "=" * 70,
            f"Timestamp:              {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Trigger Reason:         {reason}",
            f"Current Stage:          {self.current_stage}",
            f"Total Startup Time:     {time.time() - self.start_time:.2f} seconds",
            f"Last Batch Fetch Time:  {time.time() - self.last_batch_time:.2f} seconds",
            "-" * 70,
            "TELEMETRY & DATALOADER STATE:",
            f"  Workers Count:        {self.num_workers}",
            f"  Dataset Size:         {self.dataset_size}",
            f"  Batch Size:           {self.batch_size}",
            f"  Cache Status:         {self.cache_status}",
            f"  Current File Name:    {self.current_filename}",
            f"  Current Annotation:  {self.current_annotation}",
            f"  Last Success Sample:  {self.last_successful_sample}",
            "-" * 70,
            "HARDWARE & SYSTEM STATE:",
            f"  GPU Name:             {gpu_snap.get('gpu_name')}",
            f"  VRAM Allocated:       {gpu_snap.get('vram_allocated_mb')} MB",
            f"  VRAM Reserved:        {gpu_snap.get('vram_reserved_mb')} MB",
            f"  CUDA Sync Time:       {gpu_snap.get('cuda_sync_time_ms')} ms",
            f"  System RAM Used:      {gpu_snap.get('ram_used_gb')} GB",
            f"  CPU Utilization:      {gpu_snap.get('cpu_pct')} %",
            "-" * 70,
            "THREAD DUMP & STACK TRACES:",
        ]

        for thread_id, frame in sys._current_frames().items():
            lines.append(f"\n--- Thread ID: {thread_id} ---")
            lines.extend(traceback.format_stack(frame))

        lines.append("\n" + "=" * 70)

        content = "\n".join(lines)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Also save copy in root for immediate user access
        try:
            with open("startup_diagnostics.log", "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass

        logger.error(f"Saved diagnostic dump to {log_path}")
        return log_path


__all__ = ["StartupWatchdog"]
