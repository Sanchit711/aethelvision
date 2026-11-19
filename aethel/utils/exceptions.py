"""Centralized Exception Hierarchy for Aethel Vision Framework.

Defines custom domain exceptions for dataset loading, model operations,
training orchestrators, export verification, inference, configuration, registry operations,
workspace management, checkpoint compatibility, artifact verification, cache, and visualization.
"""

import datetime
import os
import sys
import threading
from typing import Any, Dict, Optional


def get_system_telemetry() -> Dict[str, Any]:
    """Gather real-time system telemetry for enriched exception diagnostics."""
    telemetry: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "thread_id": threading.get_ident(),
        "process_id": os.getpid(),
        "python_version": sys.version.split()[0],
        "os": sys.platform,
    }
    try:
        import psutil
        proc = psutil.Process()
        telemetry["ram_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
        telemetry["cpu_percent"] = proc.cpu_percent()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            telemetry["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            telemetry["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return telemetry


class AethelException(Exception):
    """Base exception class for all Aethel Vision Framework errors."""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        root_cause: Optional[str] = None,
        location: Optional[str] = None,
        recommended_fix: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
        self.root_cause = root_cause or details or message
        self.location = location or "Framework Engine"
        self.recommended_fix = recommended_fix or "Check logs and execution parameters."
        
        ctx = get_system_telemetry()
        if context:
            ctx.update(context)
        self.context = ctx

    def __str__(self) -> str:
        res = f"[{self.__class__.__name__}] {self.message}"
        if self.details:
            res += f" | Details: {self.details}"
        if self.location:
            res += f" | Location: {self.location}"
        if self.recommended_fix:
            res += f" | Recommended Fix: {self.recommended_fix}"
        if self.context:
            res += f" | Context: {self.context}"
        return res


# --- Workspace & System Exceptions ---

class WorkspaceException(AethelException):
    """Raised when workspace initialization, run indexing, or directory migration fails."""
    pass


class CleanupException(WorkspaceException):
    """Raised when workspace cleanup, zero-byte file removal, or corrupted run purging fails."""
    pass


class CacheException(WorkspaceException):
    """Raised when dataset cache, pycache, or temporary tensor buffer operations fail."""
    pass


class DiskSpaceException(WorkspaceException):
    """Raised when disk space drops below safety thresholds during training execution."""
    pass


# --- Experiment & Artifact Exceptions ---

class ExperimentException(AethelException):
    """Raised when experiment folder creation, lifecycle tracking, or indexing fails."""
    pass


class ArtifactVerificationException(AethelException):
    """Raised when a generated experiment artifact fails physical existence or integrity verification."""
    pass


class VisualizationException(AethelException):
    """Raised when validation overlays, comparison quad-views, or metric plots fail rendering."""
    pass


# --- Dataset Exceptions ---

class DatasetException(AethelException):
    """Raised when dataset loading, parsing, validation, or transformation fails."""
    pass


class DatasetFormatException(DatasetException):
    """Raised when dataset format, annotation schemas, or directory layouts are invalid."""
    pass


class DatasetInterfaceException(DatasetException):
    """Raised when dataset __getitem__ or collate_fn violates framework interface contracts."""
    pass


# --- Model & Training Exceptions ---

class ModelException(AethelException):
    """Raised when neural network instantiation, layer construction, or forward pass fails."""
    pass


class TrainingException(AethelException):
    """Raised when training orchestration, optimization, backpropagation, or checkpointing fails."""
    pass


class BatchFormatException(TrainingException):
    """Raised when batch data from DataLoader cannot be normalized into canonical batch format."""
    pass


class InvalidBatchException(TrainingException):
    """Raised when batch data contains invalid shapes, empty tensors, or corrupt annotations."""
    pass


class OOMException(TrainingException):
    """Raised when GPU VRAM or system RAM memory is exhausted during execution."""
    pass


class CUDAOOMException(OOMException):
    """Explicit CUDA Out Of Memory exception."""
    pass


class DeviceMismatchException(TrainingException):
    """Raised when model and batch tensors reside on mismatched compute devices (e.g. CPU vs CUDA)."""
    pass


class TensorDeviceMismatchException(DeviceMismatchException):
    """Raised when input tensors or layer tensors reside on mismatched compute devices."""
    pass


class EMADeviceMismatchException(DeviceMismatchException):
    """Raised when EMA shadow model tensors reside on a different device than the target model tensors."""
    pass


class InvalidShapeException(TrainingException):
    """Raised when tensor dimensions or matrix shapes fail compatibility checks."""
    pass


class ShapeMismatchException(InvalidShapeException):
    """Explicit tensor shape mismatch exception."""
    pass


class LossNaNException(TrainingException):
    """Raised when non-finite (NaN / Inf) values are produced in loss calculation."""
    pass


class LossException(LossNaNException):
    """Explicit loss computation exception."""
    pass


# --- Checkpoint Exceptions ---

class CheckpointException(TrainingException):
    """Raised when checkpoint serialization, saving, loading, or integrity verification fails."""
    pass


class CheckpointNotFoundException(CheckpointException):
    """Raised when requested model checkpoint or weights file does not exist."""
    pass


class CheckpointCompatibilityException(CheckpointException):
    """Raised when checkpoint model architecture, num_classes, or depth/width mismatches resume config."""
    pass


class SchedulerException(TrainingException):
    """Raised when learning rate scheduler setup or stepping fails."""
    pass


class OptimizerException(TrainingException):
    """Raised when optimizer instantiation, parameter registration, or stepping fails."""
    pass


class ValidationException(TrainingException):
    """Raised when evaluation metrics computation or validation loop execution fails."""
    pass


class TrainingInterruptedException(TrainingException):
    """Raised when training session is prematurely interrupted by signal or user."""
    pass


# --- Export, Metric, & Utility Exceptions ---

class ExportException(AethelException):
    """Raised when model serialization, ONNX export, or TensorRT compilation fails."""
    pass


class InferenceException(AethelException):
    """Raised when real-time stream decoding, batch prediction, or engine inference fails."""
    pass


class ConfigurationException(AethelException):
    """Raised when YAML config parsing, validation, or schema verification fails."""
    pass


class RegistryException(AethelException):
    """Raised when module registration, lookup, or duplication resolution fails."""
    pass


class MetricException(AethelException):
    """Raised when metric computation or report generation fails."""
    pass


class DeploymentException(AethelException):
    """Raised when hardware backend initialization or edge deployment runtime fails."""
    pass


class UnknownRuntimeException(AethelException):
    """Fallback exception for unknown runtime errors."""
    pass


class ExceptionClassifier:
    """Diagnostic classifier for training and execution errors."""

    @staticmethod
    def classify(
        exc: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> AethelException:
        """Classify generic Python/PyTorch exceptions into domain-specific AethelException types with diagnostic metadata."""
        if isinstance(exc, AethelException):
            if context and not exc.context:
                exc.context = context
            return exc

        msg = str(exc).lower()
        exc_type = type(exc).__name__.lower()
        ctx = context or {}

        # 1. EMA Device Mismatch classification
        if isinstance(exc, EMADeviceMismatchException) or ("ema" in msg and ("device" in msg or "same device" in msg or "cuda" in msg or "cpu" in msg)):
            return EMADeviceMismatchException(
                message="EMA device mismatch detected between shadow tensors and model parameters.",
                details=str(exc),
                root_cause="Shadow EMA tensors reside on a different compute device than model parameters.",
                location="aethel.trainer.ema.ModelEMA",
                recommended_fix="Call `ema.sync_device(model)` or ensure model parameters are moved to GPU before EMA updates.",
                context=ctx,
            )

        # 2. Tensor Device Mismatch classification
        if isinstance(exc, TensorDeviceMismatchException) or (
            ("input type" in msg and "weight type" in msg)
            or "expected all tensors to be on the same device" in msg
            or "device mismatch" in msg
            or ("cuda" in msg and "cpu" in msg and "device" in msg)
        ):
            return TensorDeviceMismatchException(
                message="Tensor device mismatch detected between model and input tensors.",
                details=str(exc),
                root_cause="Batch input tensors and model weights reside on different devices.",
                location="aethel.trainer.runner.BatchRunner",
                recommended_fix="Use `move_to_device(batch, device)` to ensure batch tensors match model compute device.",
                context=ctx,
            )

        # 3. CUDA GPU Out Of Memory (OOM) classification
        if (
            "out of memory" in msg
            or "cuda out of memory" in msg
            or "torch.cuda.outofmemoryerror" in msg
            or exc_type == "outofmemoryerror"
        ):
            return CUDAOOMException(
                message="CUDA Out Of Memory error detected during training execution.",
                details=str(exc),
                root_cause="Requested GPU VRAM allocation exceeds available physical GPU memory.",
                location="PyTorch CUDA Allocator / Forward Pass",
                recommended_fix="Reduce batch size, decrease image resolution, enable AMP precision, or clear CUDA cache.",
                context=ctx,
            )

        # 4. Shape & Dimension Mismatch
        if "size mismatch" in msg or "shape mismatch" in msg or "dimension mismatch" in msg or "mat1 and mat2 shapes" in msg:
            return ShapeMismatchException(
                message="Tensor shape or dimension mismatch detected during forward/loss computation.",
                details=str(exc),
                root_cause="Input batch dimensions fail matrix multiplication or channel compatibility checks.",
                location="Model / Loss Computation",
                recommended_fix="Verify input resolution, dataset channels, and model head class count config.",
                context=ctx,
            )

        # 5. Non-finite Loss (NaN / Inf)
        if "nan" in msg or "inf" in msg or "non-finite" in msg:
            return LossException(
                message="Non-finite NaN/Inf value encountered in loss calculation.",
                details=str(exc),
                root_cause="Gradients or activations overflowed producing NaN/Inf loss values.",
                location="aethel.losses.composite",
                recommended_fix="Lower initial learning rate, apply gradient clipping, or check dataset bounding box scaling.",
                context=ctx,
            )

        # 6. Checkpoint Compatibility or Exception
        if "incompatible checkpoint" in msg or "architecture mismatch" in msg:
            return CheckpointCompatibilityException(
                message="Incompatible model architecture or class count in checkpoint metadata.",
                details=str(exc),
                root_cause="Checkpoint metadata does not match current model configuration.",
                location="aethel.trainer.checkpoint.CheckpointManager",
                recommended_fix="Use `ResumeManager.rebuild_model_and_config()` to automatically rebuild model from checkpoint metadata.",
                context=ctx,
            )

        if "checkpoint" in msg or "no such file" in msg or "file not found" in msg:
            if "no such file" in msg or "file not found" in msg or "not found" in msg:
                return CheckpointNotFoundException(
                    message="Checkpoint file not found.",
                    details=str(exc),
                    root_cause="Requested model checkpoint or weights file does not exist on disk.",
                    location="aethel.trainer.checkpoint.CheckpointManager",
                    recommended_fix="Verify checkpoint file path integrity or fallback to previous valid checkpoint.",
                    context=ctx,
                )
            return CheckpointException(
                message="Checkpoint file error or unreadable state payload.",
                details=str(exc),
                root_cause="Checkpoint file is missing, corrupt, or contains invalid keys.",
                location="aethel.trainer.checkpoint.CheckpointManager",
                recommended_fix="Verify checkpoint file path integrity or fallback to previous valid checkpoint.",
                context=ctx,
            )

        # 7. Dataset Exception
        if "dataset" in msg or "dataloader" in msg or "corrupt" in msg:
            return DatasetFormatException(
                message="Dataset formatting or parsing error.",
                details=str(exc),
                root_cause="Dataset directory structure or annotation YAML files failed validation.",
                location="aethel.datasets",
                recommended_fix="Run `DatasetValidator` to check bounding boxes, image files, and YAML annotations.",
                context=ctx,
            )

        return UnknownRuntimeException(
            message=f"Runtime error encountered: {type(exc).__name__}",
            details=str(exc),
            root_cause=f"Unhandled exception type: {type(exc).__name__}",
            location="Framework Execution Loop",
            recommended_fix="Review stacktrace logs for underlying system failure.",
            context=ctx,
        )


__all__ = [
    "AethelException",
    "WorkspaceException",
    "CleanupException",
    "CacheException",
    "DiskSpaceException",
    "ExperimentException",
    "ArtifactVerificationException",
    "VisualizationException",
    "DatasetException",
    "DatasetFormatException",
    "DatasetInterfaceException",
    "ModelException",
    "TrainingException",
    "BatchFormatException",
    "InvalidBatchException",
    "OOMException",
    "CUDAOOMException",
    "DeviceMismatchException",
    "TensorDeviceMismatchException",
    "EMADeviceMismatchException",
    "InvalidShapeException",
    "ShapeMismatchException",
    "LossNaNException",
    "LossException",
    "CheckpointException",
    "CheckpointNotFoundException",
    "CheckpointCompatibilityException",
    "SchedulerException",
    "OptimizerException",
    "ValidationException",
    "TrainingInterruptedException",
    "ExportException",
    "InferenceException",
    "ConfigurationException",
    "RegistryException",
    "MetricException",
    "DeploymentException",
    "UnknownRuntimeException",
    "ExceptionClassifier",
]
