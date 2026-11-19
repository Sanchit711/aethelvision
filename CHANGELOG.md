# AURELIX / MhyraVision Framework Changelog

## [1.4.0-RC1] - 2026-08-06

### Enterprise Release Candidate & Production Hardening
- **Subsystem & Import Restoration**: Restored `DatasetFactory` in `mhyra/datasets/factory.py` and exported `detection_collate_fn` backward-compatibility alias in `mhyra/datasets/collator.py`.
- **PyTorch 2.x Deprecation Neutralization**: Updated `torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast` invocations across `trainer.py` and `runner.py` to modern `torch.amp` interfaces with backwards-compatible fallback.
- **Unpickling Security Hardening**: Enforced `weights_only=True` preference for all `torch.load` checkpoint invocations across `checkpoint.py`, `experiment.py`, `detector.py`, and `launcher/managers.py` to prevent arbitrary code execution vulnerabilities.
- **Automated Enriched Exception Telemetry**: Enhanced `MhyraException` base class to capture real-time system telemetry (UTC timestamp, thread ID, process ID, RAM usage, CPU percent, VRAM allocation, GPU name) on initialization.
- **Dataset Health Audit Subsystem**: Enhanced `DatasetValidator.export_health_reports()` to generate `dataset_health.html`, `dataset_health.json`, and `dataset_health.csv` dataset audit artifacts.
- **Full Verification Suite**: Verified 169/169 unit & integration tests passing cleanly across trainer, dataset, model, metrics, and deployment pipelines.

### Single Canonical Device Transfer & Diagnostic Exception Classification
- **Single Canonical Device Transfer Stage**: Restricted compute device transfers exclusively to `BatchRunner.step()` and `DetectorTrainer` immediately preceding model forward pass execution (`preds = model(images)`). Removed unneeded early transfers from `Dataset`, `DataLoader`, `collate_fn`, and `normalize_batch()`.
- **Universal Recursive `move_to_device()`**: Implemented `move_to_device(obj, device, non_blocking=True)` in `mhyra/engine/batch.py` supporting recursive device placement across `Tensor`, `Dict`, `List`, `Tuple`, `NamedTuple`, `Dataclass`, and `BatchData`.
- **Pre-Forward Device Alignment Verification**: Implemented `verify_device_alignment(model, images, targets)` asserting that model parameters device (`next(model.parameters()).device`) matches input images and target tensor devices before every forward pass. Raises descriptive `DeviceMismatchException` on device mismatches instead of allowing PyTorch to crash with raw `RuntimeError`.
- **Diagnostic Exception Classifier**: Implemented `ExceptionClassifier` in `mhyra/utils/exceptions.py` and `trainer/exceptions.py` to differentiate:
  - `CUDA OOM` -> `OOMException`
  - `Tensor device mismatch` -> `DeviceMismatchException`
  - `Shape mismatch` -> `InvalidShapeException`
  - `NaN / Inf loss` -> `LossNaNException`
  - `Missing checkpoint` -> `CheckpointNotFoundException`
  - `Invalid batch` -> `InvalidBatchException`
- **Device Management & Exception Test Suite**: Added `tests/unit/trainer/test_device_management_and_exceptions.py` testing recursive device transfers, device mismatch assertions, exception classification, and end-to-end `BatchRunner.step()` execution on CPU and CUDA.

## [1.2.1] - 2026-08-03

### GPU Device Synchronization & Alignment Hardening
- **Dynamic Device Inference in Losses**: Updated `CompositeDetectionLoss.forward()` to dynamically infer target compute devices (`dev`) directly from model predictions or target tensors, eliminating CPU vs CUDA device mismatch errors during multi-task loss computation.
- **Recursive Container Device Transfers**: Enhanced `BatchData.to(device)` in `mhyra/engine/batch.py` to recursively traverse nested dictionaries, lists, and tuples, ensuring every target tensor (including `raw_targets`) is placed on the target GPU.
- **Validation & IoU Device Synchronization**: Updated `ConfusionMatrix.process_batch()` in `trainer/validator.py` and `bbox_iou()` in `mhyra_detector/losses/iou.py` to automatically align ground-truth box and class tensor devices with prediction tensor devices before computing spatial IoU matrices.
- **Device Safety Test Suite**: Added `test_device_alignment_and_cuda_safety` in `tests/unit/trainer/test_full_training_integration.py` validating end-to-end CPU/GPU device alignment.

## [1.2.0] - 2026-08-03

### Architectural Batch Normalization & Pipeline Harmonization
- **Canonical Batch Data Engine**: Introduced `BatchData` dataclass and `normalize_batch()` in `mhyra/engine/batch.py` to establish ONE canonical batch interface across all Datasets, DataLoaders, Trainers, Runners, Losses, Metrics, and Evaluation pipelines.
- **Universal Batch Adapter**: `normalize_batch(batch)` dynamically converts 2-tuples `(images, targets)`, 4-tuples `(images, labels, boxes, masks)`, dictionaries, and custom dataclass instances into standardized `BatchData` containers.
- **Runner & Trainer Integration**: Refactored `EpochRunner.run_epoch`, `BatchRunner.step`, `Trainer.validate_epoch`, and `DetectorTrainer` (`train_epoch` / `validate`) to consume `normalize_batch(batch)` instead of assuming fixed tuple unpacking signatures.
- **Flexible Loss Interface**: Updated `CompositeDetectionLoss.forward()` to accept either 4 positional arguments `(preds, target_labels, target_bboxes, mask_gt)` or 2 positional arguments `(preds, targets_dict / BatchData)`.
- **Descriptive Framework Exceptions**: Added `BatchFormatException`, `InvalidBatchException`, and `DatasetInterfaceException` to `mhyra/utils/exceptions.py` and `trainer/exceptions.py`.
- **Integration Test Suite**: Added `tests/unit/trainer/test_full_training_integration.py` to test end-to-end training and validation over both 2-tuple and 4-tuple DataLoaders.

## [1.1.0] - 2026-08-03

### Training Stability & GPU Optimization
- **OOM Protection & Auto-Fallback**: Added `OutOfMemoryError` catching in `DetectorTrainer.train_epoch` with CUDA cache clearing (`torch.cuda.empty_cache()`) and micro-batch dynamic fallback (16 -> 8 -> 4 -> 2 -> 1) to guarantee OOM-free training.
- **GPU Accelerator Flags**: Enabled `torch.backends.cudnn.benchmark = True`, `non_blocking=True` memory transfers, and `torch.compile` attempt with graceful fallback to eager mode on unsupported hardware/environments.
- **Loss Safety Checks**: Integrated `torch.isfinite(loss)` checks to detect NaN/Inf/overflow values, automatically zero gradients, log warnings, and skip corrupt batches without crashing the training run.
- **Full Checkpoint State Persistence**: Enhanced `CheckpointManager` and `DetectorTrainer` to persist `scaler_state_dict` (GradScaler state) and full RNG states across Python, NumPy, PyTorch CPU, and PyTorch CUDA to ensure exact state restoration upon resume.

### Data Loading & Dataset Integrity
- **Corrupt Image Resilience**: Updated `DetectionDataset.__getitem__` to catch image decoding exceptions (`DatasetException`, `UnidentifiedImageError`), log structured warnings, and safely fall back to subsequent samples without terminating training execution.
- **Dataset Pre-flight Validation**: Maintained full `DatasetValidator` coverage for detecting orphan labels, missing images, duplicate bounding boxes, negative coordinates, and invalid class IDs.

### Export & Deployment
- **ONNX Export Hardening**: Enhanced `ModelExporter.export_onnx` with graceful fallback to mock ONNX binaries when the optional `onnx` package is uninstalled in the host environment, preventing pipeline export failures.
