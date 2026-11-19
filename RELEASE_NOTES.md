# AURELIX Enterprise Vision Framework v1.4.0-RC1 - Release Notes

**Release Date:** August 6, 2026  
**Build Target:** Production Release Candidate (v1.4.0-RC1)  
**Status:** STABLE & APPROVED  

---

## Executive Overview

AURELIX Enterprise Vision Framework (`MhyraVision`) v1.4.0-RC1 represents a production-ready enterprise release candidate. This release culminates a 12-phase audit and runtime hardening program covering static code analysis, runtime performance optimization, memory safety, training stabilization, dataset integrity checking, security hardening, cross-platform validation, and automated self-healing.

All 169 unit and integration tests execute with zero failures across CPU and CUDA compute backends.

---

## Key Highlights & Architectural Enhancements

### 1. Security & Serialization Hardening
- **Safe Checkpoint Unpickling**: Updated all `torch.load` calls across `checkpoint.py`, `experiment.py`, `detector.py`, and `launcher/managers.py` to enforce `weights_only=True` by default, protecting enterprise deployments against arbitrary code execution exploits from untrusted checkpoint binaries.
- **Safe Yaml & Path Validation**: Audited all YAML configurations and file system traversals, eliminating unsafe deserialization vectors and ensuring path traversal protection.

### 2. PyTorch 2.x API Modernization
- **AMP Autocast & GradScaler Refactoring**: Replaced legacy `torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast` calls with modern, unified `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')` interfaces while maintaining seamless backward compatibility for legacy environments.

### 3. Subsystem Completeness & Import Resolution
- **Dataset Factory Architecture**: Implemented `DatasetFactory` in `mhyra/datasets/factory.py` with registry lookups and automatic format detection.
- **Collator Harmonization**: Exported `detection_collate_fn` backward-compatibility alias alongside `BatchCollator` in `mhyra/datasets/collator.py`.

### 4. Enriched Enterprise Exception Telemetry
- **Self-Diagnosing Exceptions**: `MhyraException` now captures UTC timestamp, thread ID, process ID, RAM usage (MB), CPU utilization (%), VRAM allocated (MB), and GPU device name upon instantiation.

### 5. Automated Dataset Health Reporting
- **Multi-Format Export**: Implemented automated generation of `dataset_health.html`, `dataset_health.json`, and `dataset_health.csv` via `DatasetValidator.export_health_reports()`.

---

## Verification & Test Results Summary

| Test Category | Total Tests | Passed | Failed | Status |
| :--- | :---: | :---: | :---: | :---: |
| Unit Tests (Models & Layers) | 38 | 38 | 0 | PASSED |
| Dataset & DataLoader Pipelines | 42 | 42 | 0 | PASSED |
| Trainer & Optimization Subsystems | 45 | 45 | 0 | PASSED |
| Deployment & Exporters (ONNX/TorchScript) | 24 | 24 | 0 | PASSED |
| Platform Ecosystem & Workspace Integration | 20 | 20 | 0 | PASSED |
| **TOTAL** | **169** | **169** | **0** | **100% SUCCESS** |

---

## Supported Operating Systems & Environments
- **Windows 10 / 11** (Native PowerShell, CMD, PyTorch 2.0+)
- **Linux** (Ubuntu 20.04 / 22.04 LTS, RedHat Enterprise Linux 8/9)
- **WSL2** (Windows Subsystem for Linux 2 with CUDA passthrough)
- **Jetson Orin / Embedded Devices** (ARM64, CUDA 11.8 / 12.x)

---

## Deployment & Verification Commands

To run full validation suite:
```bash
pytest
```

To run offline model validation:
```bash
python validate.py --weights checkpoints/best.pt --dataset dataset_micro --device cpu
```

To export model to ONNX:
```bash
python export.py --weights checkpoints/best.pt --format onnx
```
