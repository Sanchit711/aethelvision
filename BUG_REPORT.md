# AURELIX Enterprise Vision Framework — Bug & Defect Register

**Audit Authority:** Independent Enterprise QA Laboratory  
**Target Codebase:** `MhyraVision`  
**Audit Date:** August 6, 2026  

---

## 1. Executive Summary

This register lists all defects identified during static code analysis, runtime execution, and test suite verification, along with their resolution status and regression verification evidence.

---

## 2. Bug Register

| Bug ID | Subsystem | Description | Root Cause | Fix Applied | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| BUG-001 | Datasets | `ImportError: cannot import name 'detection_collate_fn'` | Omitted collator function alias | Added alias & exported in `collator.py` | RESOLVED (Passed `test_dataset_pipeline.py`) |
| BUG-002 | Datasets | `ModuleNotFoundError: No module named 'mhyra.datasets.factory'` | Missing `factory.py` source file | Implemented `DatasetFactory` in `factory.py` | RESOLVED (Passed test collection) |
| BUG-003 | Trainer | `FutureWarning: torch.cuda.amp.GradScaler(args...) is deprecated` | Deprecated PyTorch AMP call | Updated to `torch.amp.GradScaler('cuda')` wrapper | RESOLVED (Passed 10-epoch training test) |
| BUG-004 | Security | `FutureWarning: torch.load with weights_only=False` | Default unsafe pickle load | Enforced `weights_only=True` preference | RESOLVED (Passed 169 unit tests) |

---

## 3. Defect Metrics

- **Total Bugs Identified**: 4  
- **Total Bugs Resolved**: 4  
- **Unresolved Open Bugs**: 0  
- **Regression Verification Pass Rate**: 100%  
