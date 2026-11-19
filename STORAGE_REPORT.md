# AURELIX Enterprise Storage Analysis Report

**Generated Date:** 2026-08-07 11:24:24  
**Total Workspace Disk Usage:** `5.21 GB`  
**Recoverable Space:** `82.99 MB`  

## 1. Storage Breakdown by Asset Type

| Asset Category | Disk Space | Percentage |
|---|---|---|
| Model Checkpoints (`.pt`, `.onnx`) | `5.90 GB` | High |
| Validation & Overlay Images | `79.29 MB` | Medium |
| Model Exports | `296.52 MB` | Medium |
| Generated Reports | `713.23 KB` | Low |
| System & Execution Logs | `848.51 KB` | Low |
| PyCache & Temp Caches | `82.99 MB` | Low |

## 2. Experiment Size Metrics

- **Average Experiment Size:** `67.49 MB`
- **Largest Experiment:** `exp_20260806_104014` (`829.03 MB`)
- **Oldest Experiment:** `validation_preview` (`2026-07-25 00:42:24`)
- **Newest Experiment:** `exp_20260806_174741` (`2026-08-06 17:47:41`)

## 3. Storage Optimization Recommendations

1. Run **Vacuum Workspace** to purge `82.99 MB` of temporary caches.
2. Empty `.trash/` directory to reclaim estimated recoverable space.
3. Compress older experiments to `.zip` in `archive/`.

---
*(c) 2026 AURELIX Technologies Inc. Storage Analyzer v1.0*
