# Mhyra Vision - Historical Training Environment Audit Report

**Audit Date:** 2026-07-29T00:35:40+05:30  
**Role:** Senior ML Infrastructure Engineer  
**Audit Scope:** Read-Only Audit of Historical Experiments, Logs, Metrics, and Checkpoints  

---

## 1. Executive Summary

A comprehensive read-only audit of all historical execution logs, performance CSV records, experiment summaries, and checkpoint metadata was conducted to determine whether past training runs executed on CPU or CUDA GPU hardware.

The evidence conclusively demonstrates that **all previous runs were executed on CPU**. Although configuration files requested `device: "cuda"`, the runtime environment was forced onto CPU execution because PyTorch was installed as `2.13.0+cpu` (CPU-only build, `torch.cuda.is_available() == False`).

---

## 2. Answers to Specific Audit Inquiries

| # | Inquiry | Findings & Empirical Evidence | Status / Value |
| :--- | :--- | :--- | :--- |
| **1** | **Was training performed on CPU or CUDA?** | Executed entirely on **CPU**. Confirmed by `environment_report.md`, `training_report.md`, and CSV metric tracks. | **CPU** |
| **2** | **Was `torch.cuda.is_available()` logged?** | Yes. Explicitly logged as `CUDA Available: False` in `training_report.md` (Line 6). | **Logged (`False`)** |
| **3** | **Was device recorded as `cpu` or `cuda`?** | Recorded as **`CPU`** in execution logs (`environment_report.md` Line 10: `Compute Device: CPU`). Config specs requested `cuda` but runtime fell back to `cpu`. | **CPU** |
| **4** | **What was the average epoch time?** | **0.04 to 0.05 seconds** per epoch in benchmark metric tracks (`performance_metrics.csv`, `performance_metrics_resume.csv`), and **1.5 seconds** in `training_session_summary.json`. | **0.04 - 0.05 s/epoch** |
| **5** | **Was GPU memory ever allocated?** | **No.** Zero VRAM allocated by PyTorch. Only `cpu_percent` and `ram_percent` were recorded in performance CSV logs. | **No GPU Allocation** |
| **6** | **Did any run report CUDA, cuDNN, or GPU utilization?** | **No.** All reports confirm 0% GPU utilization and `CUDA/cuDNN` unavailable. | **None** |
| **7** | **Which Python interpreter created each checkpoint?** | `.venv\Scripts\python.exe` (Python **3.13.14** 64-bit), as recorded in `crash_recovery_metadata.json` (Line 14). | `Python 3.13.14` |
| **8** | **Which PyTorch version created each checkpoint?** | PyTorch **`2.13.0+cpu`**, as recorded in `environment_report.md` (Line 6). | `2.13.0+cpu` |

---

## 3. Detailed Evidence Matrix

### 1. Environment Log Evidence (`environment_report.md`)
```markdown
- Operating System: Windows-11-10.0.26200-SP0
- Python Version: 3.13.14
- PyTorch Version: 2.13.0+cpu
- Compute Device: CPU
- CPU Topology: 4 Cores / 8 Threads
```

### 2. Single-Image Overfitting Report (`training_report.md`)
```markdown
- Python Version: 3.13.14
- CUDA Available: False
- Compute Device: CPU
```

### 3. Checkpoint Crash Recovery Metadata (`checkpoints/crash_recovery_metadata.json`)
```json
{
  "last_completed_epoch": 1,
  "last_valid_checkpoint": "checkpoints/last.pt",
  "resume_command": ".venv\\Scripts\\python.exe trainer/cli.py --resume checkpoints/last.pt --config configs/training/custom.yaml"
}
```

### 4. Performance Metrics Logs (`performance_metrics.csv` & `performance_metrics_resume.csv`)
* Logged Fields: `epoch`, `epoch_time_sec`, `images_per_sec`, `cpu_percent`, `ram_percent`, `dataloader_time_sec`, `forward_time_sec`, `backward_time_sec`
* Missing Fields: `gpu_percent`, `vram_usage_mb` (Confirming no GPU tracking or allocation occurred).

---

## 4. Final Verdict

```
PREVIOUS RUNS WERE CPU
```
