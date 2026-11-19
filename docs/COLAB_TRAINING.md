# Google Colab GPU Training Guide for Mhyra Detector

This document provides production instructions, hardware recommendations, and memory specifications for training **Mhyra Detector** on Google Colab GPU instances.

---

## 1. Google Colab GPU Recommendations & Batch Sizes

| GPU Tier | GPU Memory | Precision | Recommended Batch Size | Gradient Accumulation Steps | Virtual Batch Size | Training Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tesla T4** (Free / Pro) | 16 GB | AMP FP16 | **16** | 2 | 32 | ~90-110 hrs (COCO 300 ep) |
| **Tesla L4** (Colab Pro) | 24 GB | AMP FP16 | **32** | 1 | 32 | ~35-45 hrs (COCO 300 ep) |
| **NVIDIA A100** (Colab Pro+) | 80 GB | AMP FP16 | **64** | 1 | 64 | ~8-12 hrs (COCO 300 ep) |

---

## 2. Notebook Setup & Installation

Execute the following shell commands in a Google Colab notebook cell:

```python
# 1. Check GPU Accelerator
!nvidia-smi

# 2. Clone Repository & Install Mhyra Vision Package
!git clone https://github.com/mhyra/MhyraVision.git
%cd MhyraVision
!pip install -e .
```

---

## 3. Training Execution Examples

### Standard GPU Training (Tesla T4 / L4)
```bash
python trainer/cli.py train \
    --config configs/train.yaml \
    --device cuda \
    --precision fp16 \
    --batch 16
```

### High-Throughput A100 GPU Training
```bash
python trainer/cli.py train \
    --config configs/train.yaml \
    --device cuda \
    --precision fp16 \
    --batch 64 \
    --workers 8
```

### Resuming Training from Checkpoint
```bash
python trainer/cli.py train \
    --config configs/train.yaml \
    --resume checkpoints/last.pt
```

---

## 4. VRAM & Memory Optimization Tips
1. **Automatic Mixed Precision (AMP)**: Keep `precision: "fp16"` enabled in `configs/train.yaml` to reduce VRAM memory consumption by ~50%.
2. **Gradient Accumulation**: If running out of VRAM on smaller GPU instances, reduce `batch_size: 8` and set `grad_accum_steps: 4` to maintain an effective batch size of 32.
3. **Structured Log Monitoring**: Monitor real-time progress via `logs/training_session_log.csv` and `logs/training_session_summary.json`.

