# Mhyra Vision - CUDA Environment & Hardware Compatibility Assessment

**Assessment Date:** 2026-07-29T00:34:00+05:30  
**Role:** Senior ML Infrastructure Engineer  
**Audit Scope:** Read-Only Hardware & Software Compatibility Audit  

---

## 1. Hardware & Driver Assessment

| Attribute | Specification | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **GPU Hardware** | NVIDIA GeForce MX350 | `nvidia-smi -q` | Verified |
| **GPU Architecture** | Pascal (GP107) | Device Specs | Verified |
| **Compute Capability** | SM 6.1 | CUDA Hardware DB | Supported |
| **Total VRAM** | 2048 MiB (2 GB GDDR5) | `nvidia-smi` | Verified |
| **Installed Driver** | Version `451.67` | WDDM Driver | **Outdated** |
| **Current Max CUDA Support** | CUDA 11.0 | Driver API Manifest | Verified |

---

## 2. Operating System & Python Compatibility

### Python 3.13 Evaluation
* **Current Python Version:** `3.13.14`
* **PyTorch CUDA Support on Python 3.13 (Windows):** **UNSUPPORTED**
  * Official PyTorch CUDA binary wheels for Windows are built and distributed for Python versions **3.8, 3.9, 3.10, 3.11, and 3.12**.
  * Pre-compiled C++ extension wheels for `torchvision`, `opencv-python`, and `tensorboard` lack stable CUDA Windows binaries on Python 3.13.
  * PyTorch installed in `.venv` resolved to `2.13.0+cpu` because pip fell back to CPU binaries due to missing 3.13 CUDA wheels.

### Windows OS & MX350 GPU Evaluation
* **Windows Support:** **FULLY SUPPORTED** (Windows 10/11 x86_64 has full PyTorch CUDA binary support).
* **MX350 GPU Support:** **FULLY SUPPORTED** (Compute Capability 6.1 is supported in PyTorch CUDA 11.8 and 12.1 builds).

---

## 3. Recommended Software Stack Specification

To establish a production-grade, long-duration training environment on this machine, the following stack is recommended:

| Component | Current Version | Recommended Version | Compatibility Rationale |
| :--- | :--- | :--- | :--- |
| **Python** | `3.13.14` | **`3.10.11`** (or `3.11.9`) | Standard for PyTorch 2.x CUDA pre-compiled wheels & C++ binaries on Windows. |
| **NVIDIA Driver** | `451.67` | **`550.x`** or newer | Driver 451.67 only supports CUDA 11.0 max. PyTorch CUDA 11.8/12.1 requires driver >= 525.60+. |
| **CUDA Toolkit** | None | **`11.8`** (or `12.1`) | Fully supports Pascal architecture (SM 6.1) with long-term stability. |
| **cuDNN** | None | **`8.9.2`** | High-performance convolution and activation primitives for Pascal GPUs. |
| **PyTorch** | `2.13.0+cpu` | **`2.1.2+cu118`** or **`2.3.1+cu118`** | Fully compiled with CUDA 11.8 acceleration for SM 6.1. |
| **TorchVision** | Missing | **`0.16.2+cu118`** or **`0.18.1+cu118`** | Matches PyTorch build with C++ NMS operator bindings. |
| **OpenCV** | Missing | **`opencv-python==4.8.1.78`** | Provides fast image decoding and matrix processing on Python 3.10/3.11. |
| **TensorBoard** | Missing | **`2.15.0`** | Native visualization without gRPC/Protobuf conflicts on Windows. |

---

## 4. Technical Justification & Migration Plan

1. **Python Runtime Downgrade to 3.10 or 3.11:**
   * Python 3.13 is too new for the deep learning C++ binary ecosystem on Windows.
   * Creating a dedicated virtual environment with Python 3.10 or 3.11 provides immediate access to all pre-compiled CUDA wheels.

2. **NVIDIA Display Driver Update:**
   * Installed Driver `451.67` limits CUDA runtime to 11.0. Modern PyTorch builds require CUDA 11.8 or 12.1 (which require driver 452.39+ / 525.60+ / 550+).
   * Updating the driver unlocks full GPU acceleration on the GeForce MX350.

3. **VRAM Optimization Strategy:**
   * GeForce MX350 has 2 GB VRAM. Training detector architectures (`MhyraNet-SMALL`) should use `batch_size=4` or `batch_size=8` with AMP (`fp16`) precision and gradient accumulation to stay under the 2 GB VRAM ceiling.

---

## 5. Final Assessment Verdict

```
REQUIRES DIFFERENT PYTHON VERSION
```

*Summary: The current Python version (3.13.14) lacks PyTorch CUDA wheel availability on Windows. A new environment with Python 3.10 or 3.11 and an updated NVIDIA display driver are required before installing CUDA PyTorch.*
