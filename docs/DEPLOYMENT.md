# Mhyra Detector Multi-Platform Enterprise Deployment Guide

This guide details step-by-step production deployment procedures for **Mhyra Detector** across cloud, desktop, server, and edge environments.

---

## 1. Google Colab (Training & Export Environment)

### Environment Prerequisites
- GPU Runtime enabled (Tesla T4 or L4 GPU)
- Python 3.10+, PyTorch 2.0+

### Setup & Training Workflow
```bash
# 1. Clone Repository & Install Package
git clone https://github.com/mhyra/MhyraVision.git
cd MhyraVision
pip install -e .

# 2. Execute Production Training Loop
python trainer/cli.py train --config configs/train.yaml

# 3. Export Model Weights to ONNX & TensorRT
python export.py --weights checkpoints/best_model.pt --fp16 --dynamic --output exports/best.onnx
```

---

## 2. Windows 11 Desktop / Edge Server (CPU & DirectML)

### Environment Prerequisites
- Windows 11 OS
- Python 3.10+ with `onnxruntime` CPU / DirectML provider

### Deployment Workflow
```bash
# 1. Install Mhyra Vision Package
pip install -e .

# 2. Export Dynamic ONNX Graph
python export.py --weights checkpoints/best_model.pt --dynamic --output exports/best.onnx

# 3. Run Benchmark Verification
python benchmark.py --onnx exports/best.onnx
```

---

## 3. Linux Production Container Server (Docker & TensorRT C++)

### Environment Prerequisites
- Ubuntu 22.04 LTS
- NVIDIA Container Toolkit with CUDA 12.x support

### Container Build & Deployment
```bash
# 1. Pull Official NVIDIA TensorRT Docker Image
docker pull nvcr.io/nvidia/tensorrt:23.09-py3

# 2. Run Interactive Container with GPU Access
docker run --gpus all -it --rm -v $(pwd):/workspace/MhyraVision nvcr.io/nvidia/tensorrt:23.09-py3

# 3. Compile Optimized TensorRT Engine
cd /workspace/MhyraVision
python export.py --weights checkpoints/best_model.pt --fp16 --output exports/best.engine
```

---

## 4. NVIDIA Jetson Orin Nano 8GB (Edge Deployment)

### Target Hardware Specifications
- Jetson Orin Nano 8GB (20 FP16 TFLOPS, 40 INT8 TOPS)
- JetPack 5.1.2 or JetPack 6.0 (Ubuntu 22.04 LTS + TensorRT 8.6+)

### Edge Deployment Procedure
1. Flash Jetson Orin Nano with JetPack 5.1.2/6.0 image using NVIDIA SDK Manager.
2. Transfer exported `best.onnx` model binary to Jetson device.
3. Compile hardware-native TensorRT FP16 / INT8 engine on Jetson using `trtexec`:
```bash
# FP16 Engine Build on Jetson
trtexec --onnx=best.onnx \
        --saveEngine=best_fp16.engine \
        --fp16 \
        --minShapes=images:1x3x320x320 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:4x3x1280x1280

# INT8 Calibration & Engine Build on Jetson
python export.py --weights checkpoints/best_model.pt --int8 --output exports/best_int8.engine
```
4. Deploy using high-speed TensorRT C++ runtime bindings or Python `onnxruntime` CUDA execution provider.

