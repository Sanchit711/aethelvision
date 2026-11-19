# Mhyra Vision Hardware Abstraction Layer (HAL) Documentation

## Overview

The `hal/` package isolates execution runtimes (NVIDIA TensorRT, ONNX Runtime, PyTorch Native) behind a unified, hardware-agnostic interface ([AbstractInferenceEngine](file:///C:/Users/Sanchit/Projects/MhyraVision/hal/base.py#L48-L125)). This decouples core vision tasks, pipeline orchestration, and business logic from target edge hardware details.

---

## Architecture & Supported Runtimes

```
hal/
├── base.py                 # AbstractInferenceEngine & HAL Exceptions
├── buffer.py               # HardwareBufferBinding & BufferManager
├── cache.py                # EngineCache (SHA256 model engine caching)
├── discovery.py            # HardwareDiscovery (CUDA/TRT/DLA feature detection)
├── factory.py              # HALFactory (Auto runtime backend selection)
├── profiler.py             # HALProfiler (Stage latency timing)
├── tensorrt/               # TensorRT runtime backend (Production)
├── onnx/                   # ONNX Runtime backend (Edge/CPU fallback)
└── pytorch/                # PyTorch Native runner (Training/Debug)
```

---

## Key Subsystems

### 1. AbstractInferenceEngine Interface
Inherits from `domain.contracts.IRuntimeContract` enforcing `load_engine()`, `execute()`, and `unload_engine()` execution contracts.

### 2. Runtime Selection Strategy (`HALFactory`)
Automatically chooses the optimal backend class:
- `.engine` files $\rightarrow$ `TensorRTEngine` (Production INT8/FP16 execution)
- `.onnx` files $\rightarrow$ `TensorRTEngine` if TensorRT installed, else `ONNXRuntimeEngine`
- `.pt` / `.pth` files $\rightarrow$ `PyTorchEngine` (Debug/Training runner)

### 3. Engine Caching (`EngineCache`)
Prevents redundant engine binary deserialization by caching active engine contexts in RAM, indexed by SHA256 file digests.

### 4. Hardware Exception Translation
Translates low-level CUDA/TensorRT driver errors into structured domain exceptions:
- `HALExecutionError`: Inference forward pass execution failures.
- `VRAMAllocationError`: Device memory allocation or binding failures.
- `ModelLoadError`: Deserialization or model file reading failures.
