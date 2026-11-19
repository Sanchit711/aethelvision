# Aurelix Detector Architecture Specification (v0.1)

## Executive Overview

**Aurelix Detector** is a proprietary, high-performance, real-time object detection model architecture designed specifically for edge deployment on Windows, Ubuntu, and NVIDIA Jetson Orin platforms.

It features a decoupled anchor-free detection paradigm with Distribution Focal Loss (DFL) edge regression, multi-scale feature pyramid aggregation (PAFPN), and native TensorRT engine export capability.

---

## Overall Detector Pipeline

```
Input Image (B, 3, 640, 640)
             │
             ▼
┌───────────────────────────┐
│     Backbone (Stem)       │  Stem (Focus 4x downsampling)
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│   Backbone (CSPStages)    │  Extracts P3 (stride 8), P4 (stride 16), P5 (stride 32)
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│       Neck (PAFPN)        │  Top-down (FPN) & Bottom-up (PANet) multi-scale fusion
└────────────┬──────────────┘
             │ (P3, P4, P5 Features)
             ▼
┌───────────────────────────┐
│   Head (DecoupledDetect)  │  Parallel Classification & Box Regression (DFL) heads
└────────────┬──────────────┘
             │
      ┌──────┴──────────────────────┐
      │                             │
      ▼ (Training)                  ▼ (Inference)
┌───────────┐                 ┌───────────┐
│ Loss      │ (CIoU+DFL+      │ Postproc  │ (Decoder + NMS)
│ Compute   │  FocalLoss)     │ & Export  │
└───────────┘                 └───────────┘
```

---

## Component Responsibilities

### 1. Backbone (`aurelix_detector/backbone/`)
- **Stem (`stem.py`)**: Initial convolution and downsampling block converting 3-channel input images into initial high-dimensional feature maps ($640\times 640 \rightarrow 160\times 160$).
- **Convolutional Feature Blocks (`conv.py`)**: Strided and depthwise separable convolutions for efficient spatial reduction.
- **Residual Bottlenecks (`residual.py`)**: Shortcut connections preventing gradient degradation in deep layers.
- **Cross Stage Partial Blocks (`csp.py`)**: Partitioned feature routing reducing FLOPs while retaining gradient quality across stages.
- **Attention Modules (`attention.py`)**: Squeeze-and-excitation and spatial attention masks focusing compute on salient object regions.

### 2. Neck (`aurelix_detector/neck/`)
- **FPN (`fpn.py`)**: Top-down feature pyramid propagating rich high-level semantic context to fine-grained shallow layers.
- **PAFPN (`pafpn.py`)**: Bottom-up path aggregation shortening information paths between low-level localization cues and high-level features.
- **Feature Fusion (`fusion.py`)**: Channel alignment and multi-scale concatenation modules.

### 3. Detection Head (`aurelix_detector/head/`)
- **Decoupled Detection Head (`detect.py`)**: Anchor-free decoupled head splitting classification branches from bounding box distance regression branches across scales P3, P4, and P5.
- **Coordinate Decoder (`decoder.py`)**: Converts raw anchor-free regression logits into absolute image spatial coordinates.

### 4. Loss Computation (`aurelix_detector/losses/`)
- **IoU Loss (`iou.py`)**: CIoU (Complete IoU) and GIoU loss for scale-invariant bounding box shape optimization.
- **Focal Loss (`focal.py`)**: Varifocal / Focal loss resolving foreground-background class imbalance.
- **Distribution Focal Loss (`dfl.py`)**: Continuous probability distribution regression for precise object boundary location.

### 5. Post-Processing (`aurelix_detector/postprocess/`)
- **Decode (`decode.py`)**: Converts raw head outputs into confidence scores and normalized bounding box coordinates.
- **NMS (`nms.py`)**: Class-aware Non-Maximum Suppression (NMS) filtering redundant overlapping candidate boxes.

### 6. Export Pipeline (`aurelix_detector/export/`)
- **ONNX Export (`onnx.py`)**: Graph tracing and dynamic axes export to ONNX standard format.
- **TensorRT Compiler (`tensorrt.py`)**: Compiles ONNX graphs into hardware-accelerated `.engine` binaries.
- **Quantization (`quantization.py`)**: Post-training INT8 calibration and precision quantization.

---

## Future Implementation Roadmap

1. **Phase 1 (Skeleton)**: Complete package layout and public interface contracts. *(Current)*
2. **Phase 2 (Layers & Backbone)**: Implement PyTorch tensor operations for Stem, Conv, Residual, and CSP blocks.
3. **Phase 3 (Neck & Head)**: Implement PAFPN multi-scale fusion and Decoupled anchor-free detection head.
4. **Phase 4 (Losses & Training)**: Implement CIoU, DFL, and Focal loss functions integrated with `trainer/`.
5. **Phase 5 (Post-processing & Export)**: Implement NMS CUDA acceleration, ONNX export tracing, and TensorRT engine compilation.
