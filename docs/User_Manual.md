# AURELIX Enterprise AI Vision Platform — User Manual

**Version**: 1.0.0 Commercial Release  
**Edition**: Enterprise Commercial Suite  
**Author**: AURELIX Technologies Inc.  

---

## 1. Introduction & Product Overview
AURELIX is a commercial-grade, indigenous PyTorch object detection framework and computer vision intelligence platform. It provides end-to-end model training, real-time dataset management, automated hyperparameter advisory, ONNX/TensorRT export engines, offline challenge-response licensing, and interactive Rich console interfaces.

---

## 2. Quick Start & Execution

Launch the primary platform console from terminal or PowerShell:

```bash
python aurelix.py
```

Alternatively:
```bash
python mhyra.py
```

---

## 3. Platform Modules & Centers

### 3.1 Training Center
- **Train New Model**: Step-by-step wizard selecting dataset, model architecture (MhyraNet-TINY to XLARGE), batch size, and optimizer.
- **Resume Training**: Auto-discovers `.pt` checkpoints in `checkpoints/` or cloud storage to continue interrupted training runs.
- **Fine Tune Model**: Adapts a pre-trained base checkpoint onto a target custom dataset.

### 3.2 Dataset Center
- **Dataset Manager**: Discovers local and cloud datasets (`data/new_person_detection`, `dataset_micro`, COCO).
- **Validation**: Performs structural, image readability, and bounding-box boundary verification.
- **Health Score**: Analyzes train/val split ratios and class balance.

### 3.3 Model Registry
- Manages candidate and master model checkpoints with parameter counts, file sizes, and mAP@0.50 accuracy metrics.

### 3.4 Deployment & Export Center
- Exports trained models to **ONNX** (`exports/model.onnx`), **TensorRT**, and **TorchScript** formats.

---

## 4. Offline License Management
AURELIX features a zero-trust offline challenge-response license system:
1. Navigate to **License Manager** (`[16]`).
2. Generate your unique **Request Code**.
3. Submit the code to your enterprise administrator to receive your cryptographically signed **Activation Key**.

---
*(c) 2026 AURELIX Technologies Inc. All Rights Reserved.*
