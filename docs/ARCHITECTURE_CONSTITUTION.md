# Mhyra Vision Framework
Enterprise Computer Vision Framework  
**Version:** 1.0  
**Status:** Architecture Constitution  

---

## 1. Vision

The **Mhyra Vision Framework** is built as a completely independent, enterprise-grade computer vision platform. 

* **Complete Component Ownership**: The core objective of Mhyra is **NOT** to wrap an existing third-party detection engine or build an application on top of external frameworks. The objective is to design, implement, and own every single core AI component—including backbone layers, necks, detection heads, loss formulations, assignment algorithms, post-processing filters, dataset parsers, training orchestrators, and export compilers.
* **Unified AI Platform**: Mhyra is architected to evolve into a unified, enterprise-scale vision platform capable of supporting diverse computer vision modalities—ranging from object detection and instance segmentation to vision-language models and edge inference—under one standardized, robust codebase.

---

## 2. Core Principles

Mhyra is governed by ten fundamental engineering principles that ensure architectural integrity, performance, and long-term viability:

1. **Framework Independence**: Zero dependence on high-level third-party vision frameworks. Mhyra owns its entire execution path from raw data ingestion to hardware-accelerated inference.
2. **Modular Architecture**: Decoupled, single-responsibility subsystems communicating strictly through abstract interface contracts. Subsystems can be swapped or extended without cascading side effects.
3. **Enterprise Scalability**: Architected for scale across both training (multi-GPU, distributed data-parallel training) and deployment (edge devices, serverless cloud endpoints, microservices).
4. **Maintainability**: Clear namespace isolation, self-documenting APIs, high readability, and strict adherence to design patterns to reduce technical debt over long lifecycle horizons.
5. **Testability**: Comprehensive unit, integration, numerical parity, and regression test suites. Every module must be testable in isolation using deterministic mocks and synthetic data.
6. **Performance**: Optimized compute graphs, minimal memory allocations, CUDA/TensorRT acceleration, vectorized operations, and low-latency post-processing primitives.
7. **Hardware Agnostic Design**: Modular compute backends allowing seamless execution across CPU, CUDA, TensorRT, ONNX Runtime, and specialized Edge NPUs without modifying core model logic.
8. **Reproducibility**: Fully deterministic training pipelines featuring explicit seed configuration, versioned configuration management, and audit-trailed artifact generation.
9. **Clean Architecture**: Domain logic is decoupled from infrastructure and delivery mechanisms. Inner domain layers contain zero direct references to external libraries or I/O operations.
10. **Extensibility**: Plug-and-play architecture enabling developers to integrate new backbones, custom loss functions, novel heads, or proprietary post-processing routines effortlessly.

---

## 3. Dependency Policy

### Allowed Dependencies

Mhyra permits dependencies **only** on foundational, low-level mathematical, hardware acceleration, and runtime primitives:

* **Language**: Python (`>=3.10`), Standard Python Libraries (`pathlib`, `typing`, `dataclasses`, `argparse`, etc.)
* **Deep Learning Primitives**: `torch`, `torchvision`
* **Numerical & Data Processing**: `numpy`, `pillow`, `opencv-python`, `pyyaml`, `pydantic`
* **System & Diagnostics**: `psutil`, `matplotlib`
* **Runtimes & Hardware Acceleration**: `onnx`, `onnxruntime`, `tensorrt`, `CUDA`, `cuDNN`
* **Testing & Quality**: `pytest`, `black`, `ruff`, `mypy`

#### Why These Dependencies Are Acceptable
These allowed libraries represent fundamental building blocks, system runtimes, mathematical primitives, and hardware driver interfaces. They provide low-level tensor math, GPU memory management, raw image I/O, and compilation targets without imposing model topologies, loss formulations, training loops, or dataset structures. They serve as the canvas upon which Mhyra builds its proprietary vision framework.

---

### Forbidden Dependencies

Mhyra must **NEVER** depend on third-party object detection or computer vision application frameworks. Explicitly forbidden dependencies include:

* **Ultralytics** (YOLOv5, YOLOv8, YOLOv9, YOLOv10, etc.)
* **Detectron2** (Facebook AI Research)
* **MMDetection** / **MMEngine** / **OpenMMLab**
* **YOLO wrappers** or third-party PyTorch-YOLO repositories
* **Framework-specific training engines** (e.g., PyTorch Lightning, Catalyst)
* **Framework-specific dataset loaders & augmentors**
* **Framework-specific model exporters**

#### Research vs. Implementation Policy
* **Allowed & Encouraged**: Rigorous study of academic research papers, mathematical formulas, arXiv preprints, and open algorithm descriptions.
* **Strictly Forbidden**: Copy-pasting code, importing modules, or wrapping third-party detection frameworks. All implementations in Mhyra must be authored natively within the `mhyra` namespace.

---

## 4. Ownership Policy

Every major computer vision component must belong explicitly to the `mhyra` top-level namespace hierarchy:

* `mhyra.datasets`: Dataset parsers, augmentations, and data loader pipelines.
* `mhyra.models`: Backbones, necks, detection heads, and unified model assemblies.
* `mhyra.layers`: Building-block layers (convolutions, attention, CSP blocks, normalization).
* `mhyra.losses`: Loss functions (IoU/GIoU/CIoU/DIoU, Focal Loss, DFL, classification loss).
* `mhyra.trainer`: Training orchestrators, learning rate schedulers, checkpoints, and distributed loops.
* `mhyra.metrics`: Precision, Recall, mAP@50, mAP@50:95, and confusion matrix evaluators.
* `mhyra.export`: ONNX export, TensorRT compilation, quantization, and calibration tools.
* `mhyra.inference`: High-speed edge/server inference wrappers and engine drivers.
* `mhyra.utils`: Helper utilities, spatial bounding box math, NMS, and logger primitives.

**Zero External Framework Ownership**: No third-party framework shall own, manage, or execute any part of Mhyra's training, evaluation, post-processing, or inference logic.

---

## 5. Repository Architecture

The codebase follows a clean, single-responsibility folder structure:

```text
mhyra/
├── datasets/     # Dataset parsers (YOLO, COCO, VOC), augmentations, data loaders
├── models/       # Model assembly definitions, task heads, neck networks, backbones
├── layers/       # Core neural network building blocks (Conv, CSPStage, CBAM, SPPF)
├── losses/       # Target assignment (TAL), Bbox IoU loss, Focal loss, DFL loss
├── trainer/      # Distributed training engine, optimization loops, callbacks, checkpointing
├── metrics/      # Evaluation metrics (mAP, Precision, Recall, Confusion Matrix, Error Analysis)
├── configs/      # Model architectures, hyperparameter configs, dataset specs (YAML/Pydantic)
├── utils/        # Vectorized box math, Anchor generation, NMS, Logger, System helpers
├── inference/    # Real-time inference pipelines, batching queues, stream decoders
├── export/       # ONNX, TensorRT, TorchScript exporters & numerical parity verifiers
├── deployment/   # C++ / Rust edge deployment wrappers, C-API headers, runtime drivers
├── tests/        # Comprehensive unit, integration, performance, and parity test suites
├── docs/         # Architectural constitutions, design docs, API references, tutorials
└── scripts/      # Standalone automation scripts, benchmarks, verification tools
```

### Directory Responsibilities

| Directory | Responsibility |
| :--- | :--- |
| **`datasets`** | Implements robust dataset readers, data validation, mosaic/mixup augmentations, and PyTorch `DataLoader` integrations. |
| **`models`** | Combines backbones, necks, and heads into complete, executable neural networks (`MhyraDetector`, etc.). |
| **`layers`** | Maintains reusable, atomic layer primitives (convolutions, spatial attention, feature fusion). |
| **`losses`** | Computes loss functions and handles dynamic target assignment (Task-Aligned Assigner, Hungarian Matcher). |
| **`trainer`** | Manages training lifecycles, epoch loops, mixed-precision (AMP), EMA weight updating, and multi-GPU DDP. |
| **`metrics`** | Calculates standardized evaluation metrics, mAP curves, and detailed error breakdown reports. |
| **`configs`** | Stores declarative, versioned YAML configuration files and Pydantic schema validators. |
| **`utils`** | Contains math primitives, bounding box transformations, Non-Maximum Suppression (NMS), and loggers. |
| **`inference`** | Encapsulates production inference engines for stream processing and real-time execution. |
| **`export`** | Transforms PyTorch models into optimized execution formats (ONNX Opset 17+, TensorRT `.engine`). |
| **`deployment`** | Houses low-level C++/C-API headers and hardware runtime integration utilities. |
| **`tests`** | Houses test suites ensuring numerical parity, zero memory leaks, and high code coverage. |
| **`docs`** | Technical documentation, design specifications, and architecture constitutions. |
| **`scripts`** | CLI tools, benchmarking suites, verification drivers, and automated tasks. |

---

## 6. Modular Design Rules

Every subsystem in Mhyra must be **independently replaceable** through dependency inversion and clean interface abstractions:

* **Backbone**: Swap out CSPBackbone for Vision Transformer (ViT), EfficientNet, or MobileNet without modifying the Neck or Head.
* **Neck**: Swap PAFPN for FPN, BiFPN, or Feature Reuse Necks without altering Backbone or Head layers.
* **Loss**: Swap BboxIoULoss (CIoU) for GIoU, DIoU, or SIoU without modifying model architecture.
* **Optimizer**: Swap AdamW for SGD or Lion without modifying the Trainer engine.
* **Trainer**: Swap local single-GPU trainer for Distributed Data Parallel (DDP) trainer without modifying model or dataset classes.
* **Inference Engine**: Swap PyTorch execution backend for ONNX Runtime or TensorRT without altering application integration logic.
* **NMS**: Swap standard PyTorch NMS for Batched NMS, Matrix NMS, or C++ custom plugins without modifying the Detection Head.
* **Exporter**: Swap ONNX exporter for TorchScript or CoreML without touching model weights or neural code.

**Rule**: Replacing any component must **NEVER** require editing the codebase of any other component. Subsystems interact strictly through abstract interface contracts.

---

## 7. Model Philosophy

* **Not a YOLO Clone**: Mhyra is a general-purpose vision framework. While its initial object detector may draw inspiration from modern anchor-free YOLO design patterns, Mhyra is **NOT** a YOLO clone.
* **Multi-Architecture Support**: The framework architecture is designed to support a wide array of computer vision tasks and model families natively:
  * **Anchor-Free Detectors** (Decoupled Heads, DFL, TAL)
  * **Transformer-Based Detectors** (RT-DETR, Deformable DETR)
  * **Instance & Semantic Segmentation** (Mask heads, Proto-heads)
  * **Keypoint & Pose Estimation**
  * **Optical Character Recognition (OCR)**
  * **Multi-Object Tracking (MOT)**
  * **Vision-Language Models (VLM)**
  * **Face Recognition & Biometrics**

The core infrastructure (datasets, training, export, inference) must accommodate all these architectures without necessitating architectural redesigns.

---

## 8. Coding Standards

All code within Mhyra must strictly adhere to enterprise-grade software design standards:

1. **No Magic Numbers**: All hyperparameter constants, layer channel widths, stride ratios, and numerical thresholds must be explicitly declared as named constants or loaded via versioned `.yaml` configuration models.
2. **No Duplicated Code (DRY)**: Reusable operations, mathematical formulas, and spatial coordinate transformations must be centralized in utility modules (`mhyra.utils`).
3. **SOLID Principles**:
   * **Single Responsibility**: Each class/module must have exactly one reason to change.
   * **Open/Closed**: Software entities are open for extension, but closed for modification.
   * **Liskov Substitution**: Derived components (e.g., custom backbones) must be fully substitutable for their base abstractions.
   * **Interface Segregation**: Clients should not be forced to depend on methods they do not use.
   * **Dependency Inversion**: High-level modules must depend on abstractions, not concrete implementations.
4. **Clean Architecture**: Domain logic is decoupled from external hardware drivers, APIs, and frameworks.
5. **Object-Oriented Design (OOD)**: Clear encapsulate-extend-polymorph patterns applied across model layers, dataset readers, and exporters.
6. **Readable Code Over Clever Code**: Prioritize code clarity, explicit variable names, and explicit control flow over overly concise or obscure trickery.
7. **PEP8 & Static Typing**: Enforced PEP8 guidelines via `black` (line length 100) and `ruff`, with strict static type hints checked via `mypy`.
8. **Structured Logging**: Zero `print()` statements in production code. All logs must use the centralized `mhyra.logger`.

---

## 9. Testing Policy

No feature is considered complete or ready for production without comprehensive test coverage. Every module must include unit and integration tests.

### Required Test Suites

1. **Unit Tests**: Test every isolated component (layers, loss functions, box transformations) in isolation using synthetic data tensors.
2. **Integration Tests**: Verify end-to-end interactions between datasets, backbones, necks, detection heads, and target assigners.
3. **Regression Tests**: Guard against metric degradation or breaking API changes across model updates.
4. **Performance Tests**: Benchmark memory usage, GPU VRAM allocation, throughput (FPS), and latency distributions across execution backends.
5. **Model Validation**: Ensure model numerical stability, gradient flow, and loss convergence during training iterations.
6. **Dataset Validation**: Validate bounding box normalization, coordinate boundary checks, label format parsing, and corruption resilience.
7. **Export Validation**: Verify ONNX and TensorRT exports pass strict numerical parity checks against PyTorch evaluation predictions (`max_abs_error < 1e-4`, `cosine_similarity > 0.9999`).
8. **Inference Validation**: Test real-time stream decoding, thread safety, and batch inference queues.

---

## 10. AI Development Rules

All future AI coding agents, autonomous subagents, and human contributors modifying Mhyra **MUST** obey these strict guidelines:

1. **Never Introduce Ultralytics**: Do not add `ultralytics` to requirements, do not import `ultralytics`, and do not call Ultralytics CLI or Python APIs.
2. **Never Wrap Ultralytics**: Do not build adapters, wrappers, or bridges around Ultralytics or third-party YOLO engines.
3. **Never Regenerate Copied Framework Code**: Do not copy-paste code snippets from third-party framework repositories.
4. **Generate Original PyTorch Implementations**: Write native, clean, self-contained PyTorch modules directly under the `mhyra` namespace.
5. **Maintain Modularity**: Ensure new features are introduced via clean interface implementations without coupling unrelated subsystems.
6. **Maintain Scalability**: Design code to handle high-throughput edge inference and distributed multi-GPU training seamlessly.
7. **Maintain Enterprise Quality**: Follow strict static typing, explicit docstrings, error handling, and zero magic numbers.
8. **Document Major Architectural Decisions**: Record design choices, mathematical derivations, and trade-offs in `docs/`.
9. **Avoid Unnecessary Dependencies**: Refuse adding new external packages unless absolutely necessary and permitted by the Dependency Policy.

---

## 11. Commercial Policy

Mhyra Vision Framework is designed explicitly for commercial enterprise deployment.

* **Permissive Licensing**: Prefer foundational dependencies released under permissive, commercially friendly licenses (Apache 2.0, MIT, BSD 3-Clause).
* **Avoid Restrictive Copyleft Licensing**: Avoid introducing third-party packages licensed under viral copyleft licenses (such as GPL / AGPL) that mandate proprietary application source code disclosure or restrict commercial distribution.
* **100% Core IP Ownership**: By authoring and owning all high-level framework code, training pipelines, loss formulations, and inference engines natively within the `mhyra` namespace, Mhyra retains total intellectual property ownership and commercial freedom.

---

## 12. Performance Philosophy

Mhyra is architected for maximum inference throughput and ultra-low latency across heterogeneous compute targets:

* **Hardware Backends**:
  * **GPU** (NVIDIA CUDA, TensorRT)
  * **CPU** (Intel OpenVINO, ONNX Runtime CPU)
  * **Embedded Edge AI** (NVIDIA Jetson AGX / Orin / Xavier, Edge NPUs)
  * **Cloud Infrastructure** (Serverless microservices, Triton Inference Server)
* **Execution Paradigms**:
  * **Batch Inference**: Dynamic batching for high-throughput cloud batch processing.
  * **Streaming Inference**: Low-latency frame-by-frame processing for live RTSP camera feeds.
  * **Distributed Inference**: Multi-card, multi-node inference load balancing.
* **Future Hardware Adaptability**: Adding support for new hardware accelerators (e.g., Apple Metal / CoreML, Qualcomm Neural Processing SDK) must require implementing a new backend driver under `mhyra.export` / `mhyra.inference` without altering core neural models.

---

## 13. Long-Term Roadmap

Mhyra's vision is structured across progressive, modular release milestones:

### **Version 1.0 — Object Detection**
* Native anchor-free decoupled object detector (`MhyraDetector`).
* Distribution Focal Loss (DFL) and Task-Aligned Assignment (TAL).
* Proprietary PAFPN neck and CSP-CBAM backbone.
* Native training engine (`mhyra.trainer`), dataset loaders (YOLO/COCO/VOC), and evaluation metrics (`mAP@50`, `mAP@50:95`).
* Production export pipelines (ONNX Opset 17+, TensorRT `.engine`) with numerical parity verification.

### **Version 2.0 — Tracking**
* Real-time Multi-Object Tracking (MOT) subsystem (`mhyra.tracking`).
* Native implementation of Kalman filtering, Hungarian data association algorithms, and appearance embedding associations (ByteTrack / DeepSORT primitives).
* State persistence across live RTSP camera feeds and multi-camera stream topologies.

### **Version 3.0 — Segmentation**
* Instance and semantic segmentation capabilities (`mhyra.segmentation`).
* Native prototype mask heads, spatial mask loss functions, and polygon box-to-mask decoders.
* High-speed mask post-processing and ONNX export compatibility.

### **Version 4.0 — Optical Character Recognition (OCR)**
* End-to-end text detection and recognition pipelines (`mhyra.ocr`).
* Scene text localization backbones and sequence recognition models (CTC loss, attention decoders).
* Specialized preprocessing for licence plate recognition, document parsing, and industrial asset reading.

### **Version 5.0 — Face Recognition**
* Enterprise biometric face analysis pipeline (`mhyra.face`).
* Ultra-lightweight face detection, facial landmark alignment, and metric embedding generation.
* Vector database indexing support for high-scale identity matching and access control.

### **Version 6.0 — Vision-Language Models (VLM)**
* Multimodal vision-language capabilities (`mhyra.vlm`).
* Zero-shot open-vocabulary object detection, visual question answering (VQA), and text-prompted target localization.
* Unified image-text feature projection spaces.

### **Version 7.0 — Enterprise Rule Engine**
* Complex spatio-temporal event evaluation engine (`mhyra.rules`).
* Declarative rule definitions for intrusion detection, line crossing, dwell time calculation, loitering, and perimeter breach monitoring.
* High-frequency alert dispatching, webhook notifications, and real-time telemetry streaming.

### **Version 8.0 — Unified AI Platform**
* End-to-end edge-to-cloud computer vision intelligence platform (`mhyra.platform`).
* Multi-tenant microservice orchestration, centralized camera stream management, active model updating, and automated retraining pipelines.
* Complete enterprise control plane with unified SDK, gRPC/REST APIs, and web dashboards.

---

## 14. Final Engineering Statement

> **FORMAL DECLARATION**  
> Every future architectural decision, codebase modification, feature implementation, and AI coding agent contribution **MUST** comply with this document.  
> Whenever an implementation detail, pull request, or short-term shortcut conflicts with these principles, **the principles declared in this Constitution take absolute precedence.**
