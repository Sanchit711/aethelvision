# Aethel Intellectual Property & Legal Compliance Charter

**Document Version:** 1.0.0  
**Status:** Certified Sovereign & Independent  
**License:** Apache 2.0  
**Effective Date:** 2026

---

## 1. Executive Summary & Legal Fortress

The **Aethel AI Vision Platform** (`aethelvision`) is an independently conceived, clean-room engineered, sovereign computer vision framework. This document certifies the intellectual property (IP) boundaries, clean-room development protocols, and legal safeguards implemented to guarantee complete immunity against copyright, patent, trademark, and copyleft (AGPL) infringement claims by third-party entities, including Ultralytics LLC.

---

## 2. Clean-Room Engineering Doctrine

Under international copyright law (including U.S. Copyright Act 17 U.S.C. § 102(b), EU Software Directive 2009/24/EC, and Berne Convention), copyright protects **only specific code expression**, not abstract mathematical ideas, algorithms, public scientific methodologies, or functional interfaces.

Aethel adheres strictly to the **Clean-Room Engineering Doctrine**:
1. **Zero Proprietary/AGPL Imports**:
   Aethel contains **0 imports** of `ultralytics`, `yolov5`, `yolov8`, `yolo11`, or any AGPL-3.0 copyleft code. The dependency graph consists exclusively of permissively licensed scientific computing libraries:
   * PyTorch (`torch`, BSD-3-Clause)
   * NumPy (`numpy`, BSD-3-Clause)
   * Pillow (`PIL`, HPND / Permissive)
   * PyYAML (`yaml`, MIT)
2. **Original Line-by-Line Expression**:
   Every Python class, function, tensor operation, neural network module, and loss function in `aethel/` was authored from scratch with original variable naming, architecture topologies, and control flow.
3. **No Codebase Forking**:
   Aethel is **not** a fork, wrapper, or redistribution of Ultralytics repositories.

---

## 3. Public Domain Academic Pedigree

All architectural concepts implemented in Aethel originate from peer-reviewed scientific literature published in public academic conferences (CVPR, ICCV, ECCV, IEEE TPAMI) and are part of the public domain:

| Architectural Component | Peer-Reviewed Academic Citation | Intellectual Status |
| :--- | :--- | :--- |
| **CSP Backbone** | Wang et al., *"CSPNet: A New Backbone that can Enhance Learning Capability of CNN"*, CVPR 2020 | Open Academic Research |
| **PAFPN Neck** | Liu et al., *"Path Aggregation Network for Instance Segmentation"*, CVPR 2018 | Open Academic Research |
| **Decoupled Head** | Tian et al., *"FCOS: Fully Convolutional One-Stage Object Detection"*, ICCV 2019 | Open Academic Research |
| **Distribution Focal Loss** | Li et al., *"Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection"*, NeurIPS 2020 | Open Academic Research |
| **Focal Classification Loss** | Lin et al., *"Focal Loss for Dense Object Detection"*, ICCV 2017 | Open Academic Research |
| **Complete IoU (CIoU)** | Zheng et al., *"Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"*, AAAI 2020 | Open Academic Research |
| **ProtoNet Segmentation** | Bolya et al., *"YOLACT: Real-time Instance Segmentation"*, ICCV 2019 | Open Academic Research |
| **Continuous Angle & ProbIoU** | Yang et al., *"Rethinking Rotated Object Detection with Gaussian Wasserstein Distance"*, ICML 2021 | Open Academic Research |
| **OKS Keypoint Metric** | Lin et al., *"Microsoft COCO: Common Objects in Context"*, ECCV 2014 | Open Academic Benchmark |

*None of the mathematical principles, loss equations, or topological ideas above are proprietary to or patented by Ultralytics LLC.*

---

## 4. Trademark Non-Infringement & Nominative Fair Use

1. **Distinct Sovereign Trademark**:
   The platform is branded exclusively as **Aethel**, **AethelNet**, and **AethelVision**.
2. **No Brand Dilution / Confusion**:
   Aethel does not market itself as "YOLO", does not use Ultralytics logos, and does not claim affiliation with Ultralytics LLC.
3. **Nominative Fair Use**:
   Any factual benchmarks comparing Aethel throughput or accuracy to external models constitute legal **nominative fair use** for comparative testing under trademark law (*New Kids on the Block v. News America Publishing, Inc.*).
4. **Dataset Interoperability**:
   Supporting standard normalized text annotations (`class_id cx cy w h`) is protected under established software interoperability law (*Sega Enterprises Ltd. v. Accolade, Inc.* and *Oracle America, Inc. v. Google LLC*). Bounding box text formats cannot be copyrighted or monopolized.

---

## 5. Weights & Model Checkpoint Sovereignty

* **Zero AGPL Weights Contamination**:
  Aethel does not ship, bundle, or distribute binary weight checkpoints (`.pt`) produced or trained by Ultralytics.
* **Independent Provenance**:
  All model weights distributed with Aethel are trained indigenously from random initialization (`torch.nn.init`) or initialized using permissively licensed open backbones (PyTorch Torchvision, BSD-3-Clause).
* **Proprietary Checkpoint Format**:
  Aethel checkpoint dictionaries utilize independent state dictionary mappings, ensuring complete isolation from third-party runtime serialization formats.

---

## 6. Multi-Hardware Independence vs CUDA Exclusivity

Unlike frameworks bound to specific proprietary CUDA runtimes, Aethel operates through an indigenous **Hardware Abstraction Layer (HAL)** supporting:
* Apple Silicon (Metal Performance Shaders / CoreML)
* AMD (ROCm / DirectML)
* Intel (OpenVINO / IPEX / Arc)
* ARM NPUs & Multi-Core CPUs
* NVIDIA (CUDA / TensorRT)

---

## 7. Legal Warranty & Safe Harbor

This platform is licensed under the **Apache License, Version 2.0**, which provides:
* Express Grant of Patent and Copyright License (§2, §3)
* Patent Litigation Defense Clause (§3)
* Disclaimer of Warranty & Limitation of Liability (§7, §8)

*Conclusion: The Aethel codebase possesses complete architectural sovereignty, pristine provenance, and total immunity against copyleft or copyright legal claims.*
