# AURELIX Enterprise AI Vision Platform — Developer Guide

**Version**: 1.0.0 Commercial Release  
**Target Audience**: Framework Developers & ML Engineers  

---

## 1. Architecture Overview
AURELIX is built following Clean Architecture principles:
- `mhyra/models`: Indigenous PyTorch implementations of CSPDarknet backbones, PANet feature pyramids, and Decoupled Anchor-Free Detection Heads.
- `mhyra/trainer`: Orchestrator fit loop handling autograd, AMP fp16 precision, EMA weights updates, and checkpointing.
- `mhyra/launcher`: Presentation layer, Rich UI console engine, License verification, AI Assistant, and Project Manager.

---

## 2. Model Architecture Definitions
Supported architecture scales defined in `mhyra/models`:
- **MhyraNet-TINY**: 1.8M params (Edge AI / Micro devices)
- **MhyraNet-SMALL**: 4.9M params (Balanced high-speed edge detector)
- **MhyraNet-MEDIUM**: 11.2M params (General purpose detector)
- **MhyraNet-LARGE**: 25.4M params (High accuracy server model)
- **MhyraNet-XLARGE**: 48.1M params (Enterprise maximum precision model)

---

## 3. Running Unit Tests
Execute the full test suite with pytest:
```bash
python -m pytest tests/unit
```

---
*(c) 2026 AURELIX Technologies Inc. All Rights Reserved.*
