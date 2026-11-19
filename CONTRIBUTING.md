# Contributing to AethelVision™

Thank you for your interest in contributing to **AethelVision**! We welcome open-source contributions from researchers, vision engineers, and developers worldwide.

---

## Clean-Room Engineering Policy (Mandatory)

AethelVision is an independent, sovereign computer vision framework licensed under **Apache 2.0**. To protect all contributors and downstream commercial users:

1. **Zero Copyleft Contamination**:
   Never copy, paste, or adapt source code from AGPL-licensed repositories (such as Ultralytics YOLO).
2. **Academic Citations**:
   Any new architectural feature, block, or loss function must be backed by public academic literature (CVPR, ICCV, ECCV, NeurIPS, etc.) and implemented indigenously from scratch.

---

## Development Setup

```bash
# Fork & clone the repo
git clone https://github.com/<your-username>/aethelvision.git
cd aethelvision

# Install editable package and testing dependencies
pip install -e .
pip install pytest
```

---

## Running Unit Tests

Before submitting a Pull Request, ensure all unit tests pass with 100% success rate:

```bash
python -m pytest tests/unit/test_universal_hal.py tests/unit/test_aethelnet_family.py tests/unit/test_aethel_multimodal.py
```

---

## Code Style & Formatting

* Python 3.10+ compatible.
* Type hints on all function and method signatures.
* Use `ruff` and `black` (line length: 100).
* Clean docstrings formatted in Google/Sphinx style.

---

## Pull Request Guidelines

1. Create a descriptive feature branch (`git checkout -b feature/my-new-head`).
2. Add comprehensive unit tests in `tests/unit/`.
3. Commit with clear messages (`git commit -m "feat(pose): add hand keypoint decoder"`).
4. Submit your Pull Request against the `main` branch.
