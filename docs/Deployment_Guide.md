# AURELIX Enterprise AI Vision Platform — Deployment Guide

**Version**: 1.0.0 Commercial Release  

---

## 1. Deployment Targets
AURELIX supports deployment across 4 target platforms:

1. **Windows Standalone**: Portable executable bundle with embedded dependencies.
2. **Linux Server (Ubuntu / RHEL)**: Python wheel or native system service.
3. **NVIDIA Jetson Edge (Orin / Xavier / Nano)**: TensorRT FP16 optimized build.
4. **Docker Container**: OCI-compliant container image built via `Dockerfile`.

---

## 2. Docker Deployment

Build and launch the container:
```bash
docker build -t aurelix:v1.0.0 -f dist/AURELIX_Enterprise_v1.0.0_Docker_20260729/Dockerfile .
docker run --gpus all -it aurelix:v1.0.0
```

---
*(c) 2026 AURELIX Technologies Inc. All Rights Reserved.*
