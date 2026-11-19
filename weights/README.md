# AethelNet Pretrained Model Weights Zoo (`weights/`)

This directory houses official pretrained checkpoint files and model weight manifests for the **AethelNet** open-source foundation model family, licensed under **Apache 2.0**.

## Open-Source Weight Manifest (Apache 2.0)

As part of the open-source community release, **AethelNet-Nano (`aethel_n`)** and **AethelNet-Small (`aethel_s`)** models are provided royalty-free under the Apache 2.0 license:

| Model Checkpoint | Modality | Params | GFLOPs | Latency (T4 FP16) | mAP / Metric | File Size | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`aethel_n_clean.pt`** | 2D Bounding Box | 3.2 M | 8.4 | 1.12 ms | 41.2% mAP50-95 | 3.99 MB | **Apache 2.0** |
| **`aethel_n_fp16.pt`** | 2D Bounding Box | 3.2 M | 8.4 | 1.12 ms | 41.2% mAP50-95 | 2.04 MB | **Apache 2.0** |
| **`aethel_s_clean.pt`** | 2D Bounding Box | 9.6 M | 24.8 | 1.84 ms | 47.8% mAP50-95 | 15.3 MB | **Apache 2.0** |
| **`aethel_s_fp16.pt`** | 2D Bounding Box | 9.6 M | 24.8 | 1.84 ms | 47.8% mAP50-95 | 7.69 MB | **Apache 2.0** |
| **`aethelnet_s_seg.pt`** | Instance Segmentation | 11.2 M | 29.4 | 2.21 ms | 41.8% mask mAP | 26.6 MB | **Apache 2.0** |
| **`aethelnet_s_pose.pt`** | Pose Estimation | 10.8 M | 27.6 | 2.05 ms | 53.4% pose mAP | 26.0 MB | **Apache 2.0** |
| **`aethelnet_s_obb.pt`** | Oriented Bounding Box | 10.4 M | 26.2 | 1.95 ms | 50.8% OBB mAP | 25.9 MB | **Apache 2.0** |

> **Note on Enterprise Tiers**: Mid-to-flagship scale models (`aethel_m`, `aethel_l`, and `aethel_x` 114M parameters) are reserved for sovereign enterprise air-gapped clusters and defense teleoperation deployments. Contact `enterprise@aethel.ai` for specialized licensing.

## Checkpoint Payload Standard

Every `.pt` checkpoint contains the canonical metadata envelope:

```python
{
    "version": "1.0.0",
    "architecture": "AethelNet",
    "variant": "small",
    "modality": "detect",  # detect | segment | pose | obb
    "num_classes": 80,
    "model_state_dict": { ... },
    "metrics": {
        "mAP@50": 0.658,
        "mAP@50:95": 0.478
    },
    "license": "Apache 2.0"
}
```

## Loading Pretrained Weights Programmatically

```python
from aethel import AethelDetector

# Instantiate open-source model
detector = AethelDetector.from_pretrained("weights/aethel_s_clean.pt")

# Run inference
results = detector.predict("test_image.jpg", conf=0.25)
```
