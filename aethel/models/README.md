# Aurelix Models Submodule (`aurelix.models`)

## Purpose
The **Models Submodule** defines neural network model assemblies, modular backbones, feature pyramid necks, and decoupled detection heads for object detection and computer vision tasks.

## Submodule Architecture

```text
aurelix/models/
├── base.py             # BaseModel abstract class
├── detector.py         # Unified AurelixDetector model assembly
├── backbone/
│   ├── stem.py         # DetectorStem (4x downsampling stem)
│   ├── csp.py          # CSPBackbone (P3, P4, P5 multi-scale feature extractor)
│   └── __init__.py
├── neck/
│   ├── pafpn.py        # PAFPN (Path Aggregation Feature Pyramid Network)
│   └── __init__.py
└── head/
    ├── decoupled.py    # DecoupledDetectHead (Decoupled classification & DFL box regression)
    └── __init__.py
```

## Model Scaling Variants

| Scale Variant | Depth Multiple | Width Multiple | Target Use Case |
| :--- | :--- | :--- | :--- |
| **`aurelix_tiny`** | `0.33` | `0.25` | Ultra-lightweight edge AI & microcontrollers |
| **`aurelix_small`** | `0.33` | `0.50` | Embedded edge AI (NVIDIA Jetson Nano/Xavier) |
| **`aurelix_medium`** | `0.67` | `0.75` | Balanced edge & real-time server inference |
| **`aurelix_large`** | `1.00` | `1.00` | High-accuracy server & GPU deployment |
| **`aurelix_xlarge`** | `1.33` | `1.25` | Maximum accuracy benchmark variant |

## Tensor Feature Flow

$$\text{Input } (B, 3, H, W) \xrightarrow{\text{Backbone}} (P_3, P_4, P_5) \xrightarrow{\text{PAFPN Neck}} (N_3, N_4, N_5) \xrightarrow{\text{Decoupled Head}} [N_{3\text{out}}, N_{4\text{out}}, N_{5\text{out}}]$$

- **P3 / N3**: Stride 8 $\to$ Shape $\left(B, C_{p3}, H/8, W/8\right)$
- **P4 / N4**: Stride 16 $\to$ Shape $\left(B, C_{p4}, H/16, W/16\right)$
- **P5 / N5**: Stride 32 $\to$ Shape $\left(B, C_{p5}, H/32, W/32\right)$

Output shape for each scale level $i$:
$$\left(B, \text{num\_classes} + 4 \times \text{reg\_max}, H_i, W_i\right)$$

## Usage Example

```python
import torch
from aurelix.models import AurelixDetector

# Instantiate model variant
model = AurelixDetector(num_classes=80, scale="small")
model.eval()

# Dummy RGB input tensor (Batch=1, Channels=3, Height=640, Width=640)
x = torch.randn(1, 3, 640, 640)

# Forward pass returning raw prediction tensors
raw_outputs = model(x)

for i, out in enumerate(raw_outputs):
    print(f"Scale {i+1} Output Tensor Shape:", out.shape)
    # Scale 1 (N3): (1, 144, 80, 80)
    # Scale 2 (N4): (1, 144, 40, 40)
    # Scale 3 (N5): (1, 144, 20, 20)
```
