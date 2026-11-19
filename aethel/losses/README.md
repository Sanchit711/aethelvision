# Aurelix Losses Submodule (`aurelix.losses`)

## Purpose
The **Losses Submodule** provides modular, registered loss formulations for object detection and computer vision models.

## Available Loss Formulations

| Loss Class | Target Objective | Key Parameters |
| :--- | :--- | :--- |
| **`BboxIoULoss`** | Bounding box spatial regression | `loss_type`: `'iou'`, `'giou'`, `'diou'`, `'ciou'` |
| **`FocalLoss`** | Class imbalance classification | `alpha`: `0.25`, `gamma`: `2.0`, `label_smoothing`: `0.0` |
| **`BCEWithLogitsLoss`** | Binary classification | `pos_weight`: `1.0` |
| **`DistributionFocalLoss`** | DFL regression over probability bins | `reg_max`: `16` |
| **`CompositeDetectionLoss`** | Weighted multi-task loss orchestrator | `box_weight`: `7.5`, `cls_weight`: `0.5`, `dfl_weight`: `1.5` |

## Usage Example

```python
from aurelix.losses import CompositeDetectionLoss

# Instantiate loss composer from YAML config spec
loss_fn = CompositeDetectionLoss(config={
    "box_weight": 7.5,
    "cls_weight": 0.5,
    "dfl_weight": 1.5,
    "iou_type": "ciou"
})

# Compute loss breakdown
total_loss, loss_dict = loss_fn(predictions, targets)
print("Total Loss:", total_loss)
print("Breakdown:", loss_dict)
```
