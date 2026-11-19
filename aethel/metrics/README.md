# Aurelix Metrics Submodule (`aurelix.metrics`)

## Purpose
The **Metrics Submodule** calculates quantitative performance metrics, mAP curves, confusion matrices, and diagnostic error reports.

## Supported Metrics

- **`Precision`**: Fraction of correct positive predictions.
- **`Recall`**: Fraction of ground-truth objects correctly identified.
- **`F1 Score`**: Harmonic mean of Precision and Recall.
- **`mAP@50`**: Mean Average Precision at IoU threshold 0.50.
- **`mAP@50:95`**: Mean Average Precision averaged across IoU thresholds 0.50 to 0.95.
- **`MetricAggregator`**: Real-time batch loss accumulator across training epochs.

## Usage Example

```python
from aurelix.metrics import DetectionEvaluator

evaluator = DetectionEvaluator(num_classes=80)
evaluator.update(predictions, targets)
metrics = evaluator.compute()
print("mAP@50:", metrics["mAP@50"])
```
