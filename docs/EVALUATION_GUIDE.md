# Mhyra Vision Framework — Evaluation Guide

Provides instructions for running quantitative object detection evaluation metrics (`mAP@50`, `mAP@50:95`, Precision, Recall, F1 score).

## Evaluation Example

```python
from mhyra.metrics import DetectionEvaluator

evaluator = DetectionEvaluator(num_classes=80)
evaluator.update(predictions, targets)
metrics = evaluator.compute()
evaluator.save_json(metrics, "runs/eval/report.json")
```
