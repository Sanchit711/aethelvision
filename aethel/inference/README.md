# Aurelix Inference Submodule (`aurelix.inference`)

## Purpose
The **Inference Submodule** provides high-throughput, model-agnostic prediction engines, multi-scale DFL decoders, class-aware/agnostic Non-Maximum Suppression (NMS), video stream drivers, and unified `PredictionResult` objects.

## Submodule Architecture

```text
aurelix/inference/
├── base.py             # BasePredictor abstract interface
├── result.py           # PredictionResult dataclass container
├── decoder.py          # PredictionDecoder & GridGenerator
├── postprocess.py      # NonMaximumSuppression & SoftNMS
├── predictor.py        # AurelixPredictor primary engine
├── stream.py           # StreamingPredictor & VideoInferenceEngine
└── README.md
```

## Usage Example

```python
from aurelix.inference import AurelixPredictor

# Instantiate predictor engine
predictor = AurelixPredictor(model=model, conf_threshold=0.25, iou_threshold=0.45)

# Execute inference on image file or array
result = predictor.predict("assets/sample.jpg")

print(f"Detected {len(result)} objects:")
print("Boxes:", result.boxes)
print("Scores:", result.scores)
print("Classes:", result.classes)
print("Labels:", result.labels)
```
