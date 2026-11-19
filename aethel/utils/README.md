# Aurelix Utils Submodule (`aurelix.utils`)

## Purpose
The **Utils Submodule** provides centralized cross-cutting utilities, exception hierarchies, structured logging, tensor math operations, spatial bounding box conversion primitives, and Non-Maximum Suppression (NMS).

## Responsibilities
- Provide centralized `AurelixException` hierarchy and `setup_logger` / `get_logger` drivers.
- Provide vectorized spatial bounding box math (`xywh2xyxy`, `xyxy2xywh`, `box_iou`).
- Implement vectorized Non-Maximum Suppression (NMS).

## Public API
- Exceptions: `AurelixException`, `DatasetException`, `ModelException`, `TrainingException`, `ExportException`, `InferenceException`, `ConfigurationException`, `RegistryException`, `MetricException`, `DeploymentException`.
- Logging: `setup_logger`, `get_logger`.

## Dependencies
- Standard Python libraries (`logging`, `sys`, `pathlib`)
- `torch`, `numpy`

## Future Work
- Add custom C++ extension bindings for ultra-low latency NMS filters.
