# Aurelix Datasets Submodule (`aurelix.datasets`)

## Purpose
The **Datasets Submodule** provides a model-agnostic, universal computer vision dataset pipeline for object detection and future vision modalities. It abstracts dataset parsing, file integrity verification, image loading, memory caching, data augmentations, batch collation, and dataset statistics generation.

## Submodule Architecture

```text
aurelix/datasets/
├── base.py           # Abstract BaseDataset interface
├── detection.py      # Universal DetectionDataset implementation
├── metadata.py       # AnnotationInfo, ImageInfo, DatasetMetadata dataclasses
├── parser.py         # Multi-format DatasetParser & LabelParser (YOLO, VOC, COCO)
├── loader.py         # ImageLoader & AnnotationLoader
├── cache.py          # Memory-efficient DatasetCache
├── transforms.py     # Modular TransformPipeline (Resize, Letterbox, Flips, HSV)
├── validator.py      # DatasetValidator & DatasetIntegrityChecker
├── statistics.py     # JSON-serializable DatasetStatistics
├── collator.py       # PyTorch DataLoader BatchCollator
└── visualization.py  # VisualizationManager (box overlays, grid previews)
```

## Supported Formats & Features
- **Dataset Formats**: YOLO TXT (`[class_id, xc, yc, w, h]`), Pascal VOC XML, COCO JSON, with automatic format detection.
- **Image Extensions**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
- **Validation**: Detects corrupted images, missing annotations, orphan label files, out-of-bounds bounding box coordinates, negative dimensions, invalid class IDs, and duplicate entries.
- **Statistics**: Generates JSON reports detailing image counts, annotation counts, objects per image, class distribution, and bounding box scale breakdown.
- **Augmentation Pipeline**: Configurable via YAML (`Resize`, `Letterbox`, `Horizontal Flip`, `Vertical Flip`, `HSV Color Jitter`, plus hooks for `Mosaic` and `MixUp`).
- **Batch Collate**: `BatchCollator` outputting unified PyTorch tensor shapes `(B, C, H, W)` and `(Total_Objects, 6)` -> `[batch_idx, class_id, x1, y1, x2, y2]`.

## Public API
- `BaseDataset`, `DetectionDataset`
- `DatasetParser`, `LabelParser`
- `DatasetStatistics`, `DatasetValidator`, `DatasetIntegrityChecker`
- `ImageLoader`, `AnnotationLoader`, `DatasetCache`
- `TransformPipeline`, `BatchCollator`, `VisualizationManager`
- `DatasetMetadata`, `ImageInfo`, `AnnotationInfo`

## Usage Example

```python
from aurelix.datasets import DetectionDataset, TransformPipeline, BatchCollator
from torch.utils.data import DataLoader

# Instantiate dataset with YAML config and auto format detection
dataset = DetectionDataset(
    root="datasets/default",
    split="train",
    config={"imgsz": 640, "hflip": 0.5, "hsv_h": 0.015},
    use_cache=True,
)

# DataLoader integration with BatchCollator
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=BatchCollator(),
)

for images, targets in dataloader:
    print("Images Batch Shape:", images.shape)   # (16, 3, 640, 640)
    print("Targets Batch Shape:", targets.shape) # (N, 6) -> [batch_idx, class_id, x1, y1, x2, y2]
    break
```
