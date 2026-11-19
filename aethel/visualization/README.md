# Aurelix Visualization Submodule (`aurelix.visualization`)

## Purpose
The **Visualization Submodule** renders bounding box overlays, class labels, and confidence percentage banners onto image and video frames.

## Usage Example

```python
from aurelix.visualization import Visualizer

viz = Visualizer()
annotated_img = viz.draw(image, prediction_result)
viz.save(image, prediction_result, output_path="runs/predict/annotated.jpg")
```
