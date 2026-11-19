"""Visual Report Generation Subsystem for Aethel Vision Framework.

Generates loss curves, precision-recall curves, confusion matrices, and learning rate schedule plots.
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image, ImageDraw

from aethel.utils.logger import get_logger

logger = get_logger("aethel.visualization.reports")


class VisualReporter:
    """Reporter for plotting training progress curves and evaluation charts."""

    @staticmethod
    def plot_loss_curves(
        history: Dict[str, List[float]],
        output_path: Union[str, Path] = "runs/plots/loss_curves.png",
    ) -> Path:
        """Generate loss curves canvas image and save to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create PIL drawing canvas
        w, h = 600, 400
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Draw axis lines
        draw.line([(50, 50), (50, h - 50)], fill=(0, 0, 0), width=2)
        draw.line([(50, h - 50), (w - 50, h - 50)], fill=(0, 0, 0), width=2)
        draw.text((60, 20), "Training Loss Curves", fill=(0, 0, 0))

        # Plot loss metrics
        colors = [(220, 50, 50), (50, 120, 220), (50, 180, 50)]
        for idx, (metric_name, vals) in enumerate(history.items()):
            if not vals:
                continue
            color = colors[idx % len(colors)]
            max_v = max(vals) if max(vals) > 0 else 1.0
            pts = []
            for i, v in enumerate(vals):
                x = 50 + int((i / max(len(vals) - 1, 1)) * (w - 100))
                y = (h - 50) - int((v / max_v) * (h - 100))
                pts.append((x, y))

            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)
            draw.text((w - 150, 30 + idx * 20), f"- {metric_name}", fill=color)

        canvas.save(path)
        logger.info(f"Saved visual loss curves to {path}")
        return path


__all__ = ["VisualReporter"]
