"""Video and Stream Processing Engine for Aethel Vision Framework.

Provides StreamingPredictor for real-time video files, USB cameras, RTSP, RTMP, and HTTP streams.
"""

from typing import Any, Generator, Optional, Union

import numpy as np

from aethel.inference.predictor import AethelPredictor
from aethel.inference.result import PredictionResult
from aethel.utils.logger import get_logger

logger = get_logger("aethel.inference.stream")


class StreamingPredictor:
    """Stream processing engine for real-time video streams and camera feeds."""

    def __init__(
        self,
        predictor: AethelPredictor,
        frame_skip: int = 0,
        fps_limit: Optional[float] = None,
    ) -> None:
        """Initialize StreamingPredictor.

        Args:
            predictor: AethelPredictor instance.
            frame_skip: Number of frames to skip between inference calls.
            fps_limit: Optional max FPS processing rate limit.
        """
        self.predictor = predictor
        self.frame_skip = frame_skip
        self.fps_limit = fps_limit

    def stream_frame(self, frame: np.ndarray) -> PredictionResult:
        """Process a single video frame array."""
        result = self.predictor.predict(frame)
        return result if isinstance(result, PredictionResult) else result[0]

    def process_generator(
        self,
        frame_generator: Generator[np.ndarray, None, None],
    ) -> Generator[PredictionResult, None, None]:
        """Yield prediction results over a video frame stream generator."""
        frame_idx = 0
        for frame in frame_generator:
            if self.frame_skip > 0 and frame_idx % (self.frame_skip + 1) != 0:
                frame_idx += 1
                continue

            result = self.stream_frame(frame)
            frame_idx += 1
            yield result


VideoInferenceEngine = StreamingPredictor

__all__ = ["StreamingPredictor", "VideoInferenceEngine"]
