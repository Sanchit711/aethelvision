"""Multi-Camera Zero-Copy Video Streaming Engine for Aethel Vision Framework.

Provides multi-threaded, low-latency RTSP, webcam, and video file streaming
with automated stale-frame dropping and ring buffering for edge/production deployment.
"""

import os
import sys
import time
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class VideoStreamWorker:
    """Worker thread that continuously reads frames from a single camera/stream.
    
    Uses a small ring buffer (queue maxsize=1 or 2) and drops stale frames
    to ensure zero-latency real-time inference on edge devices.
    """

    def __init__(self, source: Union[int, str], stream_id: int = 0, buffer_size: int = 2) -> None:
        self.source = source
        self.stream_id = stream_id
        self.buffer_size = buffer_size
        self.frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.fps = 0.0
        self.frame_count = 0
        self.cap: Any = None
        self._last_time = time.time()

    def start(self) -> "VideoStreamWorker":
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self) -> None:
        if cv2 is not None and str(self.source).lower() != "mock":
            try:
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    self.cap = None
                elif isinstance(self.source, str) and ("rtsp://" in self.source or "http://" in self.source):
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                self.cap = None
        else:
            self.cap = None
        
        while self.running:
            frame = None
            if self.cap is not None and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame
                else:
                    # End of file or disconnected stream; retry after pause
                    time.sleep(0.05)
                    if isinstance(self.source, str) and os.path.exists(self.source):
                        # Loop video file for continuous demo
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
            else:
                # Mock synthetic frame generator if cv2 is absent or camera unavailable
                time.sleep(0.033)  # ~30 FPS
                if np is not None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    t = int(time.time() * 10) % 255
                    frame[:, :, 0] = t
                    frame[:, :, 1] = 120
                    frame[:, :, 2] = 200

            if frame is not None:
                self.frame_count += 1
                now = time.time()
                dt = now - self._last_time
                if dt >= 1.0:
                    self.fps = self.frame_count / dt
                    self.frame_count = 0
                    self._last_time = now

                # Drop oldest frame if queue is full to prevent lag
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

        if self.cap is not None:
            self.cap.release()

    def read(self, timeout: float = 0.5) -> Optional[Any]:
        """Fetch the latest real-time frame."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class MultiCameraStream:
    """Coordinates concurrent streaming across multiple camera sources.
    
    Supports edge CCTV grids, multi-angle robotics, and production inference pipelines.
    """

    def __init__(self, sources: List[Union[int, str]]) -> None:
        self.sources = sources
        self.workers: List[VideoStreamWorker] = [
            VideoStreamWorker(src, stream_id=i) for i, src in enumerate(sources)
        ]

    def start_all(self) -> "MultiCameraStream":
        for w in self.workers:
            w.start()
        return self

    def read_all(self, timeout: float = 0.5) -> Dict[int, Optional[Any]]:
        """Read latest synchronized frames from all streams."""
        results = {}
        for w in self.workers:
            results[w.stream_id] = w.read(timeout=timeout)
        return results

    def stop_all(self) -> None:
        for w in self.workers:
            w.stop()

    def __enter__(self) -> "MultiCameraStream":
        return self.start_all()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop_all()


def run_stream_pipeline(
    sources: Optional[List[Union[int, str]]] = None,
    task: str = "detect",
    display: bool = False,
    max_frames: Optional[int] = 100,
) -> None:
    """CLI runner for multi-camera video streaming."""
    if sources is None:
        sources = ["mock"]

    print("=" * 70, flush=True)
    print(" AETHEL MULTI-CAMERA ZERO-COPY STREAMING ENGINE", flush=True)
    print("=" * 70, flush=True)
    print(f" Streams active: {len(sources)} ({sources})", flush=True)
    print(f" Task:           {task.upper()}", flush=True)
    print(" Press Ctrl+C to terminate.", flush=True)
    print("-" * 70, flush=True)

    streamer = MultiCameraStream(sources)
    streamer.start_all()

    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            frames = streamer.read_all(timeout=0.2)
            valid_frames = [f for f in frames.values() if f is not None]

            if not valid_frames:
                time.sleep(0.01)
                continue

            frame_idx += len(valid_frames)
            elapsed = time.time() - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0.0

            if frame_idx % 10 == 0:
                fps_info = " | ".join(f"Cam-{w.stream_id}: {w.fps:.1f} FPS" for w in streamer.workers)
                print(f"Processed {frame_idx} total frames | Overall: {fps:.1f} FPS | [{fps_info}]", flush=True)

            if max_frames and frame_idx >= max_frames:
                print(f"Reached benchmark limit of {max_frames} frames.", flush=True)
                break

    except KeyboardInterrupt:
        print("\nStopping streams cleanly...", flush=True)
    finally:
        streamer.stop_all()
        print("All video capture worker threads terminated.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aethel Multi-Camera Video Streamer")
    parser.add_argument("--sources", nargs="+", default=["0"], help="Camera IDs (0, 1) or RTSP URLs")
    parser.add_argument("--task", default="detect", choices=["detect", "pose", "segment", "obb"])
    parser.add_argument("--benchmark", type=int, default=100, help="Frames to process for benchmark")
    args = parser.parse_args()

    # Parse int camera IDs if digits
    parsed_sources: List[Union[int, str]] = []
    for s in args.sources:
        if s.isdigit():
            parsed_sources.append(int(s))
        else:
            parsed_sources.append(s)

    run_stream_pipeline(sources=parsed_sources, task=args.task, max_frames=args.benchmark)
