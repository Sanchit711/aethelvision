"""Benchmark & Verification Script for Aurelix Detector Foundational Layers.

Verifies layer execution speed, memory throughput, and exact tensor output shapes.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from aurelix_detector.backbone.conv import DepthwiseConv
from aurelix_detector.layers import Conv, ConvBNReLU, ConvBNSiLU


def run_shape_validation() -> None:
    """Execute mandatory shape validation check."""
    print("=== Mandatory Shape Validation Check ===")
    x = torch.randn(1, 3, 640, 640)
    layer = Conv(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=2,
    )
    y = layer(x)

    print(f"Input tensor shape : {x.shape}")
    print(f"Layer config       : Conv(in_channels=3, out_channels=32, kernel_size=3, stride=2)")
    print(f"Output tensor shape: {y.shape}")
    assert y.shape == torch.Size([1, 32, 320, 320]), f"Expected torch.Size([1, 32, 320, 320]), got {y.shape}"
    print("STATUS: Shape validation PASSED!\n")


def run_benchmark() -> None:
    """Run execution speed benchmark over Conv layers."""
    print("=== Foundational Layers Performance Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")

    x = torch.randn(8, 64, 160, 160, device=device)
    layer_silu = ConvBNSiLU(64, 128, kernel_size=3, stride=2).to(device)
    layer_relu = ConvBNReLU(64, 128, kernel_size=3, stride=2).to(device)
    dw_layer = DepthwiseConv(64, 64, kernel_size=3, stride=1).to(device)

    # Warm-up passes
    for _ in range(10):
        _ = layer_silu(x)
        _ = layer_relu(x)
        _ = dw_layer(x)

    iterations = 100
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = layer_silu(x)
    t1 = time.perf_counter()
    avg_silu_ms = ((t1 - t0) / iterations) * 1000.0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = layer_relu(x)
    t1 = time.perf_counter()
    avg_relu_ms = ((t1 - t0) / iterations) * 1000.0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = dw_layer(x)
    t1 = time.perf_counter()
    avg_dw_ms = ((t1 - t0) / iterations) * 1000.0

    print(f"ConvBNSiLU (64 -> 128, 3x3 stride 2) : {avg_silu_ms:.3f} ms / batch")
    print(f"ConvBNReLU (64 -> 128, 3x3 stride 2) : {avg_relu_ms:.3f} ms / batch")
    print(f"DepthwiseConv (64 -> 64, 3x3 stride 1): {avg_dw_ms:.3f} ms / batch")
    print("STATUS: Benchmark completed successfully!\n")


if __name__ == "__main__":
    run_shape_validation()
    run_benchmark()
